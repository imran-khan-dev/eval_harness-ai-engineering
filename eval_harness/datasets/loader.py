"""Dataset loading utilities for JSON and JSONL formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_harness.core.exceptions import DatasetError, ValidationError
from eval_harness.datasets.models import EvaluationDataset
from eval_harness.datasets.validation import (
    check_duplicate_case_ids,
    compute_dataset_hash,
    validate_cases,
    validate_dataset_structure,
)


class DatasetLoadPolicy:
    """Policy for handling edge cases during dataset loading."""

    def __init__(
        self,
        allow_empty: bool = True,
        require_hash: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize load policy.

        Args:
            allow_empty: Whether to allow datasets with zero cases.
            require_hash: Whether to compute and store content hash.
            strip_whitespace: Whether to strip whitespace from string fields.
        """
        self.allow_empty = allow_empty
        self.require_hash = require_hash
        self.strip_whitespace = strip_whitespace


def load_json_dataset(
    file_path: str | Path,
    policy: DatasetLoadPolicy | None = None,
) -> EvaluationDataset:
    """Load an evaluation dataset from a JSON file.

    The JSON file must contain an object with at minimum:
    - "name": Dataset name (string)
    - "cases": Array of case objects

    Args:
        file_path: Path to the JSON dataset file.
        policy: Optional load policy for edge case handling.

    Returns:
        Validated EvaluationDataset instance.

    Raises:
        DatasetError: If file cannot be read or parsed.
        ValidationError: If dataset structure or content is invalid.
    """
    path = Path(file_path)
    policy = policy or DatasetLoadPolicy()

    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}",
            details={"file": str(path), "error": "file_not_found"},
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(
            f"Invalid JSON in dataset file: {e.msg}",
            details={
                "file": str(path),
                "line": e.lineno,
                "column": e.colno,
                "error": "json_decode",
            },
        ) from e
    except OSError as e:
        raise DatasetError(
            f"Failed to read dataset file: {e}",
            details={"file": str(path), "error": "io_error"},
        ) from e

    return _process_raw_dataset(raw_data, str(path), policy)


def load_jsonl_dataset(
    file_path: str | Path,
    policy: DatasetLoadPolicy | None = None,
) -> EvaluationDataset:
    """Load an evaluation dataset from a JSONL (JSON Lines) file.

    Each line must be a valid JSON object representing a single evaluation case.
    Dataset-level metadata (name, description, etc.) must be provided via the
    first line if it contains a special "_meta" key, or will use defaults.

    JSONL format options:
    1. Pure cases (one per line) - uses filename as name, defaults for rest
    2. First line with "_meta" key containing dataset metadata, followed by cases

    Args:
        file_path: Path to the JSONL dataset file.
        policy: Optional load policy for edge case handling.

    Returns:
        Validated EvaluationDataset instance.

    Raises:
        DatasetError: If file cannot be read or parsed.
        ValidationError: If dataset structure or content is invalid.
    """
    path = Path(file_path)
    policy = policy or DatasetLoadPolicy()

    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}",
            details={"file": str(path), "error": "file_not_found"},
        )

    cases_data: list[dict[str, Any]] = []
    dataset_meta: dict[str, Any] = {}

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        details={
                            "file": str(path),
                            "line": line_num,
                            "column": e.colno,
                            "error": "json_decode",
                        },
                    ) from e

                # Check for metadata record
                if line_num == 1 and "_meta" in record and isinstance(record["_meta"], dict):
                    dataset_meta = record["_meta"]
                    continue

                cases_data.append(record)

    except OSError as e:
        raise DatasetError(
            f"Failed to read dataset file: {e}",
            details={"file": str(path), "error": "io_error"},
        ) from e

    # Build raw dataset structure
    raw_data = {
        "name": dataset_meta.get("name", path.stem),
        "description": dataset_meta.get("description"),
        "version": dataset_meta.get("version", "1.0.0"),
        "metadata": dataset_meta.get("metadata", {}),
        "tags": dataset_meta.get("tags", []),
        "cases": cases_data,
    }

    return _process_raw_dataset(raw_data, str(path), policy)


def _process_raw_dataset(
    raw_data: dict[str, Any],
    file_path: str,
    policy: DatasetLoadPolicy,
) -> EvaluationDataset:
    """Process raw dataset dictionary into validated EvaluationDataset.

    Args:
        raw_data: Raw dataset dictionary.
        file_path: Source file path for error context.
        policy: Load policy for edge case handling.

    Returns:
        Validated EvaluationDataset instance.
    """
    # Validate top-level structure
    validate_dataset_structure(raw_data, file_path)

    # Validate and convert cases
    cases = validate_cases(raw_data["cases"], file_path)

    # Check for duplicate case IDs
    check_duplicate_case_ids(cases, file_path)

    # Check empty dataset policy
    if not policy.allow_empty and len(cases) == 0:
        raise ValidationError(
            "Dataset contains no cases (empty dataset not allowed by policy)",
            details={"file": file_path, "policy": "allow_empty=false"},
        )

    # Build dataset
    dataset_kwargs = {
        "name": raw_data["name"],
        "description": raw_data.get("description"),
        "cases": cases,
        "version": raw_data.get("version", "1.0.0"),
        "metadata": raw_data.get("metadata", {}),
        "tags": raw_data.get("tags", []),
    }
    # Only include id if present and valid
    if "id" in raw_data and raw_data["id"] is not None:
        dataset_kwargs["id"] = raw_data["id"]

    dataset = EvaluationDataset(**dataset_kwargs)

    # Compute and store content hash if required
    if policy.require_hash:
        content_hash = compute_dataset_hash(dataset)
        dataset.metadata["content_hash"] = content_hash

    return dataset
