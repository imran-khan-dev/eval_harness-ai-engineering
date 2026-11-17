"""Dataset validation and hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from eval_harness.core.exceptions import ValidationError
from eval_harness.datasets.models import EvaluationCase, EvaluationDataset


def validate_dataset_structure(data: dict[str, Any], file_path: str) -> None:
    """Validate that the dataset has required top-level fields.

    Args:
        data: Parsed JSON dataset dictionary.
        file_path: Path to the dataset file for error context.

    Raises:
        ValidationError: If required fields are missing or invalid.
    """
    if not isinstance(data, dict):
        raise ValidationError(
            f"Dataset must be a JSON object, got {type(data).__name__}",
            details={"file": file_path, "type": type(data).__name__},
        )

    if "name" not in data:
        raise ValidationError(
            "Dataset missing required 'name' field",
            details={"file": file_path, "missing_field": "name"},
        )

    if "cases" not in data:
        raise ValidationError(
            "Dataset missing required 'cases' field",
            details={"file": file_path, "missing_field": "cases"},
        )

    if not isinstance(data["cases"], list):
        raise ValidationError(
            "Dataset 'cases' must be a list",
            details={"file": file_path, "type": type(data["cases"]).__name__},
        )


def check_duplicate_case_ids(cases: list[EvaluationCase], file_path: str) -> None:
    """Check for duplicate case IDs in the dataset.

    Args:
        cases: List of evaluation cases.
        file_path: Path to the dataset file for error context.

    Raises:
        ValidationError: If duplicate case IDs are found.
    """
    seen_ids: dict[str, int] = {}
    duplicates: list[tuple[str, int, int]] = []

    for idx, case in enumerate(cases):
        case_id_str = str(case.id)
        if case_id_str in seen_ids:
            duplicates.append((case_id_str, seen_ids[case_id_str], idx))
        else:
            seen_ids[case_id_str] = idx

    if duplicates:
        dup_details = [
            {"case_id": cid, "first_index": first, "duplicate_index": dup}
            for cid, first, dup in duplicates
        ]
        raise ValidationError(
            f"Dataset contains {len(duplicates)} duplicate case ID(s)",
            details={"file": file_path, "duplicates": dup_details},
        )


def compute_dataset_hash(dataset: EvaluationDataset) -> str:
    """Compute a deterministic content hash for an evaluation dataset.

    The hash is computed from the canonical JSON representation of the dataset's
    cases, ensuring equivalent datasets produce the same hash regardless of
    JSON formatting or dictionary key ordering.

    Args:
        dataset: The evaluation dataset to hash.

    Returns:
        SHA-256 hex digest of the normalized dataset content.
    """
    # Create a canonical representation of cases for hashing
    canonical_cases = []
    for case in dataset.cases:
        canonical_case = {
            "id": str(case.id),
            "input": case.input,
            "expected_output": case.expected_output,
            "expected_tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "description": tc.description,
                    "required": tc.required,
                    "order": tc.order,
                }
                for tc in case.expected_tool_calls
            ],
            "expected_trajectory": (
                {
                    "steps": case.expected_trajectory.steps,
                    "description": case.expected_trajectory.description,
                    "max_deviation": case.expected_trajectory.max_deviation,
                }
                if case.expected_trajectory
                else None
            ),
            "answerability": {
                "is_answerable": case.answerability.is_answerable,
                "confidence": case.answerability.confidence,
                "requires_tools": case.answerability.requires_tools,
                "required_tools": case.answerability.required_tools,
                "context_requirements": case.answerability.context_requirements,
                "ambiguity_notes": case.answerability.ambiguity_notes,
            },
            "metadata": case.metadata,
            "tags": case.tags,
        }
        canonical_cases.append(canonical_case)

    # Sort cases by ID for deterministic ordering
    canonical_cases.sort(key=lambda c: str(c["id"]))

    # Create canonical dataset representation
    canonical_dataset = {
        "cases": canonical_cases,
        "name": dataset.name,
        "description": dataset.description,
        "version": dataset.version,
        "metadata": dataset.metadata,
        "tags": dataset.tags,
    }

    # Serialize with sorted keys for determinism
    canonical_json = json.dumps(canonical_dataset, sort_keys=True, separators=(",", ":"))

    # Compute SHA-256 hash
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def validate_cases(cases: list[dict[str, Any]], file_path: str) -> list[EvaluationCase]:
    """Validate and convert raw case dictionaries to EvaluationCase models.

    Args:
        cases: List of raw case dictionaries from JSON/JSONL.
        file_path: Path to the dataset file for error context.

    Returns:
        List of validated EvaluationCase instances.

    Raises:
        ValidationError: If any case fails validation.
    """
    validated_cases: list[EvaluationCase] = []
    errors: list[dict[str, Any]] = []

    for idx, case_data in enumerate(cases):
        try:
            validated_case = EvaluationCase.model_validate(case_data)
            validated_cases.append(validated_case)
        except Exception as e:
            errors.append(
                {
                    "index": idx,
                    "case_id": case_data.get("id", "unknown"),
                    "error": str(e),
                }
            )

    if errors:
        raise ValidationError(
            f"Failed to validate {len(errors)} case(s) in dataset",
            details={"file": file_path, "errors": errors},
        )

    return validated_cases
