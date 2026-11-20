"""Structured output evaluators for JSON validity, schema validation, and field matching.

These evaluators check whether an AI system returns the correct JSON structure and data, 
rather than just checking whether its answer is textually similar to the expected answer"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from eval_harness.datasets.models import EvaluationCase
from eval_harness.evaluators.base import BaseEvaluator
from eval_harness.metrics.models import MetricScore, MetricType

if TYPE_CHECKING:
    import jsonschema
    from jsonschema import Draft202012Validator, FormatChecker
else:
    jsonschema = Any  # type: ignore[assignment]
    Draft202012Validator = Any  # type: ignore[assignment]
    FormatChecker = Any  # type: ignore[assignment]

# Conditional import for optional jsonschema dependency
try:
    from jsonschema import Draft202012Validator, FormatChecker
    JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    JSONSCHEMA_AVAILABLE = False
    # These are already typed as Any above for TYPE_CHECKING
    # Runtime values are just placeholders
    pass


class JSONValidityEvaluator(BaseEvaluator):
    """JSON validity evaluator.

    Determines whether a prediction contains valid JSON.
    Returns 1.0 for valid JSON, 0.0 for invalid or non-JSON.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.CUSTOM

    @property
    def name(self) -> str:
        return "json_validity"

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate if prediction is valid JSON.

        Args:
            case: Evaluation case (expected_output not used).
            prediction: The actual prediction to evaluate as JSON string.
            **kwargs: Ignored.

        Returns:
            MetricScore with value 1.0 if valid JSON, 0.0 otherwise.
        """
        if prediction is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_prediction", "valid": False},
                metadata={"case_id": str(case.id)},
            )

        if not isinstance(prediction, str):
            return self._create_score(
                value=0.0,
                details={
                    "reason": "not_a_string",
                    "valid": False,
                    "prediction_type": type(prediction).__name__,
                },
                metadata={"case_id": str(case.id)},
            )

        # Check for empty or whitespace-only strings
        if not prediction.strip():
            return self._create_score(
                value=0.0,
                details={"valid": False, "reason": "empty_string"},
                metadata={"case_id": str(case.id)},
            )

        try:
            parsed = json.loads(prediction)
            # Determine parsed type
            if isinstance(parsed, dict):
                parsed_type = "dict"
            elif isinstance(parsed, list):
                parsed_type = "list"
            elif isinstance(parsed, str):
                parsed_type = "string"
            elif isinstance(parsed, (int, float)):
                parsed_type = "number"
            elif isinstance(parsed, bool):
                parsed_type = "boolean"
            elif parsed is None:
                parsed_type = "null"
            else:
                parsed_type = type(parsed).__name__

            return self._create_score(
                value=1.0,
                details={"valid": True, "parsed_type": parsed_type},
                metadata={"case_id": str(case.id)},
            )
        except json.JSONDecodeError as e:
            return self._create_score(
                value=0.0,
                details={
                    "valid": False,
                    "reason": "invalid_json",
                    "error": str(e),
                    "error_position": e.pos,
                },
                metadata={"case_id": str(case.id)},
            )


class JSONSchemaEvaluator(BaseEvaluator):
    """JSON schema validation evaluator.

    Validates a JSON prediction against a JSON Schema (Draft 2020-12).
    Returns 1.0 if valid, 0.0 if invalid or schema not available.
    """

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        """Initialize with a JSON schema.

        Args:
            schema: The JSON Schema to validate against.
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema package is required for JSONSchemaEvaluator. "
                "Install with: pip install jsonschema"
            )
        self._schema = schema
        if schema is not None:
            self._validator = Draft202012Validator(schema)

    @property
    def metric_type(self) -> MetricType:
        return MetricType.CUSTOM

    @property
    def name(self) -> str:
        return "json_schema_validation"

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate JSON prediction against a schema.

        Args:
            case: Evaluation case (expected_output not used).
            prediction: The actual prediction to evaluate as JSON.
            schema: Optional JSON Schema to override instance schema.
            **kwargs: Ignored.

        Returns:
            MetricScore with value 1.0 if valid against schema, 0.0 otherwise.
        """
        if not JSONSCHEMA_AVAILABLE:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "jsonschema_not_available",
                    "valid": False,
                    "error": "jsonschema package not installed",
                },
                metadata={"case_id": str(case.id)},
            )

        active_schema = schema or self._schema
        if active_schema is None:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "no_schema_provided",
                    "valid": False,
                },
                metadata={"case_id": str(case.id)},
            )

        if prediction is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_prediction", "valid": False},
                metadata={"case_id": str(case.id)},
            )

        if not isinstance(prediction, str):
            return self._create_score(
                value=0.0,
                details={
                    "reason": "not_a_string",
                    "valid": False,
                    "prediction_type": type(prediction).__name__,
                },
                metadata={"case_id": str(case.id)},
            )

        try:
            parsed = json.loads(prediction)
        except json.JSONDecodeError as e:
            return self._create_score(
                value=0.0,
                details={
                    "valid": False,
                    "reason": "invalid_json",
                    "json_error": str(e),
                    "error_position": e.pos,
                },
                metadata={"case_id": str(case.id)},
            )

        validator = Draft202012Validator(active_schema, format_checker=FormatChecker())
        errors = list(validator.iter_errors(parsed))

        if not errors:
            return self._create_score(
                value=1.0,
                details={"valid": True, "errors": []},
                metadata={"case_id": str(case.id)},
            )

        error_details = []
        for error in errors:
            error_details.append({
                "message": error.message,
                "path": list(error.path),
                "schema_path": list(error.schema_path),
                "validator": error.validator,
                "validator_value": error.validator_value,
            })

        return self._create_score(
            value=0.0,
            details={
                "valid": False,
                "error_count": len(errors),
                "errors": error_details,
            },
            metadata={"case_id": str(case.id)},
        )


class JSONFieldMatchEvaluator(BaseEvaluator):
    """JSON field match evaluator.

    Compares selected fields in predicted JSON against expected JSON.
    Supports nested field paths using dot notation (e.g., "user.address.city").
    Returns ratio of matching fields to total fields compared.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.CUSTOM

    @property
    def name(self) -> str:
        return "json_field_match"

    def _get_field(self, obj: Any, path: str) -> tuple[Any, bool]:
        """Get field value from object using dot notation path.

        Args:
            obj: The JSON object to traverse.
            path: Dot-notation path (e.g., "user.address.city" or "items[0].id").

        Returns:
            Tuple of (value, found) where found indicates if path exists.
        """
        if path == "" or path == ".":
            return obj, True

        current: Any = obj
        parts = path.split(".")

        for part in parts:
            if part.endswith("]"):
                # Handle array index: "items[0]" -> key="items", index=0
                key, idx_str = part[:-1].split("[")
                try:
                    index = int(idx_str)
                except ValueError:
                    return None, False

                if not isinstance(current, dict) or key not in current:
                    return None, False
                current = current[key]
                if not isinstance(current, list) or index >= len(current):
                    return None, False
                current = current[index]
            else:
                if not isinstance(current, dict) or part not in current:
                    return None, False
                current = current[part]

        return current, True

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Compare two values for equality, handling JSON types."""
        if type(val1) is not type(val2):
            return False
        if isinstance(val1, (dict, list)):
            return bool(val1 == val2)
        return bool(val1 == val2)

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        expected: str | dict[str, Any] | None = None,
        fields: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate field-level match between predicted and expected JSON.

        Args:
            case: Evaluation case with expected_output containing expected JSON.
            prediction: The actual prediction to evaluate as JSON string.
            expected: Optional expected JSON (overrides case.expected_output).
            fields: List of field paths to compare (dot notation). If None, compare all.
            **kwargs: Ignored.

        Returns:
            MetricScore with value as ratio of matching fields.
        """
        # Parse expected JSON
        if expected is None:
            expected_str = case.expected_output
            if not expected_str:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "no_expected_output",
                        "matched_fields": 0,
                        "total_fields": 0,
                        "field_results": {},
                    },
                    metadata={"case_id": str(case.id)},
                )
            try:
                expected_parsed = json.loads(expected_str)
            except json.JSONDecodeError as e:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "invalid_expected_json",
                        "error": str(e),
                        "matched_fields": 0,
                        "total_fields": 0,
                        "field_results": {},
                    },
                    metadata={"case_id": str(case.id)},
                )
        elif isinstance(expected, dict):
            expected_parsed = expected
        else:
            # expected is a string
            if not expected:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "no_expected_output",
                        "matched_fields": 0,
                        "total_fields": 0,
                        "field_results": {},
                    },
                    metadata={"case_id": str(case.id)},
                )
            try:
                expected_parsed = json.loads(expected)
            except json.JSONDecodeError as e:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "invalid_expected_json",
                        "error": str(e),
                        "matched_fields": 0,
                        "total_fields": 0,
                        "field_results": {},
                    },
                    metadata={"case_id": str(case.id)},
                )

        # Parse prediction JSON
        if prediction is None:
            details = {
                "reason": "no_prediction",
                "matched_fields": 0,
                "total_fields": 0,
                "field_results": {},
            }
            return self._create_score(
                value=0.0,
                details=details,
                metadata={"case_id": str(case.id)},
            )

        if not isinstance(prediction, str):
            return self._create_score(
                value=0.0,
                details={
                    "reason": "not_a_string",
                    "matched_fields": 0,
                    "total_fields": 0,
                    "field_results": {},
                    "prediction_type": type(prediction).__name__,
                },
                metadata={"case_id": str(case.id)},
            )

        try:
            predicted_parsed = json.loads(prediction)
        except json.JSONDecodeError as e:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "invalid_prediction_json",
                    "error": str(e),
                    "matched_fields": 0,
                    "total_fields": 0,
                    "field_results": {},
                },
                metadata={"case_id": str(case.id)},
            )

        # Determine fields to compare
        if fields is None:
            # Compare all fields in expected
            def collect_paths(obj: Any, prefix: str = "") -> list[str]:
                paths = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        paths.append(new_prefix)
                        paths.extend(collect_paths(value, new_prefix))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_prefix = f"{prefix}[{i}]"
                        paths.extend(collect_paths(item, new_prefix))
                return paths

            fields_to_compare = collect_paths(expected_parsed)
            if not fields_to_compare:
                fields_to_compare = [""]
        else:
            fields_to_compare = fields

        # Compare each field
        field_results = {}
        matched = 0

        for field_path in fields_to_compare:
            expected_val, expected_found = self._get_field(expected_parsed, field_path)
            predicted_val, predicted_found = self._get_field(predicted_parsed, field_path)

            if not expected_found and not predicted_found:
                # Both missing - consider as match
                match = True
            elif expected_found and predicted_found:
                match = self._values_equal(expected_val, predicted_val)
            else:
                # One missing, one present
                match = False

            field_results[field_path] = {
                "match": match,
                "expected": expected_val,
                "predicted": predicted_val,
                "expected_found": expected_found,
                "predicted_found": predicted_found,
            }

            if match:
                matched += 1

        total = len(fields_to_compare)
        score = matched / total if total > 0 else 1.0

        return self._create_score(
            value=score,
            details={
                "matched_fields": matched,
                "total_fields": total,
                "field_results": field_results,
            },
            metadata={"case_id": str(case.id)},
        )
