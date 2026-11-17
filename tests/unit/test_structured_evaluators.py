"""Tests for structured output evaluators."""


import pytest

from eval_harness.datasets.models import EvaluationCase
from eval_harness.evaluators import (
    JSONFieldMatchEvaluator,
    JSONSchemaEvaluator,
    JSONValidityEvaluator,
)


class TestJSONValidityEvaluator:
    """Tests for JSONValidityEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = JSONValidityEvaluator()

    def test_valid_json_object(self) -> None:
        """Test valid JSON object returns 1.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 30}')
        assert score.value == 1.0
        assert score.details["valid"] is True
        assert score.details["parsed_type"] == "dict"

    def test_valid_json_array(self) -> None:
        """Test valid JSON array returns 1.0."""
        case = EvaluationCase(input="test", expected_output="[]")
        score = self.evaluator.evaluate(case, '[1, 2, 3, "four"]')
        assert score.value == 1.0
        assert score.details["valid"] is True
        assert score.details["parsed_type"] == "list"

    def test_valid_json_primitives(self) -> None:
        """Test valid JSON primitives return 1.0."""
        case = EvaluationCase(input="test", expected_output="123")
        for pred in ["123", "3.14", "true", "false", '"hello"', "null"]:
            score = self.evaluator.evaluate(case, pred)
            assert score.value == 1.0, f"Failed for {pred}"
            assert score.details["valid"] is True

    def test_invalid_json_syntax(self) -> None:
        """Test invalid JSON syntax returns 0.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 30')
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert "error" in score.details
        assert "error_position" in score.details

    def test_invalid_json_trailing_comma(self) -> None:
        """Test JSON with trailing comma returns 0.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 30,}')
        assert score.value == 0.0
        assert score.details["valid"] is False

    def test_empty_string(self) -> None:
        """Test empty string returns 0.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, "")
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "empty_string"

    def test_whitespace_only(self) -> None:
        """Test whitespace-only string returns 0.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, "   \n\t  ")
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "empty_string"

    def test_none_prediction(self) -> None:
        """Test None prediction returns 0.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "no_prediction"

    def test_non_string_prediction(self) -> None:
        """Test non-string prediction returns 0.0."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, 123)
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "not_a_string"
        assert score.details["prediction_type"] == "int"

    def test_json_with_unicode(self) -> None:
        """Test valid JSON with unicode characters."""
        case = EvaluationCase(input="test", expected_output="{}")
        score = self.evaluator.evaluate(case, '{"name": "日本語", "emoji": "🎉"}')
        assert score.value == 1.0
        assert score.details["valid"] is True


class TestJSONSchemaEvaluator:
    """Tests for JSONSchemaEvaluator."""

    def setup_method(self) -> None:
        try:
            self.evaluator = JSONSchemaEvaluator({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["name", "age"],
            })
        except ImportError:
            pytest.skip("jsonschema not available")

    def test_valid_against_schema(self) -> None:
        """Test valid JSON against schema returns 1.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(
            case,
            '{"name": "John", "age": 30, "email": "john@example.com"}',
        )
        assert score.value == 1.0
        assert score.details["valid"] is True
        assert score.details["errors"] == []

    def test_missing_required_field(self) -> None:
        """Test missing required field returns 0.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(case, '{"name": "John"}')
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["error_count"] > 0
        assert any("required" in e["message"].lower() for e in score.details["errors"])

    def test_wrong_type(self) -> None:
        """Test wrong type returns 0.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(case, '{"name": "John", "age": "thirty"}')
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert any("integer" in e["message"].lower() for e in score.details["errors"])

    def test_invalid_format(self) -> None:
        """Test invalid email format returns 0.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(
            case,
            '{"name": "John", "age": 30, "email": "not-an-email"}',
        )
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert any("email" in e["message"].lower() for e in score.details["errors"])

    def test_minimum_constraint(self) -> None:
        """Test minimum constraint violation returns 0.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(case, '{"name": "John", "age": -5}')
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert any("minimum" in e["message"].lower() for e in score.details["errors"])

    def test_invalid_json_syntax(self) -> None:
        """Test invalid JSON syntax returns 0.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 30')
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "invalid_json"

    def test_no_prediction(self) -> None:
        """Test None prediction returns 0.0."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "no_prediction"

    def test_override_schema(self) -> None:
        """Test schema override in evaluate call."""
        case = EvaluationCase(input="test")
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        score = self.evaluator.evaluate(case, '{"name": "John"}', schema=schema)
        assert score.value == 1.0
        assert score.details["valid"] is True

    def test_no_schema_provided(self) -> None:
        """Test evaluator without schema returns 0.0."""
        evaluator = JSONSchemaEvaluator()
        case = EvaluationCase(input="test")
        score = evaluator.evaluate(case, '{"name": "John"}')
        assert score.value == 0.0
        assert score.details["valid"] is False
        assert score.details["reason"] == "no_schema_provided"


class TestJSONFieldMatchEvaluator:
    """Tests for JSONFieldMatchEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = JSONFieldMatchEvaluator()

    def test_exact_match_all_fields(self) -> None:
        """Test exact match of all fields returns 1.0."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John", "age": 30, "city": "NYC"}',
        )
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 30, "city": "NYC"}')
        assert score.value == 1.0
        assert score.details["matched_fields"] == 3
        assert score.details["total_fields"] == 3

    def test_partial_match(self) -> None:
        """Test partial field match returns proportional score."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John", "age": 30, "city": "NYC"}',
        )
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 25, "city": "NYC"}')
        assert score.value == pytest.approx(2/3)
        assert score.details["matched_fields"] == 2
        assert score.details["total_fields"] == 3

    def test_no_fields_match(self) -> None:
        """Test no fields match returns 0.0."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John", "age": 30}',
        )
        score = self.evaluator.evaluate(case, '{"name": "Jane", "age": 25}')
        assert score.value == 0.0
        assert score.details["matched_fields"] == 0

    def test_nested_field_paths(self) -> None:
        """Test nested field paths using dot notation."""
        case = EvaluationCase(
            input="test",
            expected_output='{"user": {"name": "John", "address": {"city": "NYC"}}}',
        )
        score = self.evaluator.evaluate(
            case,
            '{"user": {"name": "John", "address": {"city": "NYC"}}}',
        )
        assert score.value == 1.0

    def test_nested_field_mismatch(self) -> None:
        """Test nested field mismatch."""
        case = EvaluationCase(
            input="test",
            expected_output='{"user": {"name": "John", "address": {"city": "NYC"}}}',
        )
        score = self.evaluator.evaluate(
            case,
            '{"user": {"name": "John", "address": {"city": "LA"}}}',
        )
        # Should match user.name but not user.address.city
        assert score.value < 1.0

    def test_field_selection(self) -> None:
        """Test explicit field selection."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John", "age": 30, "city": "NYC", "country": "USA"}',
        )
        score = self.evaluator.evaluate(
            case,
            '{"name": "John", "age": 30, "city": "LA", "country": "USA"}',
            fields=["name", "age", "country"],
        )
        assert score.value == pytest.approx(1.0)  # All 3 specified fields match

    def test_array_index_access(self) -> None:
        """Test array index access in field paths."""
        case = EvaluationCase(
            input="test",
            expected_output='{"items": [{"id": 1}, {"id": 2}]}',
        )
        score = self.evaluator.evaluate(
            case,
            '{"items": [{"id": 1}, {"id": 3}]}',
        )
        # Should compare items.0.id and items.1.id
        assert score.value < 1.0

    def test_missing_predicted_field(self) -> None:
        """Test missing field in prediction."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John", "age": 30}',
        )
        score = self.evaluator.evaluate(case, '{"name": "John"}')
        assert score.value == 0.5  # Only name matches
        assert "age" in score.details["field_results"]
        assert score.details["field_results"]["age"]["match"] is False
        assert score.details["field_results"]["age"]["predicted_found"] is False

    def test_extra_predicted_field(self) -> None:
        """Test extra field in prediction (ignored by default)."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"}',
        )
        score = self.evaluator.evaluate(case, '{"name": "John", "extra": "ignored"}')
        assert score.value == 1.0  # Only expected fields compared by default

    def test_both_missing_field(self) -> None:
        """Test field missing in both expected and predicted."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"}',
        )
        score = self.evaluator.evaluate(
            case,
            '{}',
            fields=["name", "missing_field"],
        )
        # missing_field is missing in both - counts as match
        assert score.value == 0.5

    def test_invalid_prediction_json(self) -> None:
        """Test invalid prediction JSON returns 0.0."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"}',
        )
        score = self.evaluator.evaluate(case, '{"name": "John"')
        assert score.value == 0.0
        assert score.details["reason"] == "invalid_prediction_json"

    def test_invalid_expected_json(self) -> None:
        """Test invalid expected JSON returns 0.0."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"',  # Invalid JSON
        )
        score = self.evaluator.evaluate(case, '{"name": "John"}')
        assert score.value == 0.0
        assert score.details["reason"] == "invalid_expected_json"

    def test_no_expected_output(self) -> None:
        """Test case with no expected output."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(case, '{"name": "John"}')
        assert score.value == 0.0
        assert score.details["reason"] == "no_expected_output"

    def test_no_prediction(self) -> None:
        """Test None prediction."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"}',
        )
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_prediction"

    def test_non_string_prediction(self) -> None:
        """Test non-string prediction."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"}',
        )
        score = self.evaluator.evaluate(case, 123)
        assert score.value == 0.0
        assert score.details["reason"] == "not_a_string"

    def test_expected_as_dict(self) -> None:
        """Test expected output provided as dict."""
        case = EvaluationCase(input="test")
        score = self.evaluator.evaluate(
            case,
            '{"name": "John", "age": 30}',
            expected={"name": "John", "age": 30},
        )
        assert score.value == 1.0

    def test_field_results_structure(self) -> None:
        """Test field_results has correct structure."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John", "age": 30}',
        )
        score = self.evaluator.evaluate(case, '{"name": "John", "age": 25}')
        field_results = score.details["field_results"]
        assert "name" in field_results
        assert "age" in field_results
        assert field_results["name"]["match"] is True
        assert field_results["age"]["match"] is False
        assert "expected" in field_results["age"]
        assert "predicted" in field_results["age"]
        assert "expected_found" in field_results["age"]
        assert "predicted_found" in field_results["age"]

    def test_empty_fields_list(self) -> None:
        """Test empty fields list returns 1.0."""
        case = EvaluationCase(
            input="test",
            expected_output='{"name": "John"}',
        )
        score = self.evaluator.evaluate(case, '{}', fields=[])
        assert score.value == 1.0  # 0/0 treated as 1.0
        assert score.details["total_fields"] == 0


class TestStructuredEvaluatorsProtocol:
    """Tests for structured evaluators protocol compliance."""

    def test_all_evaluators_have_metric_type(self) -> None:
        """Test all evaluators have metric_type property."""
        evaluators = [
            JSONValidityEvaluator(),
            JSONSchemaEvaluator({"type": "object"}),
            JSONFieldMatchEvaluator(),
        ]
        for ev in evaluators:
            assert hasattr(ev, "metric_type")
            assert ev.metric_type is not None

    def test_all_evaluators_have_name(self) -> None:
        """Test all evaluators have name property."""
        evaluators = [
            JSONValidityEvaluator(),
            JSONSchemaEvaluator({"type": "object"}),
            JSONFieldMatchEvaluator(),
        ]
        for ev in evaluators:
            assert hasattr(ev, "name")
            assert isinstance(ev.name, str)
            assert len(ev.name) > 0

    def test_all_evaluators_return_metric_score(self) -> None:
        """Test all evaluators return MetricScore."""
        case = EvaluationCase(input="test", expected_output='{"test": true}')
        evaluators = [
            JSONValidityEvaluator(),
            JSONSchemaEvaluator({"type": "object"}),
            JSONFieldMatchEvaluator(),
        ]
        for ev in evaluators:
            score = ev.evaluate(case, '{"test": true}')
            assert score.metric_type == ev.metric_type
            assert score.name == ev.name
            assert isinstance(score.value, float)
            assert 0.0 <= score.value <= 1.0
