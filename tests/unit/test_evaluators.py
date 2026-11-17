"""Tests for deterministic answer evaluators."""

from eval_harness.datasets.models import EvaluationCase
from eval_harness.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    NormalizedExactMatchEvaluator,
    StringSimilarityEvaluator,
)


class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = ExactMatchEvaluator()

    def test_exact_match(self) -> None:
        """Test exact string match returns 1.0."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 1.0
        assert score.metric_type == "exact_match"
        assert score.name == "exact_match"
        assert score.details["match"] is True

    def test_exact_mismatch(self) -> None:
        """Test exact string mismatch returns 0.0."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "5")
        assert score.value == 0.0
        assert score.details["match"] is False

    def test_case_sensitive(self) -> None:
        """Test that comparison is case-sensitive."""
        case = EvaluationCase(
            input="Capital of France?",
            expected_output="Paris",
        )
        score = self.evaluator.evaluate(case, "paris")
        assert score.value == 0.0

    def test_whitespace_sensitive(self) -> None:
        """Test that whitespace differences cause mismatch."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, " 4 ")
        assert score.value == 0.0

    def test_no_expected_output(self) -> None:
        """Test case with no expected output returns 0.0."""
        case = EvaluationCase(input="What is 2+2?")
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 0.0
        assert score.details["reason"] == "no_expected_output"

    def test_no_prediction(self) -> None:
        """Test case with no prediction returns 0.0."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_prediction"

    def test_empty_strings(self) -> None:
        """Test empty string handling."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="",
        )
        score = self.evaluator.evaluate(case, "")
        assert score.value == 1.0

    def test_both_none(self) -> None:
        """Test when both expected and prediction are None."""
        case = EvaluationCase(input="What is 2+2?")
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0


class TestNormalizedExactMatchEvaluator:
    """Tests for NormalizedExactMatchEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = NormalizedExactMatchEvaluator()

    def test_exact_match(self) -> None:
        """Test exact match after normalization."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 1.0

    def test_case_insensitive(self) -> None:
        """Test that normalization makes comparison case-insensitive."""
        case = EvaluationCase(
            input="Capital of France?",
            expected_output="Paris",
        )
        score = self.evaluator.evaluate(case, "paris")
        assert score.value == 1.0

    def test_whitespace_normalization(self) -> None:
        """Test that whitespace is normalized."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "  4  ")
        assert score.value == 1.0

    def test_multiple_whitespace_collapse(self) -> None:
        """Test that multiple whitespace characters are collapsed."""
        case = EvaluationCase(
            input="Describe the sky",
            expected_output="The sky is blue",
        )
        score = self.evaluator.evaluate(case, "The    sky  is   blue")
        assert score.value == 1.0

    def test_tab_and_newline_normalization(self) -> None:
        """Test that tabs and newlines are normalized to spaces."""
        case = EvaluationCase(
            input="Multi-line answer",
            expected_output="line one\nline two",
        )
        score = self.evaluator.evaluate(case, "line one\tline two")
        assert score.value == 1.0

    def test_mismatch_after_normalization(self) -> None:
        """Test that different content still mismatches after normalization."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "5")
        assert score.value == 0.0

    def test_no_expected_output(self) -> None:
        """Test case with no expected output."""
        case = EvaluationCase(input="What is 2+2?")
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 0.0
        assert score.details["reason"] == "no_expected_output"

    def test_no_prediction(self) -> None:
        """Test case with no prediction."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_prediction"

    def test_normalized_values_in_details(self) -> None:
        """Test that normalized values are included in details."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output=" 4 ",
        )
        score = self.evaluator.evaluate(case, "  4  ")
        assert score.details["normalized_prediction"] == "4"
        assert score.details["normalized_expected"] == "4"


class TestContainsEvaluator:
    """Tests for ContainsEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = ContainsEvaluator()

    def test_substring_match(self) -> None:
        """Test expected answer found in prediction."""
        case = EvaluationCase(
            input="Describe Paris",
            expected_output="Paris is the capital of France",
        )
        score = self.evaluator.evaluate(case, "Paris is the capital of France and a beautiful city")
        assert score.value == 1.0
        assert score.details["match"] is True

    def test_substring_not_found(self) -> None:
        """Test expected answer not found in prediction."""
        case = EvaluationCase(
            input="Describe Paris",
            expected_output="capital of France",
        )
        score = self.evaluator.evaluate(case, "Paris is a beautiful city")
        assert score.value == 0.0
        assert score.details["match"] is False

    def test_case_sensitive_by_default(self) -> None:
        """Test that default is case-sensitive."""
        case = EvaluationCase(
            input="Capital of France?",
            expected_output="Paris",
        )
        score = self.evaluator.evaluate(case, "paris is the capital")
        assert score.value == 0.0

    def test_case_insensitive_option(self) -> None:
        """Test case-insensitive option."""
        case = EvaluationCase(
            input="Capital of France?",
            expected_output="Paris",
        )
        score = self.evaluator.evaluate(case, "paris is the capital", case_sensitive=False)
        assert score.value == 1.0

    def test_exact_match_counts(self) -> None:
        """Test that exact match is also a substring match."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 1.0

    def test_empty_expected(self) -> None:
        """Test empty expected output."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="",
        )
        score = self.evaluator.evaluate(case, "anything")
        assert score.value == 1.0  # Empty string is substring of everything

    def test_empty_prediction(self) -> None:
        """Test empty prediction with non-empty expected."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "")
        assert score.value == 0.0

    def test_no_expected_output(self) -> None:
        """Test case with no expected output."""
        case = EvaluationCase(input="What is 2+2?")
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 0.0
        assert score.details["reason"] == "no_expected_output"

    def test_no_prediction(self) -> None:
        """Test case with no prediction."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_prediction"


class TestStringSimilarityEvaluator:
    """Tests for StringSimilarityEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = StringSimilarityEvaluator()

    def test_identical_strings(self) -> None:
        """Test similarity of identical strings is 1.0."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 1.0

    def test_completely_different_strings(self) -> None:
        """Test similarity of completely different strings is 0.0 or near 0.0."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "completely different")
        assert 0.0 <= score.value < 0.5

    def test_partial_match(self) -> None:
        """Test similarity for partially matching strings."""
        case = EvaluationCase(
            input="Describe the sky",
            expected_output="The sky is blue and beautiful",
        )
        score = self.evaluator.evaluate(case, "The sky is blue")
        assert 0.5 < score.value < 1.0

    def test_case_sensitivity(self) -> None:
        """Test that similarity is case-sensitive."""
        case = EvaluationCase(
            input="Capital of France?",
            expected_output="Paris",
        )
        score1 = self.evaluator.evaluate(case, "Paris")
        score2 = self.evaluator.evaluate(case, "paris")
        assert score1.value == 1.0
        assert score2.value < 1.0

    def test_no_expected_output(self) -> None:
        """Test case with no expected output."""
        case = EvaluationCase(input="What is 2+2?")
        score = self.evaluator.evaluate(case, "4")
        assert score.value == 0.0
        assert score.details["reason"] == "no_expected_output"

    def test_no_prediction(self) -> None:
        """Test case with no prediction."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_prediction"

    def test_empty_strings(self) -> None:
        """Test empty string handling."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="",
        )
        score = self.evaluator.evaluate(case, "")
        assert score.value == 1.0

    def test_details_include_lengths(self) -> None:
        """Test that details include prediction and expected lengths."""
        case = EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
        )
        score = self.evaluator.evaluate(case, "four")
        assert "prediction_length" in score.details
        assert "expected_length" in score.details
        assert score.details["prediction_length"] == 4
        assert score.details["expected_length"] == 1

    def test_score_in_range(self) -> None:
        """Test that score is always in [0, 1]."""
        case = EvaluationCase(
            input="test",
            expected_output="abcdefghijklmnopqrstuvwxyz",
        )
        score = self.evaluator.evaluate(case, "zyxwvutsrqponmlkjihgfedcba")
        assert 0.0 <= score.value <= 1.0


class TestEvaluatorProtocol:
    """Tests for evaluator protocol compliance."""

    def test_all_evaluators_have_metric_type(self) -> None:
        """Test all evaluators have metric_type property."""
        evaluators = [
            ExactMatchEvaluator(),
            NormalizedExactMatchEvaluator(),
            ContainsEvaluator(),
            StringSimilarityEvaluator(),
        ]
        for ev in evaluators:
            assert hasattr(ev, "metric_type")
            assert ev.metric_type is not None

    def test_all_evaluators_have_name(self) -> None:
        """Test all evaluators have name property."""
        evaluators = [
            ExactMatchEvaluator(),
            NormalizedExactMatchEvaluator(),
            ContainsEvaluator(),
            StringSimilarityEvaluator(),
        ]
        for ev in evaluators:
            assert hasattr(ev, "name")
            assert isinstance(ev.name, str)
            assert len(ev.name) > 0

    def test_all_evaluators_return_metric_score(self) -> None:
        """Test all evaluators return MetricScore."""
        case = EvaluationCase(input="test", expected_output="test")
        evaluators = [
            ExactMatchEvaluator(),
            NormalizedExactMatchEvaluator(),
            ContainsEvaluator(),
            StringSimilarityEvaluator(),
        ]
        for ev in evaluators:
            score = ev.evaluate(case, "test")
            assert score.metric_type == ev.metric_type
            assert score.name == ev.name
            assert isinstance(score.value, float)
            assert 0.0 <= score.value <= 1.0
