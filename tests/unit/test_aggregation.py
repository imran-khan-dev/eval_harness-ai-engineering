"""Tests for metric aggregation functionality."""

from uuid import uuid4

import pytest

from eval_harness.metrics import (
    AggregationResult,
    MetricAggregator,
    MetricScore,
    MetricType,
    aggregate_scores,
)
from eval_harness.metrics.models import EvaluationResult, EvaluationStatus, ExperimentResult


class TestAggregationResult:
    """Tests for AggregationResult dataclass."""

    def test_aggregation_result_creation(self) -> None:
        result = AggregationResult(
            count=5,
            mean=0.8,
            min=0.5,
            max=1.0,
            std_dev=0.2,
            pass_count=3,
            fail_count=1,
            no_threshold_count=1,
            pass_rate=0.75,
        )
        assert result.count == 5
        assert result.mean == 0.8
        assert result.pass_rate == 0.75


class TestMetricAggregator:
    """Tests for MetricAggregator class."""

    def setup_method(self) -> None:
        self.aggregator = MetricAggregator()

    def _make_scores(self) -> list[MetricScore]:
        """Create a set of test scores."""
        return [
            MetricScore(
                metric_type=MetricType.ACCURACY,
                name="accuracy",
                value=0.9,
                threshold=0.8,
            ),
            MetricScore(
                metric_type=MetricType.ACCURACY,
                name="accuracy",
                value=0.7,
                threshold=0.8,
            ),
            MetricScore(
                metric_type=MetricType.F1,
                name="f1_score",
                value=0.85,
                threshold=0.8,
            ),
            MetricScore(
                metric_type=MetricType.PRECISION,
                name="precision",
                value=0.8,
                threshold=None,
            ),
            MetricScore(
                metric_type=MetricType.PRECISION,
                name="precision",
                value=0.6,
                threshold=0.7,
            ),
        ]

    def test_empty_aggregation(self) -> None:
        """Test aggregation with no scores."""
        result = self.aggregator.aggregate()
        assert result["overall"].count == 0
        assert result["overall"].mean == 0.0
        assert result["overall"].pass_rate == 0.0
        assert result["by_type"] == {}
        assert result["by_name"] == {}

    def test_add_and_aggregate(self) -> None:
        """Test adding scores and aggregating."""
        scores = self._make_scores()
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()

        # Overall
        assert result["overall"].count == 5
        assert result["overall"].mean == 0.77  # (0.9+0.7+0.85+0.8+0.6)/5
        assert result["overall"].min == 0.6
        assert result["overall"].max == 0.9
        assert result["overall"].pass_count == 2  # accuracy 0.9, f1 0.85
        assert result["overall"].fail_count == 2  # accuracy 0.7, precision 0.6
        assert result["overall"].no_threshold_count == 1  # precision 0.8
        assert result["overall"].pass_rate == 0.5  # 2/(2+2)

        # By type
        assert MetricType.ACCURACY in result["by_type"]
        acc = result["by_type"][MetricType.ACCURACY]
        assert acc.count == 2
        assert acc.mean == 0.8
        assert acc.pass_rate == 0.5

        assert MetricType.F1 in result["by_type"]
        f1 = result["by_type"][MetricType.F1]
        assert f1.count == 1
        assert f1.mean == 0.85
        assert f1.pass_rate == 1.0

        # By name
        assert "accuracy" in result["by_name"]
        assert "f1_score" in result["by_name"]
        assert "precision" in result["by_name"]

    def test_aggregate_by_type(self) -> None:
        """Test grouping by metric type."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc1", value=0.9),
            MetricScore(metric_type=MetricType.ACCURACY, name="acc2", value=0.8),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.85),
        ]
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()

        assert len(result["by_type"]) == 2
        assert result["by_type"][MetricType.ACCURACY].count == 2
        assert result["by_type"][MetricType.F1].count == 1

    def test_aggregate_by_name(self) -> None:
        """Test grouping by metric name."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=0.9),
            MetricScore(metric_type=MetricType.F1, name="accuracy", value=0.8),
        ]
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()

        assert len(result["by_name"]) == 1
        assert result["by_name"]["accuracy"].count == 2
        assert result["by_name"]["accuracy"].mean == pytest.approx(0.85)

    def test_std_dev_calculation(self) -> None:
        """Test standard deviation calculation."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=1.0),
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.0),
        ]
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()

        # Population std dev of [1.0, 0.0] with mean 0.5 is 0.5
        assert result["overall"].std_dev == 0.5

    def test_single_value_std_dev(self) -> None:
        """Test std dev with single value is 0."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8),
        ]
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()

        assert result["overall"].std_dev == 0.0

    def test_clear(self) -> None:
        """Test clearing scores."""
        scores = self._make_scores()
        self.aggregator.add_scores(scores)
        self.aggregator.clear()
        result = self.aggregator.aggregate()
        assert result["overall"].count == 0


class TestAggregateScoresFunction:
    """Tests for the aggregate_scores convenience function."""

    def test_basic_aggregation(self) -> None:
        """Test basic unweighted aggregation."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8),
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.7, threshold=0.8),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.85, threshold=0.8),
        ]
        result = aggregate_scores(scores)

        assert result["overall"].count == 3
        assert result["overall"].mean == pytest.approx(0.8167, rel=1e-3)
        assert result["overall"].pass_count == 2
        assert result["overall"].fail_count == 1
        assert result["overall"].pass_rate == pytest.approx(2 / 3)

    def test_empty_input(self) -> None:
        """Test aggregation with empty list."""
        result = aggregate_scores([])
        assert result["overall"].count == 0
        assert result["overall"].mean == 0.0

    def test_no_thresholds(self) -> None:
        """Test aggregation with no thresholds."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.8),
        ]
        result = aggregate_scores(scores)

        assert result["overall"].pass_count == 0
        assert result["overall"].fail_count == 0
        assert result["overall"].no_threshold_count == 2
        assert result["overall"].pass_rate == 0.0


class TestWeightedAggregation:
    """Tests for weighted aggregation."""

    def test_weighted_mean(self) -> None:
        """Test weighted average calculation."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=1.0),
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=0.5),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.8),
        ]
        weights = {"accuracy": 2.0, "f1": 1.0}
        result = aggregate_scores(scores, weights=weights)

        # Weighted mean: (1.0*2 + 0.5*2 + 0.8*1) / (2+2+1) = 3.8/5 = 0.76
        assert result["overall"].mean == pytest.approx(0.76)

    def test_weighted_pass_rate(self) -> None:
        """Test weighted pass rate."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=0.9, threshold=0.8),
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=0.7, threshold=0.8),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.85, threshold=0.8),
        ]
        weights = {"accuracy": 2.0, "f1": 1.0}
        result = aggregate_scores(scores, weights=weights)

        # accuracy: 1 pass (weight 2), 1 fail (weight 2)
        # f1: 1 pass (weight 1)
        # Weighted pass rate: (2 + 1) / (2 + 2 + 1) = 3/5 = 0.6
        assert result["overall"].pass_rate == pytest.approx(0.6)

    def test_default_weight(self) -> None:
        """Test default weight for unlisted metrics."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=1.0),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.5),
        ]
        weights = {"accuracy": 3.0}  # f1 not listed
        result = aggregate_scores(scores, weights=weights, default_weight=1.0)

        # (1.0*3 + 0.5*1) / (3+1) = 3.5/4 = 0.875
        assert result["overall"].mean == pytest.approx(0.875)

    def test_negative_weight_raises(self) -> None:
        """Test that negative weights raise ValueError."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9),
        ]
        weights = {"acc": -1.0}
        try:
            aggregate_scores(scores, weights=weights)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "non-negative" in str(e)

    def test_zero_total_weight_raises(self) -> None:
        """Test that zero total weight raises ValueError."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9),
        ]
        weights = {"acc": 0.0}
        try:
            aggregate_scores(scores, weights=weights, default_weight=0.0)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Total weight is zero" in str(e)

    def test_weights_do_not_affect_min_max(self) -> None:
        """Test that weights don't affect min/max/std_dev (they remain unweighted)."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=1.0),
            MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=0.2),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.8),
        ]
        weights = {"accuracy": 10.0, "f1": 1.0}
        result = aggregate_scores(scores, weights=weights)

        # Min/max should be unweighted
        assert result["overall"].min == 0.2
        assert result["overall"].max == 1.0


class TestEvaluationResultAggregation:
    """Tests for EvaluationResult aggregation methods."""

    def test_evaluation_result_aggregate_metrics(self) -> None:
        """Test EvaluationResult.aggregate_metrics()."""
        case_id = uuid4()
        result = EvaluationResult(case_id=case_id, status=EvaluationStatus.PASSED)

        result.add_metric_score(
            MetricScore(
                metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8
            )
        )
        result.add_metric_score(
            MetricScore(
                metric_type=MetricType.F1, name="f1", value=0.85, threshold=0.8
            )
        )
        result.add_metric_score(
            MetricScore(metric_type=MetricType.PRECISION, name="prec", value=0.8)
        )

        agg = result.aggregate_metrics()

        assert agg["overall"].count == 3
        assert agg["overall"].mean == pytest.approx(0.85)
        assert MetricType.ACCURACY in agg["by_type"]
        assert MetricType.F1 in agg["by_type"]
        assert MetricType.PRECISION in agg["by_type"]

    def test_evaluation_result_weighted_aggregation(self) -> None:
        """Test EvaluationResult.aggregate_metrics() with weights."""
        case_id = uuid4()
        result = EvaluationResult(case_id=case_id, status=EvaluationStatus.PASSED)

        result.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=1.0))
        result.add_metric_score(MetricScore(metric_type=MetricType.F1, name="f1", value=0.5))

        weights = {"acc": 3.0, "f1": 1.0}
        agg = result.aggregate_metrics(weights=weights)

        # (1.0*3 + 0.5*1) / 4 = 0.875
        assert agg["overall"].mean == pytest.approx(0.875)

    def test_evaluation_result_get_overall_score(self) -> None:
        """Test EvaluationResult.get_overall_score()."""
        case_id = uuid4()
        result = EvaluationResult(case_id=case_id, status=EvaluationStatus.PASSED)

        result.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9))
        result.add_metric_score(MetricScore(metric_type=MetricType.F1, name="f1", value=0.7))

        score = result.get_overall_score()
        assert score == 0.8


class TestExperimentResultAggregation:
    """Tests for ExperimentResult aggregation methods."""

    def test_experiment_result_aggregate_all_metrics(self) -> None:
        """Test ExperimentResult.aggregate_all_metrics()."""
        exp = ExperimentResult(config_name="test", dataset_name="test")

        # Case 1
        case1 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        case1.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9))
        case1.add_metric_score(MetricScore(metric_type=MetricType.F1, name="f1", value=0.85))
        exp.add_case_result(case1)

        # Case 2
        case2 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        case2.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8))
        case2.add_metric_score(MetricScore(metric_type=MetricType.F1, name="f1", value=0.75))
        exp.add_case_result(case2)

        agg = exp.aggregate_all_metrics()

        assert agg["overall"].count == 4
        assert agg["overall"].mean == pytest.approx(0.825)
        assert agg["by_type"][MetricType.ACCURACY].count == 2
        assert agg["by_type"][MetricType.F1].count == 2

    def test_experiment_result_case_level_aggregates(self) -> None:
        """Test ExperimentResult.get_case_level_aggregates()."""
        exp = ExperimentResult(config_name="test", dataset_name="test")

        case1 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        case1.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9))
        exp.add_case_result(case1)

        case2 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        case2.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.7))
        exp.add_case_result(case2)

        case_aggs = exp.get_case_level_aggregates()

        assert len(case_aggs) == 2
        assert case_aggs[0]["overall"].mean == 0.9
        assert case_aggs[1]["overall"].mean == 0.7

    def test_experiment_result_weighted_aggregation(self) -> None:
        """Test ExperimentResult.aggregate_all_metrics() with weights."""
        exp = ExperimentResult(config_name="test", dataset_name="test")

        case1 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        case1.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=1.0))
        case1.add_metric_score(MetricScore(metric_type=MetricType.F1, name="f1", value=0.5))
        exp.add_case_result(case1)

        weights = {"acc": 2.0, "f1": 1.0}
        agg = exp.aggregate_all_metrics(weights=weights)

        # (1.0*2 + 0.5*1) / 3 = 2.5/3 ≈ 0.8333
        assert agg["overall"].mean == pytest.approx(0.8333, rel=1e-3)


class TestEdgeCases:
    """Tests for edge cases in aggregation."""

    def test_all_scores_same_value(self) -> None:
        """Test aggregation when all scores are identical."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8),
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8),
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8),
        ]
        result = aggregate_scores(scores)

        assert result["overall"].mean == pytest.approx(0.8)
        assert result["overall"].min == 0.8
        assert result["overall"].max == 0.8
        assert result["overall"].std_dev == pytest.approx(0.0)

    def test_scores_at_boundaries(self) -> None:
        """Test aggregation with scores at 0.0 and 1.0."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.0),
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=1.0),
        ]
        result = aggregate_scores(scores)

        assert result["overall"].mean == 0.5
        assert result["overall"].min == 0.0
        assert result["overall"].max == 1.0

    def test_mixed_threshold_and_no_threshold(self) -> None:
        """Test aggregation with mix of thresholded and non-thresholded scores."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.85, threshold=0.8),
            MetricScore(metric_type=MetricType.PRECISION, name="prec", value=0.8),
        ]
        result = aggregate_scores(scores)

        assert result["overall"].pass_count == 2
        assert result["overall"].fail_count == 0
        assert result["overall"].no_threshold_count == 1
        assert result["overall"].pass_rate == 1.0  # 2/2

    def test_all_failed_thresholds(self) -> None:
        """Test aggregation when all thresholded scores fail."""
        scores = [
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.5, threshold=0.8),
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.6, threshold=0.8),
        ]
        result = aggregate_scores(scores)

        assert result["overall"].pass_count == 0
        assert result["overall"].fail_count == 2
        assert result["overall"].pass_rate == 0.0
