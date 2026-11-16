"""Tests for metrics models."""

from datetime import datetime
from uuid import uuid4

import pytest

from eval_harness.metrics.models import (
    AggregatedMetrics,
    EvaluationResult,
    EvaluationStatus,
    ExperimentResult,
    MetricScore,
    MetricType,
)


class TestMetricScore:
    """Tests for MetricScore model."""

    def test_minimal_score(self) -> None:
        score = MetricScore(metric_type=MetricType.ACCURACY, name="accuracy", value=0.9)
        assert score.metric_type == MetricType.ACCURACY
        assert score.name == "accuracy"
        assert score.value == 0.9
        assert score.threshold is None
        assert score.passed is None

    def test_with_threshold(self) -> None:
        score = MetricScore(
            metric_type=MetricType.F1,
            name="f1_score",
            value=0.85,
            threshold=0.8,
        )
        assert score.threshold == 0.8
        assert score.passed is True

    def test_failed_threshold(self) -> None:
        score = MetricScore(
            metric_type=MetricType.PRECISION,
            name="precision",
            value=0.7,
            threshold=0.8,
        )
        assert score.passed is False

    def test_explicit_passed(self) -> None:
        score = MetricScore(
            metric_type=MetricType.CUSTOM,
            name="custom",
            value=0.5,
            threshold=0.6,
            passed=True,  # Explicitly override
        )
        assert score.passed is True

    def test_details_and_metadata(self) -> None:
        score = MetricScore(
            metric_type=MetricType.SEMANTIC_SIMILARITY,
            name="similarity",
            value=0.9,
            details={"model": "embed-v3", "dimensions": 768},
            metadata={"source": "test"},
        )
        assert score.details == {"model": "embed-v3", "dimensions": 768}
        assert score.metadata == {"source": "test"}


class TestAggregatedMetrics:
    """Tests for AggregatedMetrics model."""

    def test_empty_aggregates(self) -> None:
        agg = AggregatedMetrics()
        assert agg.metric_scores == []
        assert agg.mean_scores == {}
        assert agg.total_cases == 0

    def test_compute_aggregates_single_type(self) -> None:
        agg = AggregatedMetrics(
            metric_scores=[
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8),
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8, threshold=0.8),
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.7, threshold=0.8),
            ],
            total_cases=3,
        )
        agg.compute_aggregates()
        assert MetricType.ACCURACY in agg.mean_scores
        assert agg.mean_scores[MetricType.ACCURACY] == pytest.approx(0.8)
        assert agg.min_scores[MetricType.ACCURACY] == 0.7
        assert agg.max_scores[MetricType.ACCURACY] == 0.9
        assert agg.pass_rates[MetricType.ACCURACY] == pytest.approx(2 / 3)

    def test_compute_aggregates_multiple_types(self) -> None:
        agg = AggregatedMetrics(
            metric_scores=[
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9),
                MetricScore(metric_type=MetricType.F1, name="f1", value=0.85),
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8),
                MetricScore(metric_type=MetricType.F1, name="f1", value=0.75),
            ],
        )
        agg.compute_aggregates()
        assert MetricType.ACCURACY in agg.mean_scores
        assert MetricType.F1 in agg.mean_scores
        assert agg.mean_scores[MetricType.ACCURACY] == pytest.approx(0.85)
        assert agg.mean_scores[MetricType.F1] == pytest.approx(0.8)

    def test_std_dev_calculation(self) -> None:
        agg = AggregatedMetrics(
            metric_scores=[
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=1.0),
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.0),
            ],
        )
        agg.compute_aggregates()
        # std dev of [1.0, 0.0] with mean 0.5 is 0.5
        assert agg.std_scores[MetricType.ACCURACY] == pytest.approx(0.5)

    def test_single_value_std_dev(self) -> None:
        agg = AggregatedMetrics(
            metric_scores=[
                MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.8),
            ],
        )
        agg.compute_aggregates()
        assert agg.std_scores[MetricType.ACCURACY] == 0.0


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_minimal_result(self) -> None:
        case_id = uuid4()
        result = EvaluationResult(case_id=case_id, status=EvaluationStatus.PASSED)
        assert result.case_id == case_id
        assert result.status == EvaluationStatus.PASSED
        assert result.metric_scores == []
        assert isinstance(result.started_at, datetime)

    def test_full_result(self) -> None:
        case_id = uuid4()
        run_id = uuid4()
        result = EvaluationResult(
            case_id=case_id,
            run_id=run_id,
            status=EvaluationStatus.FAILED,
            actual_output="Wrong answer",
            actual_tool_calls=[{"tool": "search"}],
            latency_ms=1500.0,
            error="Assertion failed",
            metadata={"attempt": 1},
        )
        assert result.run_id == run_id
        assert result.status == EvaluationStatus.FAILED
        assert result.actual_output == "Wrong answer"
        assert len(result.actual_tool_calls) == 1
        assert result.latency_ms == 1500.0
        assert result.error == "Assertion failed"
        assert result.metadata == {"attempt": 1}

    def test_add_metric_score(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        score = MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9)
        result.add_metric_score(score)
        assert len(result.metric_scores) == 1
        assert result.metric_scores[0] is score

    def test_get_metric(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        score1 = MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9)
        score2 = MetricScore(metric_type=MetricType.F1, name="f1", value=0.85)
        result.add_metric_score(score1)
        result.add_metric_score(score2)

        found = result.get_metric(MetricType.ACCURACY)
        assert found is score1

        not_found = result.get_metric(MetricType.PRECISION)
        assert not_found is None

    def test_is_passed(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result.add_metric_score(
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8)
        )
        result.add_metric_score(
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.85, threshold=0.8)
        )
        assert result.is_passed() is True

    def test_is_passed_with_failure(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result.add_metric_score(
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8)
        )
        result.add_metric_score(
            MetricScore(metric_type=MetricType.F1, name="f1", value=0.7, threshold=0.8)
        )
        assert result.is_passed() is False

    def test_is_passed_no_thresholds(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result.add_metric_score(MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9))
        # No thresholds = vacuously true
        assert result.is_passed() is True

    def test_finalize(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result.finalize(status=EvaluationStatus.FAILED, error="Failed")
        assert result.completed_at is not None
        assert result.status == EvaluationStatus.FAILED
        assert result.error == "Failed"

    def test_finalize_error_status(self) -> None:
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result.finalize(error="Exception occurred")
        assert result.status == EvaluationStatus.ERROR
        assert result.error == "Exception occurred"


class TestExperimentResult:
    """Tests for ExperimentResult model."""

    def test_minimal_experiment(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        assert exp.config_name == "config1"
        assert exp.dataset_name == "dataset1"
        assert exp.case_results == []
        assert exp.status == "in_progress"

    def test_add_case_result(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        result1 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED, latency_ms=100)
        result2 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.FAILED, latency_ms=200)
        exp.add_case_result(result1)
        exp.add_case_result(result2)
        assert len(exp.case_results) == 2
        assert exp.total_latency_ms == 300

    def test_finalize(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        exp.add_case_result(
            EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        )
        exp.add_case_result(
            EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        )
        exp.add_case_result(
            EvaluationResult(case_id=uuid4(), status=EvaluationStatus.FAILED)
        )
        exp.add_case_result(
            EvaluationResult(case_id=uuid4(), status=EvaluationStatus.ERROR)
        )
        exp.finalize(status="completed")

        assert exp.status == "completed"
        assert exp.completed_at is not None
        assert exp.aggregated_metrics.total_cases == 4
        assert exp.aggregated_metrics.passed_cases == 2
        assert exp.aggregated_metrics.failed_cases == 1
        assert exp.aggregated_metrics.error_cases == 1

    def test_finalize_with_error(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        exp.finalize(status="failed", error="Dataset not found")
        assert exp.status == "failed"
        assert exp.error == "Dataset not found"

    def test_get_pass_rate(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        assert exp.get_pass_rate() == 0.0

        exp.add_case_result(EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED))
        exp.add_case_result(EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED))
        exp.add_case_result(EvaluationResult(case_id=uuid4(), status=EvaluationStatus.FAILED))
        exp.finalize()
        assert exp.get_pass_rate() == pytest.approx(2 / 3)

    def test_aggregated_metrics_computed(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        result = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result.add_metric_score(
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8)
        )
        exp.add_case_result(result)
        exp.finalize()

        agg = exp.aggregated_metrics
        assert MetricType.ACCURACY in agg.mean_scores
        assert agg.mean_scores[MetricType.ACCURACY] == 0.9
        assert agg.pass_rates[MetricType.ACCURACY] == 1.0
