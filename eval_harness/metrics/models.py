"""Metrics models for evaluation results and scores."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOOL_SELECTION_ACCURACY = "tool_selection_accuracy"
    TOOL_CALL_CORRECTNESS = "tool_call_correctness"
    TRAJECTORY_SIMILARITY = "trajectory_similarity"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST_EFFICIENCY = "cost_efficiency"
    CUSTOM = "custom"


class EvaluationStatus(str, Enum):
    """Status of an evaluation case result."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class MetricScore(BaseModel):
    """A single metric score for an evaluation case.

    Attributes:
        metric_type: Type of metric.
        name: Human-readable metric name.
        value: Numeric score (typically 0.0 to 1.0 for quality metrics).
        threshold: Optional passing threshold.
        passed: Whether the score meets the threshold.
        details: Additional metric-specific details.
        metadata: Arbitrary metadata.
    """

    metric_type: MetricType = Field(..., description="Type of metric")
    name: str = Field(..., min_length=1, description="Metric name")
    value: float = Field(..., description="Metric value")
    threshold: float | None = Field(None, description="Passing threshold")
    passed: bool | None = Field(None, description="Whether threshold met")
    details: dict[str, Any] = Field(default_factory=dict, description="Metric details")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    @model_validator(mode="after")
    def compute_passed(self) -> MetricScore:
        if self.passed is None and self.threshold is not None:
            self.passed = self.value >= self.threshold
        return self

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across multiple evaluation cases.

    Attributes:
        metric_scores: List of individual metric scores.
        mean_scores: Mean value per metric type.
        std_scores: Standard deviation per metric type.
        min_scores: Minimum value per metric type.
        max_scores: Maximum value per metric type.
        pass_rates: Pass rate per metric type.
        total_cases: Total number of cases evaluated.
        passed_cases: Number of cases that passed overall.
        failed_cases: Number of cases that failed overall.
        error_cases: Number of cases with errors.
    """

    metric_scores: list[MetricScore] = Field(default_factory=list, description="All scores")
    mean_scores: dict[MetricType, float] = Field(
        default_factory=dict, description="Mean per metric type"
    )
    std_scores: dict[MetricType, float] = Field(
        default_factory=dict, description="Std dev per metric type"
    )
    min_scores: dict[MetricType, float] = Field(
        default_factory=dict, description="Min per metric type"
    )
    max_scores: dict[MetricType, float] = Field(
        default_factory=dict, description="Max per metric type"
    )
    pass_rates: dict[MetricType, float] = Field(
        default_factory=dict, description="Pass rate per metric type"
    )
    total_cases: int = Field(default=0, ge=0, description="Total cases")
    passed_cases: int = Field(default=0, ge=0, description="Passed cases")
    failed_cases: int = Field(default=0, ge=0, description="Failed cases")
    error_cases: int = Field(default=0, ge=0, description="Error cases")

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from metric scores."""
        import math

        by_type: dict[MetricType, list[float]] = defaultdict(list)
        pass_counts: dict[MetricType, int] = defaultdict(int)
        total_counts: dict[MetricType, int] = defaultdict(int)

        for score in self.metric_scores:
            by_type[score.metric_type].append(score.value)
            total_counts[score.metric_type] += 1
            if score.passed:
                pass_counts[score.metric_type] += 1

        for mtype, values in by_type.items():
            self.mean_scores[mtype] = sum(values) / len(values)
            self.min_scores[mtype] = min(values)
            self.max_scores[mtype] = max(values)
            if len(values) > 1:
                mean = self.mean_scores[mtype]
                self.std_scores[mtype] = math.sqrt(
                    sum((v - mean) ** 2 for v in values) / len(values)
                )
            else:
                self.std_scores[mtype] = 0.0
            self.pass_rates[mtype] = (
                pass_counts[mtype] / total_counts[mtype] if total_counts[mtype] > 0 else 0.0
            )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }


class EvaluationResult(BaseModel):
    """Result of evaluating a single case.

    Attributes:
        id: Unique result identifier.
        case_id: ID of the evaluated case.
        run_id: Optional run/session identifier.
        status: Overall evaluation status.
        metric_scores: List of metric scores for this case.
        trace: Optional execution trace.
        actual_output: Actual output produced.
        actual_tool_calls: Actual tool calls made.
        actual_trajectory: Actual trajectory taken.
        latency_ms: Total latency in milliseconds.
        token_usage: Aggregated token usage.
        cost_estimate: Estimated cost.
        error: Error message if status is ERROR.
        started_at: Evaluation start timestamp.
        completed_at: Evaluation completion timestamp.
        metadata: Additional arbitrary metadata.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique result identifier")
    case_id: UUID = Field(..., description="Evaluated case ID")
    run_id: UUID | None = Field(None, description="Run/session ID")
    status: EvaluationStatus = Field(..., description="Evaluation status")
    metric_scores: list[MetricScore] = Field(default_factory=list, description="Metric scores")
    trace: Any | None = Field(None, description="Execution trace (ExecutionTrace)")
    actual_output: str | None = Field(None, description="Actual output produced")
    actual_tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="Actual tool calls"
    )
    actual_trajectory: list[dict[str, Any]] | None = Field(
        None, description="Actual trajectory"
    )
    latency_ms: float | None = Field(None, ge=0, description="Total latency in ms")
    token_usage: Any | None = Field(None, description="Token usage (TokenUsage)")
    cost_estimate: Any | None = Field(None, description="Cost estimate (CostEstimate)")
    error: str | None = Field(None, description="Error message if failed")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Start timestamp"
    )
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def add_metric_score(self, score: MetricScore) -> None:
        """Add a metric score."""
        self.metric_scores.append(score)

    def get_metric(self, metric_type: MetricType) -> MetricScore | None:
        """Get a metric score by type."""
        for score in self.metric_scores:
            if score.metric_type == metric_type:
                return score
        return None

    def is_passed(self) -> bool:
        """Check if all thresholded metrics passed."""
        return all(
            score.passed for score in self.metric_scores if score.threshold is not None
        )

    def finalize(
        self, status: EvaluationStatus | None = None, error: str | None = None
    ) -> None:
        """Mark the evaluation as complete."""
        self.completed_at = datetime.now(timezone.utc)
        if status:
            self.status = status
        if error:
            self.error = error
            # Only override status to ERROR if no explicit status was provided
            if status is None:
                self.status = EvaluationStatus.ERROR

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }


class ExperimentResult(BaseModel):
    """Result of an experiment run across multiple cases.

    Attributes:
        id: Unique experiment identifier.
        config_name: Name of the evaluation config used.
        dataset_name: Name of the dataset evaluated.
        case_results: List of individual case results.
        aggregated_metrics: Aggregated metrics across all cases.
        total_latency_ms: Total experiment latency.
        total_tokens: Aggregated token usage.
        total_cost: Aggregated cost estimate.
        started_at: Experiment start timestamp.
        completed_at: Experiment completion timestamp.
        status: Overall experiment status.
        error: Error message if experiment failed.
        metadata: Additional arbitrary metadata.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique experiment identifier")
    config_name: str = Field(..., description="Evaluation config name")
    dataset_name: str = Field(..., description="Dataset name")
    case_results: list[EvaluationResult] = Field(
        default_factory=list, description="Individual case results"
    )
    aggregated_metrics: AggregatedMetrics = Field(
        default_factory=AggregatedMetrics, description="Aggregated metrics"
    )
    total_latency_ms: float = Field(default=0.0, ge=0, description="Total latency")
    total_tokens: Any | None = Field(None, description="Total tokens (TokenUsage)")
    total_cost: Any | None = Field(None, description="Total cost (CostEstimate)")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Start timestamp"
    )
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    status: str = Field(default="in_progress", description="Experiment status")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def add_case_result(self, result: EvaluationResult) -> None:
        """Add a case result and update aggregates."""
        self.case_results.append(result)
        if result.latency_ms:
            self.total_latency_ms += result.latency_ms

    def finalize(self, status: str = "completed", error: str | None = None) -> None:
        """Mark the experiment as complete and compute aggregates."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = status
        self.error = error

        # Update aggregated metrics
        agg = self.aggregated_metrics
        agg.total_cases = len(self.case_results)
        agg.passed_cases = sum(
            1 for r in self.case_results if r.status == EvaluationStatus.PASSED
        )
        agg.failed_cases = sum(
            1 for r in self.case_results if r.status == EvaluationStatus.FAILED
        )
        agg.error_cases = sum(
            1 for r in self.case_results if r.status == EvaluationStatus.ERROR
        )

        # Collect all metric scores
        for result in self.case_results:
            agg.metric_scores.extend(result.metric_scores)

        agg.compute_aggregates()

    def get_pass_rate(self) -> float:
        """Get overall pass rate."""
        if self.aggregated_metrics.total_cases == 0:
            return 0.0
        return self.aggregated_metrics.passed_cases / self.aggregated_metrics.total_cases

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }
