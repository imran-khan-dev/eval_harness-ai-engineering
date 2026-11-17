"""Execution models for system evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from eval_harness.tracing.models import CostEstimate, ExecutionTrace, TokenUsage

if TYPE_CHECKING:
    from eval_harness.metrics.models import ExperimentResult


class SystemExecutionStatus(str, Enum):
    """Status of a system execution attempt.

    Represents the lifecycle of a single case execution through the system.
    """

    SUCCESS = "success"
    """Execution completed and produced a prediction."""

    ERROR = "error"
    """Execution failed with an exception or unrecoverable error."""

    TIMEOUT = "timeout"
    """Execution exceeded allowed time limit."""

    PARTIAL = "partial"
    """Execution produced a prediction but trace is incomplete."""


class SystemExecutionResult(BaseModel):
    """Result of executing a single evaluation case through the system.

    This model captures the complete outcome of one system execution,
    including the prediction, execution trace, status, and metadata.
    It is designed to be provider-agnostic and reusable across different
    system types (functions, LLM apps, agents, orchestrators).
    """

    id: UUID = Field(default_factory=uuid4, description="Unique execution result identifier")
    case_id: UUID = Field(..., description="ID of the evaluation case that was executed")
    run_id: UUID | None = Field(None, description="Optional run/session identifier")
    status: SystemExecutionStatus = Field(..., description="Execution status")
    prediction: str | None = Field(None, description="System's output/prediction for the case")
    trace: ExecutionTrace | None = Field(
        None, description="Complete execution trace"
    )
    error: str | None = Field(None, description="Error message if execution failed")
    latency_ms: float | None = Field(
        None, ge=0, description="Total wall-clock latency in milliseconds"
    )
    token_usage: TokenUsage | None = Field(
        None, description="Aggregated token usage from trace"
    )
    cost_estimate: CostEstimate | None = Field(
        None, description="Estimated cost from trace"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Execution start timestamp",
    )
    completed_at: datetime | None = Field(
        None, description="Execution completion timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )

    @model_validator(mode="after")
    def extract_aggregates_from_trace(self) -> SystemExecutionResult:
        """Extract token usage and cost from trace if not explicitly set."""
        if self.trace is not None:
            if self.token_usage is None:
                self.token_usage = self.trace.total_tokens
            if self.cost_estimate is None:
                self.cost_estimate = self.trace.total_cost
        return self

    @model_validator(mode="after")
    def validate_error_consistency(self) -> SystemExecutionResult:
        """Ensure error field is present for ERROR status."""
        if self.status == SystemExecutionStatus.ERROR and not self.error:
            self.error = "Execution failed without error message"
        return self

    def finalize(
        self,
        status: SystemExecutionStatus | None = None,
        error: str | None = None,
    ) -> None:
        """Mark the execution as complete.

        Idempotent: subsequent calls after the first have no effect.
        """
        if self.completed_at is not None:
            return
        self.completed_at = datetime.now(timezone.utc)
        if status:
            self.status = status
        if error:
            self.error = error
            if status is None:
                self.status = SystemExecutionStatus.ERROR

    def is_successful(self) -> bool:
        """Check if execution completed successfully with a prediction."""
        return (
            self.status == SystemExecutionStatus.SUCCESS
            and self.prediction is not None
        )

    def has_trace(self) -> bool:
        """Check if a trace is available."""
        return self.trace is not None and len(self.trace.events) > 0

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }


class RunnerExperimentResult(BaseModel):
    """Result of running an experiment with a SystemUnderEvaluation.

    This model captures the complete outcome of executing a dataset through
    a system, including all individual execution results and aggregate statistics.
    It is designed to be provider-agnostic and can be converted to an
    EvaluationResult-based ExperimentResult for later metric evaluation.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique experiment identifier")
    dataset_id: UUID = Field(..., description="ID of the dataset that was executed")
    dataset_name: str = Field(..., description="Name of the dataset")
    run_id: UUID | None = Field(None, description="Optional run/session identifier")
    execution_results: list[SystemExecutionResult] = Field(
        default_factory=list, description="Individual system execution results"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Experiment start timestamp"
    )
    completed_at: datetime | None = Field(None, description="Experiment completion timestamp")
    status: str = Field(default="in_progress", description="Experiment status")
    error: str | None = Field(None, description="Error message if experiment failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_execution_result(self, result: SystemExecutionResult) -> None:
        """Add an execution result and update counts."""
        self.execution_results.append(result)

    def finalize(self, status: str = "completed", error: str | None = None) -> None:
        """Mark the experiment as complete and compute aggregates.

        Idempotent: subsequent calls after the first have no effect.
        """
        if self.completed_at is not None:
            return
        self.completed_at = datetime.now(timezone.utc)
        self.status = status
        self.error = error

    @property
    def total_cases(self) -> int:
        """Total number of cases executed."""
        return len(self.execution_results)

    @property
    def successful_cases(self) -> int:
        """Number of cases that completed successfully."""
        return sum(1 for r in self.execution_results if r.is_successful())

    @property
    def failed_cases(self) -> int:
        """Number of cases that failed (ERROR status)."""
        return sum(
            1 for r in self.execution_results if r.status == SystemExecutionStatus.ERROR
        )

    @property
    def timeout_cases(self) -> int:
        """Number of cases that timed out."""
        return sum(
            1 for r in self.execution_results if r.status == SystemExecutionStatus.TIMEOUT
        )

    @property
    def partial_cases(self) -> int:
        """Number of cases with partial results."""
        return sum(
            1 for r in self.execution_results if r.status == SystemExecutionStatus.PARTIAL
        )

    @property
    def success_rate(self) -> float:
        """Fraction of cases that completed successfully."""
        if self.total_cases == 0:
            return 0.0
        return self.successful_cases / self.total_cases

    def get_execution_result(self, case_id: UUID) -> SystemExecutionResult | None:
        """Get execution result for a specific case."""
        for result in self.execution_results:
            if result.case_id == case_id:
                return result
        return None

    def to_metrics_experiment_result(
        self,
        config_name: str = "runner_experiment",
    ) -> ExperimentResult:
        """Convert to metrics ExperimentResult for evaluation.

        Creates EvaluationResult objects from SystemExecutionResult for each case.
        Note: This creates minimal EvaluationResult objects without metric scores.
        Evaluators should be run separately to populate metric_scores.

        Args:
            config_name: Name to use for the evaluation config.

        Returns:
            ExperimentResult compatible with the metrics module.
        """
        # Import here to avoid circular dependency
        from eval_harness.metrics.models import (
            EvaluationResult,
            EvaluationStatus,
            ExperimentResult,
        )

        exp_result = ExperimentResult(
            config_name=config_name,
            dataset_name=self.dataset_name,
            total_tokens=None,
            total_cost=None,
            completed_at=self.completed_at,
            error=self.error,
        )
        exp_result.id = self.id
        exp_result.started_at = self.started_at
        exp_result.status = self.status
        exp_result.metadata = self.metadata

        for exec_result in self.execution_results:
            # Map system execution status to evaluation status
            if exec_result.is_successful():
                eval_status = EvaluationStatus.PASSED
            elif exec_result.status == SystemExecutionStatus.ERROR:
                eval_status = EvaluationStatus.ERROR
            elif exec_result.status == SystemExecutionStatus.TIMEOUT:
                eval_status = EvaluationStatus.FAILED
            elif exec_result.status == SystemExecutionStatus.PARTIAL:
                eval_status = EvaluationStatus.PARTIAL
            else:
                eval_status = EvaluationStatus.ERROR

            eval_result = EvaluationResult(
                id=exec_result.id,
                case_id=exec_result.case_id,
                run_id=exec_result.run_id or self.run_id,
                status=eval_status,
                actual_output=exec_result.prediction,
                actual_trajectory=None,
                trace=exec_result.trace,
                latency_ms=exec_result.latency_ms,
                token_usage=exec_result.token_usage,
                cost_estimate=exec_result.cost_estimate,
                error=exec_result.error,
                started_at=exec_result.started_at,
                completed_at=exec_result.completed_at,
                metadata=exec_result.metadata,
            )
            exp_result.add_case_result(eval_result)

        exp_result.finalize(status=self.status, error=self.error)
        return exp_result

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }
