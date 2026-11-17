"""Baseline comparison models."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChangeDirection(str, Enum):
    INCREASED = "increased"
    DECREASED = "decreased"
    UNCHANGED = "unchanged"


class ChangeOutcome(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


DEFAULT_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "accuracy": MetricDirection.HIGHER_IS_BETTER,
    "f1": MetricDirection.HIGHER_IS_BETTER,
    "precision": MetricDirection.HIGHER_IS_BETTER,
    "recall": MetricDirection.HIGHER_IS_BETTER,
    "exact_match": MetricDirection.HIGHER_IS_BETTER,
    "semantic_similarity": MetricDirection.HIGHER_IS_BETTER,
    "tool_selection_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "tool_call_correctness": MetricDirection.HIGHER_IS_BETTER,
    "trajectory_similarity": MetricDirection.HIGHER_IS_BETTER,
    "latency": MetricDirection.LOWER_IS_BETTER,
    "token_efficiency": MetricDirection.HIGHER_IS_BETTER,
    "cost_efficiency": MetricDirection.HIGHER_IS_BETTER,
    "cost": MetricDirection.LOWER_IS_BETTER,
    "error_rate": MetricDirection.LOWER_IS_BETTER,
}


def metric_direction(metric_type: Any, name: str | None = None) -> MetricDirection:
    if name and name.lower() in DEFAULT_METRIC_DIRECTIONS:
        return DEFAULT_METRIC_DIRECTIONS[name.lower()]
    key = str(metric_type).lower()
    if key in DEFAULT_METRIC_DIRECTIONS:
        return DEFAULT_METRIC_DIRECTIONS[key]
    return MetricDirection.HIGHER_IS_BETTER


class MetricComparison(BaseModel):
    metric_type: str
    name: str
    baseline_value: float | None = None
    current_value: float | None = None
    absolute_difference: float | None = None
    relative_change: float | None = None
    direction: ChangeDirection = ChangeDirection.UNCHANGED
    outcome: ChangeOutcome = ChangeOutcome.UNCHANGED
    present_in_baseline: bool = True
    present_in_current: bool = True

    model_config = {"extra": "forbid"}


class CaseComparison(BaseModel):
    case_id: str
    outcome: ChangeOutcome
    baseline_score: float | None = None
    current_score: float | None = None

    model_config = {"extra": "forbid"}


class ExperimentSummary(BaseModel):
    experiment_id: str
    dataset_name: str | None = None
    success_rate: float | None = None
    total_cost: float | None = None
    total_latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class BaselineComparisonResult(BaseModel):
    baseline: ExperimentSummary
    current: ExperimentSummary
    metric_comparisons: list[MetricComparison] = Field(default_factory=list)
    improved_cases: list[str] = Field(default_factory=list)
    regressed_cases: list[str] = Field(default_factory=list)
    unchanged_cases: list[str] = Field(default_factory=list)
    added_cases: list[str] = Field(default_factory=list)
    removed_cases: list[str] = Field(default_factory=list)
    cost_change: float | None = None
    latency_change: float | None = None
    success_rate_change: float | None = None

    model_config = {"extra": "forbid"}
