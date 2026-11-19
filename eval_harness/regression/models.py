from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_EVALUATED = "not_evaluated"


class RuleType(str, Enum):
    MIN_METRIC = "min_metric"
    MIN_PASS_RATE = "min_pass_rate"
    MAX_REGRESSION = "max_regression"
    MAX_LATENCY = "max_latency"
    MAX_COST = "max_cost"
    MAX_ERROR_RATE = "max_error_rate"
    REQUIRED_METRIC = "required_metric"


class QualityGateRule(BaseModel):
    id: str
    description: str = ""
    rule_type: RuleType
    metric: str | None = None
    threshold: float | None = None
    baseline_relative: bool = False

    model_config = {"extra": "forbid"}


class QualityGateConfig(BaseModel):
    rules: list[QualityGateRule] = Field(default_factory=list)
    missing_baseline_policy: GateStatus = GateStatus.NOT_EVALUATED
    missing_metric_policy: GateStatus = GateStatus.FAIL
    added_cases_policy: GateStatus = GateStatus.NOT_EVALUATED
    removed_cases_policy: GateStatus = GateStatus.NOT_EVALUATED

    model_config = {"extra": "forbid"}


class GateCheckResult(BaseModel):
    rule_id: str
    description: str
    status: GateStatus
    actual_value: float | None = None
    expected_threshold: float | None = None
    explanation: str = ""
    baseline_value: float | None = None
    current_value: float | None = None

    model_config = {"extra": "forbid"}


class QualityGateResult(BaseModel):
    passed: bool
    status: GateStatus
    checks: list[GateCheckResult] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
