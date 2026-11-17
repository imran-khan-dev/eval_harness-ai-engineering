"""Baseline comparison module."""

from eval_harness.baselines.comparator import BaselineComparator
from eval_harness.baselines.models import (
    BaselineComparisonResult,
    ChangeDirection,
    ChangeOutcome,
    MetricComparison,
    MetricDirection,
)

__all__ = [
    "BaselineComparator",
    "BaselineComparisonResult",
    "ChangeDirection",
    "ChangeOutcome",
    "MetricComparison",
    "MetricDirection",
]
