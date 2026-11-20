"""Metric aggregation utilities for combining and analyzing evaluation scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from eval_harness.metrics.models import MetricScore, MetricType


@dataclass(frozen=True)
class AggregationResult:
    """Result of aggregating multiple metric scores.

    Attributes:
        count: Number of scores aggregated.
        mean: Mean value.
        min: Minimum value.
        max: Maximum value.
        std_dev: Standard deviation (population).
        pass_count: Number of scores that passed their threshold.
        fail_count: Number of scores that failed their threshold.
        no_threshold_count: Number of scores without a threshold.
        pass_rate: Pass rate among scores with thresholds.
    """

    count: int
    mean: float
    min: float
    max: float
    std_dev: float
    pass_count: int
    fail_count: int
    no_threshold_count: int
    pass_rate: float

class MetricAggregator:
    """Aggregates multiple MetricScore objects into summary statistics.

    Supports both unweighted and weighted aggregation. Groups results
    by metric type and metric name for detailed analysis.
    """

    def __init__(self) -> None:
        self._scores: list[MetricScore] = []

    def add_score(self, score: MetricScore) -> None:
        """Add a metric score to the aggregator."""
        self._scores.append(score)

    def add_scores(self, scores: list[MetricScore]) -> None:
        """Add multiple metric scores."""
        self._scores.extend(scores)

    def clear(self) -> None:
        """Clear all added scores."""
        self._scores.clear()

    def _compute_stats(
        self,
        values: list[float],
        pass_count: int,
        fail_count: int,
        no_threshold_count: int,
    ) -> AggregationResult:
        """Compute statistics from a list of values."""
        count = len(values)
        if count == 0:
            return AggregationResult(
                count=0,
                mean=0.0,
                min=0.0,
                max=0.0,
                std_dev=0.0,
                pass_count=0,
                fail_count=0,
                no_threshold_count=0,
                pass_rate=0.0,
            )

        mean = sum(values) / count
        min_val = min(values)
        max_val = max(values)

        if count > 1:
            variance = sum((v - mean) ** 2 for v in values) / count
            std_dev = variance**0.5
        else:
            std_dev = 0.0

        total_with_threshold = pass_count + fail_count
        pass_rate = (
            pass_count / total_with_threshold if total_with_threshold > 0 else 0.0
        )

        return AggregationResult(
            count=count,
            mean=mean,
            min=min_val,
            max=max_val,
            std_dev=std_dev,
            pass_count=pass_count,
            fail_count=fail_count,
            no_threshold_count=no_threshold_count,
            pass_rate=pass_rate,
        )

    def aggregate(self) -> dict[str, Any]:
        """Aggregate all added scores.

        Returns:
            Dictionary containing:
            - overall: AggregationResult for all scores
            - by_type: Dict[MetricType, AggregationResult]
            - by_name: Dict[str, AggregationResult]
        """
        if not self._scores:
            return {
                "overall": self._compute_stats([], 0, 0, 0),
                "by_type": {},
                "by_name": {},
            }

        # Group by metric type
        by_type: dict[MetricType, list[MetricScore]] = defaultdict(list)
        # Group by metric name
        by_name: dict[str, list[MetricScore]] = defaultdict(list)

        for score in self._scores:
            by_type[score.metric_type].append(score)
            by_name[score.name].append(score)

        # Overall stats
        all_values = [s.value for s in self._scores]
        overall_pass = sum(1 for s in self._scores if s.passed is True)
        overall_fail = sum(1 for s in self._scores if s.passed is False)
        overall_no_threshold = sum(1 for s in self._scores if s.passed is None)

        overall = self._compute_stats(
            all_values, overall_pass, overall_fail, overall_no_threshold
        )

        # By type
        type_results = {}
        for mtype, scores in by_type.items():
            values = [s.value for s in scores]
            pass_count = sum(1 for s in scores if s.passed is True)
            fail_count = sum(1 for s in scores if s.passed is False)
            no_threshold_count = sum(1 for s in scores if s.passed is None)
            type_results[mtype] = self._compute_stats(
                values, pass_count, fail_count, no_threshold_count
            )

        # By name
        name_results = {}
        for name, scores in by_name.items():
            values = [s.value for s in scores]
            pass_count = sum(1 for s in scores if s.passed is True)
            fail_count = sum(1 for s in scores if s.passed is False)
            no_threshold_count = sum(1 for s in scores if s.passed is None)
            name_results[name] = self._compute_stats(
                values, pass_count, fail_count, no_threshold_count
            )

        return {
            "overall": overall,
            "by_type": type_results,
            "by_name": name_results,
        }

    def aggregate_weighted(
        self,
        weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Aggregate scores with optional weights per metric name.

        Args:
            weights: Mapping of metric name to weight. If a metric name
                is not in weights, default_weight is used.
            default_weight: Weight for metrics not explicitly weighted.

        Returns:
            Dictionary with same structure as aggregate() but with
            weighted mean and weighted pass rate where applicable.

        Raises:
            ValueError: If any weight is negative or sum of weights is zero.
        """

        # Handle empty scores
        if not self._scores:
            return {
                "overall": self._compute_stats([], 0, 0, 0),
                "by_type": {},
                "by_name": {},
            }
     
        # Set default weights
        if weights is None:
            weights = {}

        # Validate weights = A weight cannot be negative.
        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"Weight for '{name}' must be non-negative")

        # Group scores by metric name
        by_name: dict[str, list[MetricScore]] = defaultdict(list)
        for score in self._scores:
            by_name[score.name].append(score)

        # Compute total weight
        total_weight = 0.0
        for name, scores in by_name.items():
            weight = weights.get(name, default_weight)
            total_weight += weight * len(scores)

        if total_weight == 0:
            raise ValueError("Total weight is zero; cannot compute weighted average")

        # Weighted overall mean = This is the actual weighted average calculation.
        weighted_sum = 0.0
        for name, scores in by_name.items():
            weight = weights.get(name, default_weight)
            for score in scores:
                weighted_sum += score.value * weight

        weighted_mean = weighted_sum / total_weight

       
        # Calculate other statistics
        all_values = [s.value for s in self._scores]
        # Collects all the original scores.
        overall_pass = sum(1 for s in self._scores if s.passed is True)
        overall_fail = sum(1 for s in self._scores if s.passed is False)
        overall_no_threshold = sum(1 for s in self._scores if s.passed is None)

        # Weighted pass rate: sum of weights for passed / sum of weights for thresholded
        weighted_pass_weight = 0.0
        weighted_threshold_weight = 0.0
        for name, scores in by_name.items():
            weight = weights.get(name, default_weight)
            for score in scores:
                if score.passed is True:
                    weighted_pass_weight += weight
                    weighted_threshold_weight += weight
                elif score.passed is False:
                    weighted_threshold_weight += weight
                # No threshold scores don't contribute to pass rate

        weighted_pass_rate = (
            weighted_pass_weight / weighted_threshold_weight
            if weighted_threshold_weight > 0
            else 0.0
        )

        # Create the overall result
        overall = AggregationResult(
            count=len(self._scores),
            mean=weighted_mean,
            min=min(all_values) if all_values else 0.0,
            max=max(all_values) if all_values else 0.0,
            std_dev=0.0,  # Not meaningful for weighted
            pass_count=overall_pass,
            fail_count=overall_fail,
            no_threshold_count=overall_no_threshold,
            pass_rate=weighted_pass_rate,
        )

        # Group results by metric type
        type_results = {}
        for mtype, scores in defaultdict(list, {
            mtype: [s for s in self._scores if s.metric_type == mtype]
            for mtype in {s.metric_type for s in self._scores}
        }).items():
            values = [s.value for s in scores]
            pass_count = sum(1 for s in scores if s.passed is True)
            fail_count = sum(1 for s in scores if s.passed is False)
            no_threshold_count = sum(1 for s in scores if s.passed is None)
            type_results[mtype] = self._compute_stats(
                values, pass_count, fail_count, no_threshold_count
            )

        # Group results by metric name
        name_results = {}
        for name, scores in by_name.items():
            weight = weights.get(name, default_weight)
            values = [s.value for s in scores]
            pass_count = sum(1 for s in scores if s.passed is True)
            fail_count = sum(1 for s in scores if s.passed is False)
            no_threshold_count = sum(1 for s in scores if s.passed is None)

            # Weighted mean for this name group
            name_weighted_mean = sum(s.value for s in scores) / len(scores) if weight > 0 else 0.0

            name_results[name] = AggregationResult(
                count=len(scores),
                mean=name_weighted_mean,
                min=min(values) if values else 0.0,
                max=max(values) if values else 0.0,
                std_dev=0.0,
                pass_count=pass_count,
                fail_count=fail_count,
                no_threshold_count=no_threshold_count,
                pass_rate=(
                    pass_count / (pass_count + fail_count)
                    if (pass_count + fail_count) > 0
                    else 0.0
                ),
            )

        return {
            "overall": overall,
            "by_type": type_results,
            "by_name": name_results,
        }

def aggregate_scores(
    scores: list[MetricScore],
    weights: dict[str, float] | None = None,
    default_weight: float = 1.0,
) -> dict[str, Any]:
    """Convenience function to aggregate a list of metric scores.

    Args:
        scores: List of MetricScore objects to aggregate.
        weights: Optional mapping of metric name to weight.
        default_weight: Default weight for metrics not in weights.

    Returns:
        Aggregation result dictionary (weighted if weights provided).
    """
    aggregator = MetricAggregator()
    aggregator.add_scores(scores)

    if weights is not None:
        return aggregator.aggregate_weighted(weights, default_weight)
    return aggregator.aggregate()
