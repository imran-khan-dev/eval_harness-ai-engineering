"""Baseline comparator."""

from __future__ import annotations

from typing import Any

from eval_harness.baselines.models import (
    BaselineComparisonResult,
    CaseComparison,
    ChangeDirection,
    ChangeOutcome,
    ExperimentSummary,
    MetricComparison,
    MetricDirection,
    metric_direction,
)
from eval_harness.metrics.models import ExperimentResult


class BaselineComparator:
    def __init__(self, metric_directions: dict[str, Any] | None = None) -> None:
        self.metric_directions = metric_directions or {}

    def _direction(self, metric_type: Any, name: str) -> Any:
        key = name or str(metric_type)
        if key in self.metric_directions:
            return self.metric_directions[key]
        return metric_direction(metric_type, name)

    def _summary(self, exp: ExperimentResult) -> ExperimentSummary:
        # success rate from case statuses
        total = len(exp.case_results)
        succ = sum(1 for c in exp.case_results if str(c.status) in ("passed", "EvaluationStatus.PASSED"))
        sr = succ / total if total else None
        cost = None
        if exp.total_cost is not None:
            cost = getattr(exp.total_cost, "total_cost", None) or getattr(exp.total_cost, "cost", None)
            if cost is None:
                try:
                    cost = float(exp.total_cost)
                except Exception:
                    cost = None
        return ExperimentSummary(
            experiment_id=str(exp.id),
            dataset_name=exp.dataset_name,
            success_rate=sr,
            total_cost=cost,
            total_latency_ms=exp.total_latency_ms,
            metadata=dict(exp.metadata),
        )

    def compare(self, baseline: ExperimentResult, current: ExperimentResult) -> BaselineComparisonResult:
        # metric aggregation means
        base_means = baseline.aggregated_metrics.mean_scores
        cur_means = current.aggregated_metrics.mean_scores

        # map metric type -> name (use type as name fallback)
        base_names: dict[Any, str] = {}
        for s in baseline.aggregated_metrics.metric_scores:
            base_names[s.metric_type] = s.name
        cur_names: dict[Any, str] = {}
        for s in current.aggregated_metrics.metric_scores:
            cur_names[s.metric_type] = s.name

        all_types = set(base_means) | set(cur_means)
        metric_comps: list[MetricComparison] = []
        for mt in all_types:
            name = base_names.get(mt) or cur_names.get(mt) or str(mt)
            bv = base_means.get(mt)
            cv = cur_means.get(mt)
            present_b = mt in base_means
            present_c = mt in cur_means
            abs_diff = None
            rel = None
            if bv is not None and cv is not None:
                abs_diff = cv - bv
                if bv != 0:
                    rel = (cv - bv) / bv * 100.0
            # direction
            if bv is None or cv is None:
                direction = ChangeDirection.UNCHANGED
            elif cv > bv:
                direction = ChangeDirection.INCREASED
            elif cv < bv:
                direction = ChangeDirection.DECREASED
            else:
                direction = ChangeDirection.UNCHANGED
            # outcome
            d = self._direction(mt, name)
            if bv is None or cv is None:
                outcome = ChangeOutcome.UNCHANGED
            elif direction == ChangeDirection.UNCHANGED:
                outcome = ChangeOutcome.UNCHANGED
            else:
                if d == MetricDirection.HIGHER_IS_BETTER:
                    outcome = ChangeOutcome.IMPROVED if direction == ChangeDirection.INCREASED else ChangeOutcome.REGRESSED
                else:
                    outcome = ChangeOutcome.IMPROVED if direction == ChangeDirection.DECREASED else ChangeOutcome.REGRESSED
            metric_comps.append(MetricComparison(
                metric_type=str(mt),
                name=name,
                baseline_value=bv,
                current_value=cv,
                absolute_difference=abs_diff,
                relative_change=rel,
                direction=direction,
                outcome=outcome,
                present_in_baseline=present_b,
                present_in_current=present_c,
            ))

        # case-level
        base_cases = {str(c.case_id): c for c in baseline.case_results}
        cur_cases = {str(c.case_id): c for c in current.case_results}
        improved: list[str] = []
        regressed: list[str] = []
        unchanged: list[str] = []
        added: list[str] = []
        removed: list[str] = []

        def case_score(c: Any) -> float | None:
            if not c.metric_scores:
                return None
            return float(sum(s.value for s in c.metric_scores) / len(c.metric_scores))

        for cid in cur_cases:
            if cid not in base_cases:
                added.append(cid)
                continue
            bs = case_score(base_cases[cid])
            cs = case_score(cur_cases[cid])
            if bs is None or cs is None:
                unchanged.append(cid)
                continue
            if cs > bs:
                improved.append(cid)
            elif cs < bs:
                regressed.append(cid)
            else:
                unchanged.append(cid)
        for cid in base_cases:
            if cid not in cur_cases:
                removed.append(cid)

        # success rate change
        bs = self._summary(baseline).success_rate
        cs = self._summary(current).success_rate
        sr_change = (cs - bs) if (bs is not None and cs is not None) else None

        bsum = self._summary(baseline)
        csum = self._summary(current)
        cost_change = None
        if bsum.total_cost is not None and csum.total_cost is not None:
            cost_change = csum.total_cost - bsum.total_cost
        lat_change = None
        if bsum.total_latency_ms is not None and csum.total_latency_ms is not None:
            lat_change = csum.total_latency_ms - bsum.total_latency_ms

        return BaselineComparisonResult(
            baseline=bsum,
            current=csum,
            metric_comparisons=metric_comps,
            improved_cases=improved,
            regressed_cases=regressed,
            unchanged_cases=unchanged,
            added_cases=added,
            removed_cases=removed,
            cost_change=cost_change,
            latency_change=lat_change,
            success_rate_change=sr_change,
        )
