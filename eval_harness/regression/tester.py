from __future__ import annotations

from typing import Any

from eval_harness.baselines.comparator import BaselineComparator
from eval_harness.baselines.models import BaselineComparisonResult
from eval_harness.metrics.models import ExperimentResult
from eval_harness.regression.models import (
    GateCheckResult,
    GateStatus,
    QualityGateConfig,
    QualityGateResult,
    QualityGateRule,
    RuleType,
)


class RegressionTester:
    def __init__(self, config: QualityGateConfig) -> None:
        self.config = config

    def evaluate( 
        self,
        current: ExperimentResult,
        baseline: ExperimentResult | None = None,
    ) -> QualityGateResult:
        comparison: BaselineComparisonResult | None = None
        if baseline is not None:
            comparison = BaselineComparator().compare(baseline, current)

        checks: list[GateCheckResult] = []
        for rule in self.config.rules:
            checks.append(self._eval_rule(rule, current, baseline, comparison))

        # added/removed cases checks
        if comparison is not None:
            if comparison.added_cases:
                checks.append(GateCheckResult(
                    rule_id="added_cases",
                    description="Added cases",
                    status=self.config.added_cases_policy,
                    actual_value=float(len(comparison.added_cases)),
                    explanation=f"{len(comparison.added_cases)} added cases",
                ))
            if comparison.removed_cases:
                checks.append(GateCheckResult(
                    rule_id="removed_cases",
                    description="Removed cases",
                    status=self.config.removed_cases_policy,
                    actual_value=float(len(comparison.removed_cases)),
                    explanation=f"{len(comparison.removed_cases)} removed cases",
                ))

        overall_status = GateStatus.PASS
        for c in checks:
            if c.status == GateStatus.FAIL:
                overall_status = GateStatus.FAIL
            elif c.status == GateStatus.WARNING and overall_status == GateStatus.PASS:
                overall_status = GateStatus.WARNING
        passed = overall_status in (GateStatus.PASS, GateStatus.WARNING, GateStatus.NOT_EVALUATED)
        # FAIL means not passed
        if overall_status == GateStatus.FAIL:
            passed = False
        return QualityGateResult(passed=passed, status=overall_status, checks=checks)

    def _eval_rule(
        self,
        rule: QualityGateRule,
        current: ExperimentResult,
        baseline: ExperimentResult | None,
        comparison: BaselineComparisonResult | None,
    ) -> GateCheckResult:
        rt = rule.rule_type
        if rt == RuleType.REQUIRED_METRIC:
            present = any(str(s.metric_type) == rule.metric or s.name == rule.metric for s in current.aggregated_metrics.metric_scores) if False else any(
                (rule.metric is not None and (str(s.metric_type) == rule.metric or s.name == rule.metric))
                for s in current.aggregated_metrics.metric_scores
            )
            # also check metric_scores across cases
            if not present:
                present = any(
                    (rule.metric is not None and any((str(sc.metric_type) == rule.metric or sc.name == rule.metric) for sc in c.metric_scores))
                    for c in current.case_results
                )
            status = GateStatus.PASS if present else self.config.missing_metric_policy
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, explanation="required metric present" if present else "required metric missing")

        if rt == RuleType.MIN_METRIC:
            vals = [s.value for c in current.case_results for s in c.metric_scores if (rule.metric is None or str(s.metric_type) == rule.metric or s.name == rule.metric)]
            if not vals:
                # try aggregated
                mt = None
                for k, v in current.aggregated_metrics.mean_scores.items():
                    if rule.metric is None or str(k) == rule.metric:
                        vals = [v]
                        break
            if not vals:
                status = self.config.missing_metric_policy
                return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, explanation="metric not found")
            actual = sum(vals) / len(vals)
            thr = rule.threshold if rule.threshold is not None else 0.0
            status = GateStatus.PASS if actual >= thr else GateStatus.FAIL
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, actual_value=actual, expected_threshold=thr, current_value=actual, explanation=f"{actual} vs {thr}")

        if rt == RuleType.MIN_PASS_RATE:
            total = current.aggregated_metrics.total_cases
            passed = current.aggregated_metrics.passed_cases
            rate = (passed / total) if total else 0.0
            thr = rule.threshold if rule.threshold is not None else 0.0
            status = GateStatus.PASS if rate >= thr else GateStatus.FAIL
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, actual_value=rate, expected_threshold=thr, explanation=f"{rate} vs {thr}")

        if rt == RuleType.MAX_LATENCY:
            actual = current.total_latency_ms
            thr = rule.threshold if rule.threshold is not None else float("inf")
            status = GateStatus.PASS if actual <= thr else GateStatus.FAIL
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, actual_value=actual, expected_threshold=thr, explanation=f"{actual} vs {thr}")

        if rt == RuleType.MAX_COST:
            cost = None
            if current.total_cost is not None:
                cost = getattr(current.total_cost, "total_cost", None) or getattr(current.total_cost, "cost", None)
                if cost is None:
                    try:
                        cost = float(current.total_cost)
                    except Exception:
                        cost = None
            if cost is None:
                status = self.config.missing_metric_policy
                return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, explanation="cost not available")
            thr = rule.threshold if rule.threshold is not None else float("inf")
            status = GateStatus.PASS if cost <= thr else GateStatus.FAIL
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, actual_value=cost, expected_threshold=thr, explanation=f"{cost} vs {thr}")

        if rt == RuleType.MAX_ERROR_RATE:
            total = len(current.case_results)
            errors = sum(1 for c in current.case_results if str(c.status) in ("error", "EvaluationStatus.ERROR"))
            rate = (errors / total) if total else 0.0
            thr = rule.threshold if rule.threshold is not None else 0.0
            status = GateStatus.PASS if rate <= thr else GateStatus.FAIL
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, actual_value=rate, expected_threshold=thr, explanation=f"{rate} vs {thr}")

        if rt == RuleType.MAX_REGRESSION:
            if comparison is None:
                status = self.config.missing_baseline_policy
                return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, explanation="no baseline")
            # find metric comparison
            mc = None
            for m in comparison.metric_comparisons:
                if rule.metric is None or m.metric_type == rule.metric or m.name == rule.metric:
                    mc = m
                    break
            if mc is None or mc.absolute_difference is None:
                status = self.config.missing_metric_policy
                return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, explanation="metric not found in comparison")
            # regression amount (drop for higher-is-better)
            reg = 0.0
            if mc.absolute_difference is not None:
                # if higher is better, regression is negative diff
                reg = -mc.absolute_difference if mc.absolute_difference < 0 else 0.0
            # threshold is max allowed regression (positive)
            thr = rule.threshold if rule.threshold is not None else 0.0
            status = GateStatus.PASS if reg <= thr else GateStatus.FAIL
            return GateCheckResult(rule_id=rule.id, description=rule.description, status=status, actual_value=reg, expected_threshold=thr, baseline_value=mc.baseline_value, current_value=mc.current_value, explanation=f"regression {reg} vs {thr}")

        return GateCheckResult(rule_id=rule.id, description=rule.description, status=GateStatus.NOT_EVALUATED, explanation="unknown rule")
