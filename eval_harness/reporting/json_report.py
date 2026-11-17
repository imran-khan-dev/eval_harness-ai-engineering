"""JSON report generation for experiment results."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from eval_harness.metrics.models import (
    AggregatedMetrics,
    EvaluationResult,
    ExperimentResult,
    MetricScore,
    MetricType,
)
from eval_harness.runner.models import RunnerExperimentResult, SystemExecutionResult


def _serialize_datetime(dt: datetime | None) -> str | None:
    """Serialize datetime to ISO format string."""
    if dt is None:
        return None
    return dt.isoformat()


def _serialize_token_usage(token_usage: Any) -> dict[str, Any] | None:
    """Serialize TokenUsage to dict."""
    if token_usage is None:
        return None
    if hasattr(token_usage, "model_dump"):
        return cast(dict[str, Any], token_usage.model_dump())
    return cast(dict[str, Any], dict(token_usage))


def _serialize_cost_estimate(cost_estimate: Any) -> dict[str, Any] | None:
    """Serialize CostEstimate to dict."""
    if cost_estimate is None:
        return None
    if hasattr(cost_estimate, "model_dump"):
        return cast(dict[str, Any], cost_estimate.model_dump())
    return cast(dict[str, Any], dict(cost_estimate))


def _serialize_metric_score(score: MetricScore) -> dict[str, Any]:
    """Serialize MetricScore to dict."""
    metric_type_val: str = (
        score.metric_type.value
        if hasattr(score.metric_type, "value")
        else score.metric_type
    )
    return {
        "metric_type": metric_type_val,
        "name": score.name,
        "value": score.value,
        "threshold": score.threshold,
        "passed": score.passed,
        "details": score.details,
        "metadata": score.metadata,
    }


def _serialize_evaluation_result(result: EvaluationResult) -> dict[str, Any]:
    """Serialize EvaluationResult to dict."""
    status_val: str = (
        result.status.value if hasattr(result.status, "value") else result.status
    )
    return {
        "id": str(result.id),
        "case_id": str(result.case_id),
        "run_id": str(result.run_id) if result.run_id else None,
        "status": status_val,
        "metric_scores": [_serialize_metric_score(s) for s in result.metric_scores],
        "actual_output": result.actual_output,
        "actual_tool_calls": result.actual_tool_calls,
        "actual_trajectory": result.actual_trajectory,
        "latency_ms": result.latency_ms,
        "token_usage": _serialize_token_usage(result.token_usage),
        "cost_estimate": _serialize_cost_estimate(result.cost_estimate),
        "error": result.error,
        "started_at": _serialize_datetime(result.started_at),
        "completed_at": _serialize_datetime(result.completed_at),
        "metadata": result.metadata,
    }


def _serialize_system_execution_result(result: SystemExecutionResult) -> dict[str, Any]:
    """Serialize SystemExecutionResult to dict."""
    status_val = result.status.value if hasattr(result.status, "value") else result.status
    return {
        "id": str(result.id),
        "case_id": str(result.case_id),
        "run_id": str(result.run_id) if result.run_id else None,
        "status": status_val,
        "prediction": result.prediction,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "token_usage": _serialize_token_usage(result.token_usage),
        "cost_estimate": _serialize_cost_estimate(result.cost_estimate),
        "started_at": _serialize_datetime(result.started_at),
        "completed_at": _serialize_datetime(result.completed_at),
        "metadata": result.metadata,
    }


def _serialize_aggregated_metrics(agg: AggregatedMetrics) -> dict[str, Any]:
    """Serialize AggregatedMetrics to dict."""

    def _key_str(k: Any) -> str:
        val = k.value if hasattr(k, "value") else k
        return str(val)

    return {
        "mean_scores": {_key_str(k): v for k, v in agg.mean_scores.items()},
        "std_scores": {_key_str(k): v for k, v in agg.std_scores.items()},
        "min_scores": {_key_str(k): v for k, v in agg.min_scores.items()},
        "max_scores": {_key_str(k): v for k, v in agg.max_scores.items()},
        "pass_rates": {_key_str(k): v for k, v in agg.pass_rates.items()},
        "total_cases": agg.total_cases,
        "passed_cases": agg.passed_cases,
        "failed_cases": agg.failed_cases,
        "error_cases": agg.error_cases,
    }


class JSONReportGenerator:
    """Generates JSON reports from experiment results.

    Supports both ExperimentResult (metrics-based) and RunnerExperimentResult
    (execution-based) result types.
    """

    def __init__(self, include_traces: bool = False, indent: int = 2):
        """Initialize the JSON report generator.

        Args:
            include_traces: Whether to include full trace data in the report.
                Defaults to False to keep reports compact.
            indent: JSON indentation level. Defaults to 2.
        """
        self.include_traces = include_traces
        self.indent = indent

    def generate(self, result: ExperimentResult | RunnerExperimentResult) -> dict[str, Any]:
        """Generate a JSON-serializable report from an experiment result.

        Args:
            result: The experiment result to report on. Can be either
                ExperimentResult (with evaluation scores) or
                RunnerExperimentResult (with execution results only).

        Returns:
            A dictionary containing the complete report, ready for JSON serialization.
        """
        if isinstance(result, ExperimentResult):
            return self._generate_from_experiment_result(result)
        elif isinstance(result, RunnerExperimentResult):
            return self._generate_from_runner_result(result)
        else:
            raise TypeError(
                f"Unsupported result type: {type(result).__name__}. "
                f"Expected ExperimentResult or RunnerExperimentResult."
            )

    def _generate_from_experiment_result(self, result: ExperimentResult) -> dict[str, Any]:
        """Generate report from ExperimentResult (metrics-based)."""
        # Compute experiment-level aggregates
        all_scores = []
        for case_result in result.case_results:
            all_scores.extend(case_result.metric_scores)

        # Group scores by metric type for summary
        scores_by_type: dict[MetricType, list[MetricScore]] = defaultdict(list)
        for score in all_scores:
            scores_by_type[score.metric_type].append(score)

        # Build per-metric-type summary
        metric_summaries = {}
        for mtype, scores in scores_by_type.items():
            values = [s.value for s in scores]
            thresholds = [s.threshold for s in scores if s.threshold is not None]
            passed_count = sum(1 for s in scores if s.passed)

            # Handle both MetricType enum and string keys
            mtype_key = mtype.value if hasattr(mtype, "value") else mtype

            metric_summaries[mtype_key] = {
                "count": len(scores),
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "pass_rate": passed_count / len(scores) if scores else 0.0,
                "thresholds": thresholds if thresholds else None,
            }

        # Build cost and token summaries
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        latencies = []

        for case_result in result.case_results:
            if case_result.cost_estimate:
                if hasattr(case_result.cost_estimate, "total_cost_usd"):
                    total_cost += case_result.cost_estimate.total_cost_usd
                elif isinstance(case_result.cost_estimate, dict):
                    total_cost += case_result.cost_estimate.get("total_cost_usd", 0.0)

            if case_result.token_usage:
                if hasattr(case_result.token_usage, "prompt_tokens"):
                    total_input_tokens += case_result.token_usage.prompt_tokens
                    total_output_tokens += case_result.token_usage.completion_tokens
                    total_cached_tokens += case_result.token_usage.cached_tokens
                elif isinstance(case_result.token_usage, dict):
                    total_input_tokens += case_result.token_usage.get("prompt_tokens", 0)
                    total_output_tokens += case_result.token_usage.get("completion_tokens", 0)
                    total_cached_tokens += case_result.token_usage.get("cached_tokens", 0)

            if case_result.latency_ms:
                latencies.append(case_result.latency_ms)

        # Status counts
        status_counts: dict[str, int] = defaultdict(int)
        for case_result in result.case_results:
            status_val: str = (
                case_result.status.value
                if hasattr(case_result.status, "value")
                else case_result.status
            )
            status_counts[status_val] += 1

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_format": "json",
                "report_version": "1.0",
            },
            "experiment": {
                "id": str(result.id),
                "config_name": result.config_name,
                "dataset_name": result.dataset_name,
                "status": result.status,
                "error": result.error,
                "started_at": _serialize_datetime(result.started_at),
                "completed_at": _serialize_datetime(result.completed_at),
                "total_latency_ms": result.total_latency_ms,
                "metadata": result.metadata,
            },
            "execution_summary": {
                "total_cases": result.aggregated_metrics.total_cases,
                "passed_cases": result.aggregated_metrics.passed_cases,
                "failed_cases": result.aggregated_metrics.failed_cases,
                "error_cases": result.aggregated_metrics.error_cases,
                "pass_rate": result.get_pass_rate(),
                "status_breakdown": dict(status_counts),
            },
            "metric_summary": metric_summaries,
            "aggregated_metrics": _serialize_aggregated_metrics(result.aggregated_metrics),
            "cost_summary": {
                "total_cost_usd": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_tokens": total_input_tokens + total_output_tokens + total_cached_tokens,
            },
            "latency_summary": {
                "total_cases_with_latency": len(latencies),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "min_latency_ms": min(latencies) if latencies else 0.0,
                "max_latency_ms": max(latencies) if latencies else 0.0,
                "total_latency_ms": result.total_latency_ms,
            },
            "case_results": [
                _serialize_evaluation_result(cr) for cr in result.case_results
            ],
        }

        return report

    def _generate_from_runner_result(self, result: RunnerExperimentResult) -> dict[str, Any]:
        """Generate report from RunnerExperimentResult (execution-based)."""
        # Compute execution aggregates
        successful = sum(1 for r in result.execution_results if r.is_successful())
        failed = sum(
            1
            for r in result.execution_results
            if (r.status.value if hasattr(r.status, "value") else r.status) == "error"
        )
        timeout = sum(
            1
            for r in result.execution_results
            if (r.status.value if hasattr(r.status, "value") else r.status) == "timeout"
        )
        partial = sum(
            1
            for r in result.execution_results
            if (r.status.value if hasattr(r.status, "value") else r.status) == "partial"
        )

        # Build cost and token summaries
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        latencies = []

        for exec_result in result.execution_results:
            if exec_result.cost_estimate:
                if hasattr(exec_result.cost_estimate, "total_cost_usd"):
                    total_cost += exec_result.cost_estimate.total_cost_usd
                elif isinstance(exec_result.cost_estimate, dict):
                    total_cost += exec_result.cost_estimate.get("total_cost_usd", 0.0)

            if exec_result.token_usage:
                if hasattr(exec_result.token_usage, "prompt_tokens"):
                    total_input_tokens += exec_result.token_usage.prompt_tokens
                    total_output_tokens += exec_result.token_usage.completion_tokens
                    total_cached_tokens += exec_result.token_usage.cached_tokens
                elif isinstance(exec_result.token_usage, dict):
                    total_input_tokens += exec_result.token_usage.get("prompt_tokens", 0)
                    total_output_tokens += exec_result.token_usage.get("completion_tokens", 0)
                    total_cached_tokens += exec_result.token_usage.get("cached_tokens", 0)

            if exec_result.latency_ms:
                latencies.append(exec_result.latency_ms)

        # Status breakdown
        status_counts: dict[str, int] = defaultdict(int)
        for exec_result in result.execution_results:
            status_val = (
                exec_result.status.value
                if hasattr(exec_result.status, "value")
                else exec_result.status
            )
            status_counts[status_val] += 1

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_format": "json",
                "report_version": "1.0",
            },
            "experiment": {
                "id": str(result.id),
                "dataset_id": str(result.dataset_id),
                "dataset_name": result.dataset_name,
                "run_id": str(result.run_id) if result.run_id else None,
                "status": result.status,
                "error": result.error,
                "started_at": _serialize_datetime(result.started_at),
                "completed_at": _serialize_datetime(result.completed_at),
                "metadata": result.metadata,
            },
            "execution_summary": {
                "total_cases": result.total_cases,
                "successful_cases": successful,
                "failed_cases": failed,
                "timeout_cases": timeout,
                "partial_cases": partial,
                "success_rate": result.success_rate,
                "status_breakdown": dict(status_counts),
            },
            "cost_summary": {
                "total_cost_usd": total_cost,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_tokens": total_input_tokens + total_output_tokens + total_cached_tokens,
            },
            "latency_summary": {
                "total_cases_with_latency": len(latencies),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "min_latency_ms": min(latencies) if latencies else 0.0,
                "max_latency_ms": max(latencies) if latencies else 0.0,
            },
            "execution_results": [
                _serialize_system_execution_result(er) for er in result.execution_results
            ],
        }

        return report

    def generate_json(self, result: ExperimentResult | RunnerExperimentResult) -> str:
        """Generate a JSON string report from an experiment result.

        Args:
            result: The experiment result to report on.

        Returns:
            A JSON string containing the complete report.
        """
        import json

        report = self.generate(result)
        return json.dumps(report, indent=self.indent, default=str)

    def write_json(
        self, result: ExperimentResult | RunnerExperimentResult, filepath: str
    ) -> None:
        """Write a JSON report to a file.

        Args:
            result: The experiment result to report on.
            filepath: Path to write the JSON report.
        """
        import json

        report = self.generate(result)
        Path(filepath).write_text(
            json.dumps(report, indent=self.indent, default=str), encoding="utf-8"
        )


__all__ = ["JSONReportGenerator"]
