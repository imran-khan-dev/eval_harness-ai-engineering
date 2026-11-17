"""Markdown report generation for experiment results."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_harness.metrics.models import (
    EvaluationResult,
    ExperimentResult,
    MetricScore,
    MetricType,
)
from eval_harness.runner.models import RunnerExperimentResult, SystemExecutionResult


def _format_datetime(dt: datetime | None) -> str:
    """Format datetime for markdown display."""
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_cost(cost: float | None) -> str:
    """Format cost for markdown display."""
    if cost is None or cost == 0.0:
        return "N/A"
    return f"${cost:.6f}"


def _format_latency(latency: float | None) -> str:
    """Format latency for markdown display."""
    if latency is None:
        return "N/A"
    if latency < 1000:
        return f"{latency:.1f} ms"
    return f"{latency / 1000:.2f} s"


def _format_tokens(tokens: int | None) -> str:
    """Format token count for markdown display."""
    if tokens is None or tokens == 0:
        return "N/A"
    return f"{tokens:,}"


def _status_badge(status: str | Any) -> str:
    """Return a markdown badge for a status."""
    status_val = status.value if hasattr(status, "value") else status
    badges = {
        "passed": "`✅ PASSED`",
        "failed": "`❌ FAILED`",
        "error": "`💥 ERROR`",
        "skipped": "`⏭️ SKIPPED`",
        "partial": "`⚠️ PARTIAL`",
        "success": "`✅ SUCCESS`",
        "timeout": "`⏱️ TIMEOUT`",
    }
    return badges.get(status_val.lower(), f"`{status_val.upper()}`")


def _metric_type_str(metric_type: str | Any) -> str:
    """Return string representation of metric type."""
    return metric_type.value if hasattr(metric_type, "value") else metric_type


def _build_metric_table(metric_summaries: dict[str, dict[str, Any]]) -> str:
    """Build a markdown table for metric summaries."""
    if not metric_summaries:
        return "*No metrics available*"

    lines = [
        "| Metric Type | Count | Mean | Min | Max | Pass Rate |",
        "|-------------|-------|------|-----|-----|-----------|",
    ]
    for mtype, summary in sorted(metric_summaries.items()):
        pass_rate = summary.get("pass_rate", 0.0)
        lines.append(
            f"| {mtype} | {summary['count']} | "
            f"{summary['mean']:.3f} | {summary['min']:.3f} | "
            f"{summary['max']:.3f} | {pass_rate:.1%} |"
        )
    return "\n".join(lines)


def _build_case_table(case_results: list[EvaluationResult]) -> str:
    """Build a markdown table for per-case results."""
    if not case_results:
        return "*No cases evaluated*"

    lines = [
        "| Case ID | Status | Metrics | Output | Error |",
        "|---------|--------|---------|--------|-------|",
    ]
    for result in case_results:
        # Summarize metrics
        metric_summary = []
        for score in result.metric_scores:
            passed_str = "✓" if score.passed else "✗" if score.passed is False else "–"
            metric_summary.append(
                f"{score.name}: {score.value:.2f} {passed_str}"
            )
        metrics_str = "<br>".join(metric_summary) if metric_summary else "—"

        # Truncate output
        output = result.actual_output or "—"
        if len(output) > 60:
            output = output[:57] + "..."

        error = result.error or "—"
        if len(error) > 60:
            error = error[:57] + "..."

        lines.append(
            f"| `{str(result.case_id)[:8]}` | "
            f"{_status_badge(result.status)} | "
            f"{metrics_str} | "
            f"{output} | "
            f"{error} |"
        )
    return "\n".join(lines)


def _build_execution_case_table(execution_results: list[SystemExecutionResult]) -> str:
    """Build a markdown table for execution results."""
    if not execution_results:
        return "*No cases executed*"

    lines = [
        "| Case ID | Status | Prediction | Latency | Tokens | Cost | Error |",
        "|---------|--------|------------|---------|--------|------|-------|",
    ]
    for result in execution_results:
        prediction = result.prediction or "—"
        if len(prediction) > 50:
            prediction = prediction[:47] + "..."

        latency = _format_latency(result.latency_ms)

        tokens = "N/A"
        if result.token_usage:
            if hasattr(result.token_usage, "total_tokens"):
                tokens = _format_tokens(result.token_usage.total_tokens)
            elif isinstance(result.token_usage, dict):
                tokens = _format_tokens(result.token_usage.get("total_tokens"))

        cost = "N/A"
        if result.cost_estimate:
            if hasattr(result.cost_estimate, "total_cost_usd"):
                cost = _format_cost(result.cost_estimate.total_cost_usd)
            elif isinstance(result.cost_estimate, dict):
                cost = _format_cost(result.cost_estimate.get("total_cost_usd"))

        error = result.error or "—"
        if len(error) > 50:
            error = error[:47] + "..."

        lines.append(
            f"| `{str(result.case_id)[:8]}` | "
            f"{_status_badge(result.status)} | "
            f"{prediction} | {latency} | {tokens} | {cost} | {error} |"
        )
    return "\n".join(lines)


def _build_failed_cases_section(case_results: list[EvaluationResult]) -> str:
    """Build a detailed section for failed/error cases."""
    failed = [
        r
        for r in case_results
        if (r.status.value if hasattr(r.status, "value") else r.status) in ("failed", "error")
    ]
    if not failed:
        return ""

    lines = ["\n## Failed Cases Details\n"]
    for result in failed:
        lines.append(f"### Case `{result.case_id}` ({_status_badge(result.status)})")
        lines.append(f"**Error:** {result.error or 'No error message'}")
        if result.metric_scores:
            lines.append("**Metric Scores:**")
            for score in result.metric_scores:
                passed_str = "✓" if score.passed else "✗" if score.passed is False else "–"
                lines.append(
                    f"- {score.name} ({_metric_type_str(score.metric_type)}): "
                    f"{score.value:.3f} {passed_str}"
                    + (f" (threshold: {score.threshold})" if score.threshold else "")
                )
        lines.append("")
    return "\n".join(lines)


def _build_skipped_cases_section(case_results: list[EvaluationResult]) -> str:
    """Build a section for skipped cases."""
    skipped = [
        r
        for r in case_results
        if (r.status.value if hasattr(r.status, "value") else r.status) == "skipped"
    ]
    if not skipped:
        return ""

    lines = ["\n## Skipped Cases\n"]
    for result in skipped:
        lines.append(f"- Case `{result.case_id}`: {result.error or 'No reason provided'}")
    return "\n".join(lines)


class MarkdownReportGenerator:
    """Generates Markdown reports from experiment results.

    Supports both ExperimentResult (metrics-based) and RunnerExperimentResult
    (execution-based) result types.
    """

    def __init__(self, include_traces: bool = False, max_cases_in_table: int = 50):
        """Initialize the Markdown report generator.

        Args:
            include_traces: Whether to include trace details. Currently not implemented
                for markdown (use JSON for full trace data).
            max_cases_in_table: Maximum number of cases to show in detail tables.
                Larger experiments will show summary only.
        """
        self.include_traces = include_traces
        self.max_cases_in_table = max_cases_in_table

    def generate(self, result: ExperimentResult | RunnerExperimentResult) -> str:
        """Generate a Markdown report from an experiment result.

        Args:
            result: The experiment result to report on.

        Returns:
            A Markdown string containing the complete report.
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

    def _generate_from_experiment_result(self, result: ExperimentResult) -> str:
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
            passed_count = sum(1 for s in scores if s.passed)

            # Handle both MetricType enum and string keys
            mtype_key = mtype.value if hasattr(mtype, "value") else mtype

            metric_summaries[mtype_key] = {
                "count": len(scores),
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "pass_rate": passed_count / len(scores) if scores else 0.0,
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

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Status counts
        status_counts: dict[str, int] = defaultdict(int)
        for case_result in result.case_results:
            status_val: str = (
                case_result.status.value
                if hasattr(case_result.status, "value")
                else case_result.status
            )
            status_counts[status_val] += 1

        lines = [
            f"# Experiment Report: {result.dataset_name}",
            "",
            f"**Config:** `{result.config_name}`  ",
            f"**Experiment ID:** `{result.id}`  ",
            f"**Status:** {_status_badge(result.status)}  ",
            f"**Started:** {_format_datetime(result.started_at)}  ",
            f"**Completed:** {_format_datetime(result.completed_at)}  ",
            "",
            "## Summary",
            "",
            f"- **Total Cases:** {result.aggregated_metrics.total_cases}",
            f"- **Passed:** {result.aggregated_metrics.passed_cases} "
            f"({result.get_pass_rate():.1%})",
            f"- **Failed:** {result.aggregated_metrics.failed_cases}",
            f"- **Errors:** {result.aggregated_metrics.error_cases}",
            f"- **Total Latency:** {_format_latency(result.total_latency_ms)}",
            f"- **Total Cost:** {_format_cost(total_cost)}",
            f"- **Total Tokens:** "
            f"{_format_tokens(total_input_tokens + total_output_tokens + total_cached_tokens)} "
            f"(input: {_format_tokens(total_input_tokens)}, "
            f"output: {_format_tokens(total_output_tokens)}, "
            f"cached: {_format_tokens(total_cached_tokens)})",
            f"- **Avg Latency/Case:** {_format_latency(avg_latency)}",
            "",
            "### Status Breakdown",
            "",
        ]

        for status, count in sorted(status_counts.items()):
            lines.append(f"- {_status_badge(status)}: {count}")

        lines.extend([
            "",
            "## Metric Summary",
            "",
            _build_metric_table(metric_summaries),
            "",
            "## Cost & Performance Summary",
            "",
            f"- **Total Estimated Cost:** {_format_cost(total_cost)}",
            f"- **Total Input Tokens:** {_format_tokens(total_input_tokens)}",
            f"- **Total Output Tokens:** {_format_tokens(total_output_tokens)}",
            f"- **Total Cached Tokens:** {_format_tokens(total_cached_tokens)}",
            f"- **Avg Latency per Case:** {_format_latency(avg_latency)}",
            f"- **Min Latency:** {_format_latency(min(latencies) if latencies else None)}",
            f"- **Max Latency:** {_format_latency(max(latencies) if latencies else None)}",
            "",
            "## Per-Case Results",
            "",
        ])

        # Show table for reasonable number of cases
        if len(result.case_results) <= self.max_cases_in_table:
            lines.append(_build_case_table(result.case_results))
        else:
            lines.append(
                f"*{len(result.case_results)} cases — showing summary only. "
                f"Use JSON report for full details.*"
            )

        lines.append(_build_failed_cases_section(result.case_results))
        lines.append(_build_skipped_cases_section(result.case_results))

        lines.extend([
            "",
            "---",
            f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        ])

        return "\n".join(lines)

    def _generate_from_runner_result(self, result: RunnerExperimentResult) -> str:
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

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Status breakdown
        status_counts: dict[str, int] = defaultdict(int)
        for exec_result in result.execution_results:
            status_val = (
                exec_result.status.value
                if hasattr(exec_result.status, "value")
                else exec_result.status
            )
            status_counts[status_val] += 1

        lines = [
            f"# Execution Report: {result.dataset_name}",
            "",
            f"**Dataset ID:** `{result.dataset_id}`  ",
            f"**Run ID:** `{result.run_id}`  ",
            f"**Experiment ID:** `{result.id}`  ",
            f"**Status:** {_status_badge(result.status)}  ",
            f"**Started:** {_format_datetime(result.started_at)}  ",
            f"**Completed:** {_format_datetime(result.completed_at)}  ",
            "",
            "## Summary",
            "",
            f"- **Total Cases:** {result.total_cases}",
            f"- **Successful:** {successful} ({result.success_rate:.1%})",
            f"- **Failed:** {failed}",
            f"- **Timeouts:** {timeout}",
            f"- **Partial:** {partial}",
            f"- **Total Cost:** {_format_cost(total_cost)}",
            f"- **Total Tokens:** "
            f"{_format_tokens(total_input_tokens + total_output_tokens + total_cached_tokens)} "
            f"(input: {_format_tokens(total_input_tokens)}, "
            f"output: {_format_tokens(total_output_tokens)}, "
            f"cached: {_format_tokens(total_cached_tokens)})",
            f"- **Avg Latency/Case:** {_format_latency(avg_latency)}",
            "",
            "### Status Breakdown",
            "",
        ]

        for status, count in sorted(status_counts.items()):
            lines.append(f"- {_status_badge(status)}: {count}")

        lines.extend([
            "",
            "## Cost & Performance Summary",
            "",
            f"- **Total Estimated Cost:** {_format_cost(total_cost)}",
            f"- **Total Input Tokens:** {_format_tokens(total_input_tokens)}",
            f"- **Total Output Tokens:** {_format_tokens(total_output_tokens)}",
            f"- **Total Cached Tokens:** {_format_tokens(total_cached_tokens)}",
            f"- **Avg Latency per Case:** {_format_latency(avg_latency)}",
            f"- **Min Latency:** {_format_latency(min(latencies) if latencies else None)}",
            f"- **Max Latency:** {_format_latency(max(latencies) if latencies else None)}",
            "",
            "## Execution Results",
            "",
        ])

        # Show table for reasonable number of cases
        if len(result.execution_results) <= self.max_cases_in_table:
            lines.append(_build_execution_case_table(result.execution_results))
        else:
            lines.append(
                f"*{len(result.execution_results)} cases — showing summary only. "
                f"Use JSON report for full details.*"
            )

        lines.extend([
            "",
            "---",
            f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        ])

        return "\n".join(lines)

    def generate_comparison(self, comparison: Any) -> str:
        lines = ["# Baseline Comparison Report", ""]
        lines.append(f"**Baseline:** {comparison.baseline.experiment_id}")
        lines.append(f"**Current:** {comparison.current.experiment_id}")
        lines.append("")
        lines.append("## Metric Comparisons")
        lines.append("| Metric | Baseline | Current | Diff | Change | Outcome |")
        lines.append("|---|---|---|---|---|---|")
        for m in comparison.metric_comparisons:
            lines.append(f"| {m.name} | {m.baseline_value} | {m.current_value} | {m.absolute_difference} | {m.direction} | {m.outcome} |")
        lines.append("")
        lines.append(f"Improved cases: {len(comparison.improved_cases)}")
        lines.append(f"Regressed cases: {len(comparison.regressed_cases)}")
        lines.append(f"Added cases: {len(comparison.added_cases)}")
        lines.append(f"Removed cases: {len(comparison.removed_cases)}")
        return chr(10).join(lines)

    def generate_quality_gate(self, result: Any) -> str:
        lines = ["# Quality Gate Report", ""]
        lines.append(f"Overall: {result.status} (passed={result.passed})")
        lines.append("")
        lines.append("| Rule | Status | Actual | Threshold | Explanation |")
        lines.append("|---|---|---|---|---|")
        for c in result.checks:
            lines.append(f"| {c.rule_id} | {c.status} | {c.actual_value} | {c.expected_threshold} | {c.explanation} |")
        return chr(10).join(lines)

    def write_markdown(
        self, result: ExperimentResult | RunnerExperimentResult, filepath: str
    ) -> None:
        """Write a Markdown report to a file.

        Args:
            result: The experiment result to report on.
            filepath: Path to write the Markdown report.
        """
        report = self.generate(result)
        Path(filepath).write_text(report, encoding="utf-8")


__all__ = ["MarkdownReportGenerator"]
