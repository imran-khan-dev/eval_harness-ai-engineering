"""Tool evaluation metrics for tool selection, arguments, and execution."""

from __future__ import annotations

from collections import Counter
from typing import Any

from eval_harness.datasets.models import EvaluationCase, ExpectedToolCall
from eval_harness.evaluators.base import BaseEvaluator
from eval_harness.metrics.models import MetricScore, MetricType
from eval_harness.tracing.models import ExecutionTrace, TraceEvent, TraceEventType


class ToolSelectionEvaluator(BaseEvaluator):
    """Tool selection evaluator.

    Evaluates whether the correct tools were selected by comparing
    expected tool calls against actual tool calls from trace events.

    Computes precision, recall, and F1 score based on tool name matching.
    """

    def __init__(
        self,
        *,
        count_duplicates: bool = False,
        required_only: bool = False,
    ) -> None:
        """Initialize the tool selection evaluator.

        Args:
            count_duplicates: If True, count duplicate tool calls separately.
                If False (default), treat tool calls as a multiset where
                duplicates are counted up to the expected count.
            required_only: If True, only evaluate required tool calls
                (those with required=True in ExpectedToolCall).
        """
        self._count_duplicates = count_duplicates
        self._required_only = required_only

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TOOL_SELECTION_ACCURACY

    @property
    def name(self) -> str:
        return "tool_selection_accuracy"

    def _extract_tool_names(
        self,
        tool_calls: list[ExpectedToolCall],
    ) -> list[str]:
        """Extract tool names from expected tool calls."""
        if self._required_only:
            return [tc.name for tc in tool_calls if tc.required]
        return [tc.name for tc in tool_calls]

    def _extract_tool_names_from_trace(self, trace: ExecutionTrace) -> list[str]:
        """Extract tool names from actual tool call events in trace."""
        tool_events = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        return [event.data.get("tool", "") for event in tool_events if event.data.get("tool")]

    def _compute_metrics(
        self,
        expected: list[str],
        actual: list[str],
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Compute precision, recall, F1 and detailed breakdown.

        Args:
            expected: List of expected tool names.
            actual: List of actual tool names from trace.

        Returns:
            Tuple of (precision, recall, f1, details_dict).
        """
        if self._count_duplicates:
            # Use Counter for multiset comparison
            expected_counter = Counter(expected)
            actual_counter = Counter(actual)

            # True positives: min of expected and actual for each tool
            true_positives = sum(
                min(expected_counter[tool], actual_counter.get(tool, 0))
                for tool in expected_counter
            )
            # False positives: actual tools not in expected (or excess duplicates)
            false_positives = sum(
                max(0, actual_counter[tool] - expected_counter.get(tool, 0))
                for tool in actual_counter
            )
            # False negatives: expected tools not in actual (or missing duplicates)
            false_negatives = sum(
                max(0, expected_counter[tool] - actual_counter.get(tool, 0))
                for tool in expected_counter
            )

            details = {
                "expected_counts": dict(expected_counter),
                "actual_counts": dict(actual_counter),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        else:
            # Set-based comparison (each tool counted once)
            expected_set = set(expected)
            actual_set = set(actual)

            true_positives = len(expected_set & actual_set)
            false_positives = len(actual_set - expected_set)
            false_negatives = len(expected_set - actual_set)

            details = {
                "expected_tools": list(expected_set),
                "actual_tools": list(actual_set),
                "matched_tools": list(expected_set & actual_set),
                "missing_tools": list(expected_set - actual_set),
                "unexpected_tools": list(actual_set - expected_set),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 1.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 1.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 1.0
        )

        return precision, recall, f1, details

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        trace: ExecutionTrace | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate tool selection against expected tool calls.

        Args:
            case: Evaluation case with expected_tool_calls.
            prediction: Not used for this evaluator (uses trace instead).
            trace: ExecutionTrace containing actual tool call events.
            **kwargs: Ignored.

        Returns:
            MetricScore with F1 as value, and precision/recall in details.
        """
        expected_tools = self._extract_tool_names(case.expected_tool_calls)

        if trace is None:
            return self._create_score(
                value=0.0,
                threshold=0.5,
                details={
                    "reason": "no_trace_provided",
                    "expected_tools": expected_tools,
                    "actual_tools": [],
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
                metadata={"case_id": str(case.id)},
            )

        actual_tools = self._extract_tool_names_from_trace(trace)

        # Handle empty expected and empty actual
        if not expected_tools and not actual_tools:
            return self._create_score(
                value=1.0,
                threshold=0.5,
                details={
                    "expected_tools": [],
                    "actual_tools": [],
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                },
                metadata={"case_id": str(case.id)},
            )

        precision, recall, f1, details = self._compute_metrics(expected_tools, actual_tools)

        return self._create_score(
            value=f1,
            threshold=0.5,
            details={
                **details,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            metadata={"case_id": str(case.id)},
        )


class ToolArgumentEvaluator(BaseEvaluator):
    """Tool argument evaluator.

    Evaluates whether tool call arguments match expected arguments.
    Supports exact matching and nested dictionary comparison.
    """

    def __init__(
        self,
        *,
        exact_match: bool = True,
        required_only: bool = False,
    ) -> None:
        """Initialize the tool argument evaluator.

        Args:
            exact_match: If True (default), require exact argument match.
                If False, check that expected args are subset of actual args.
            required_only: If True, only evaluate required tool calls.
        """
        self._exact_match = exact_match
        self._required_only = required_only

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TOOL_CALL_CORRECTNESS

    @property
    def name(self) -> str:
        return "tool_argument_correctness"

    def _extract_expected_calls(
        self,
        expected_calls: list[ExpectedToolCall],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group expected tool calls by name, preserving order for duplicate handling."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for call in expected_calls:
            if self._required_only and not call.required:
                continue
            grouped.setdefault(call.name, []).append(call.arguments)
        return grouped

    def _extract_actual_calls(self, trace: ExecutionTrace) -> dict[str, list[dict[str, Any]]]:
        """Group actual tool calls by name from trace events."""
        tool_events = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in tool_events:
            tool_name = event.data.get("tool")
            if tool_name:
                args = event.data.get("arguments", {})
                grouped.setdefault(tool_name, []).append(args)
        return grouped

    def _compare_arguments(
        self,
        expected_args: dict[str, Any],
        actual_args: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Compare expected and actual arguments.

        Returns:
            Tuple of (match, details_dict).
        """
        if self._exact_match:
            match = expected_args == actual_args
            details = {
                "match": match,
                "expected": expected_args,
                "actual": actual_args,
                "missing_keys": list(set(expected_args.keys()) - set(actual_args.keys())),
                "extra_keys": list(set(actual_args.keys()) - set(expected_args.keys())),
                "mismatched_keys": [
                    k for k in expected_args
                    if k in actual_args and expected_args[k] != actual_args[k]
                ],
            }
        else:
            # Subset match: expected args must be present in actual with same values
            missing_keys = []
            mismatched_keys = []
            for key, expected_val in expected_args.items():
                if key not in actual_args:
                    missing_keys.append(key)
                elif actual_args[key] != expected_val:
                    mismatched_keys.append(key)

            match = len(missing_keys) == 0 and len(mismatched_keys) == 0
            details = {
                "match": match,
                "expected": expected_args,
                "actual": actual_args,
                "missing_keys": missing_keys,
                "extra_keys": list(set(actual_args.keys()) - set(expected_args.keys())),
                "mismatched_keys": mismatched_keys,
            }

        return match, details

    def _deep_compare(self, expected: Any, actual: Any) -> bool:
        """Recursively compare nested structures."""
        if type(expected) is not type(actual):
            return False
        if isinstance(expected, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False
            return all(self._deep_compare(expected[k], actual[k]) for k in expected)
        if isinstance(expected, list):
            if len(expected) != len(actual):
                return False
            return all(self._deep_compare(e, a) for e, a in zip(expected, actual, strict=False))
        return bool(expected == actual)

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        trace: ExecutionTrace | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate tool arguments against expected arguments.

        Args:
            case: Evaluation case with expected_tool_calls containing arguments.
            prediction: Not used for this evaluator.
            trace: ExecutionTrace containing actual tool call events.
            **kwargs: Ignored.

        Returns:
            MetricScore with ratio of matching argument sets as value.
        """
        if trace is None:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "no_trace_provided",
                    "per_tool_results": {},
                    "matched_calls": 0,
                    "total_expected_calls": 0,
                },
                metadata={"case_id": str(case.id)},
            )

        expected_grouped = self._extract_expected_calls(case.expected_tool_calls)
        actual_grouped = self._extract_actual_calls(trace)

        total_expected_calls = sum(len(calls) for calls in expected_grouped.values())

        if total_expected_calls == 0:
            # No expected tool calls - check if any actual calls were made
            if actual_grouped:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "unexpected_tool_calls",
                        "per_tool_results": {
                            tool: [
                                {"match": False, "expected": None, "actual": args}
                                for args in args_list
                            ]
                            for tool, args_list in actual_grouped.items()
                        },
                        "matched_calls": 0,
                        "total_expected_calls": 0,
                        "total_actual_calls": sum(len(calls) for calls in actual_grouped.values()),
                    },
                    metadata={"case_id": str(case.id)},
                )
            return self._create_score(
                value=1.0,
                details={
                    "per_tool_results": {},
                    "matched_calls": 0,
                    "total_expected_calls": 0,
                    "total_actual_calls": 0,
                },
                metadata={"case_id": str(case.id)},
            )

        per_tool_results = {}
        matched_calls = 0

        for tool_name, expected_calls_list in expected_grouped.items():
            actual_calls_list = actual_grouped.get(tool_name, [])
            tool_results = []

            for i, expected_args in enumerate(expected_calls_list):
                if i < len(actual_calls_list):
                    actual_args = actual_calls_list[i]
                    match, details = self._compare_arguments(expected_args, actual_args)
                    if match:
                        matched_calls += 1
                else:
                    match = False
                    details = {
                        "match": False,
                        "expected": expected_args,
                        "actual": None,
                        "missing_keys": list(expected_args.keys()),
                        "extra_keys": [],
                        "mismatched_keys": [],
                    }
                tool_results.append(details)

            # Check for extra actual calls beyond expected
            for i in range(len(expected_calls_list), len(actual_calls_list)):
                actual_args = actual_calls_list[i]
                tool_results.append({
                    "match": False,
                    "expected": None,
                    "actual": actual_args,
                    "missing_keys": [],
                    "extra_keys": list(actual_args.keys()),
                    "mismatched_keys": [],
                })

            per_tool_results[tool_name] = tool_results

        score = matched_calls / total_expected_calls if total_expected_calls > 0 else 1.0

        return self._create_score(
            value=score,
            details={
                "per_tool_results": per_tool_results,
                "matched_calls": matched_calls,
                "total_expected_calls": total_expected_calls,
                "total_actual_calls": sum(len(calls) for calls in actual_grouped.values()),
            },
            metadata={"case_id": str(case.id)},
        )


class ToolExecutionEvaluator(BaseEvaluator):
    """Tool execution evaluator.

    Evaluates whether expected tool calls were executed and whether
    they succeeded or failed based on trace events.
    """

    def __init__(
        self,
        *,
        required_only: bool = False,
        check_arguments: bool = False,
    ) -> None:
        """Initialize the tool execution evaluator.

        Args:
            required_only: If True, only evaluate required tool calls.
            check_arguments: If True, also verify arguments match (subset check).
        """
        self._required_only = required_only
        self._check_arguments = check_arguments

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TOOL_CALL_CORRECTNESS

    @property
    def name(self) -> str:
        return "tool_execution_correctness"

    def _extract_expected_calls(
        self,
        expected_calls: list[ExpectedToolCall],
    ) -> list[ExpectedToolCall]:
        """Filter expected tool calls based on required_only."""
        if self._required_only:
            return [tc for tc in expected_calls if tc.required]
        return expected_calls

    def _get_tool_call_response_pairs(
        self, trace: ExecutionTrace
    ) -> list[tuple[TraceEvent, TraceEvent | None]]:
        """Pair tool calls with their responses by correlation_id or order."""
        tool_calls = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        tool_responses = trace.get_events_by_type(TraceEventType.TOOL_RESPONSE)
        tool_errors = trace.get_events_by_type(TraceEventType.ERROR)

        # Build response lookup by correlation_id
        response_by_corr: dict[str, list[TraceEvent]] = {}
        for resp in tool_responses:
            if resp.correlation_id:
                response_by_corr.setdefault(str(resp.correlation_id), []).append(resp)

        # Build error lookup by correlation_id
        error_by_corr: dict[str, list[TraceEvent]] = {}
        for err in tool_errors:
            if err.correlation_id:
                error_by_corr.setdefault(str(err.correlation_id), []).append(err)

        pairs: list[tuple[TraceEvent, TraceEvent | None]] = []
        for call in tool_calls:
            corr_id = str(call.correlation_id) if call.correlation_id else None
            if corr_id and corr_id in response_by_corr:
                # Use first matching response
                resp = response_by_corr[corr_id].pop(0)
                pairs.append((call, resp))
            elif corr_id and corr_id in error_by_corr:
                # Use first matching error as response
                resp = error_by_corr[corr_id].pop(0)
                pairs.append((call, resp))
            else:
                pairs.append((call, None))

        return pairs

    def _is_success(self, response: TraceEvent | None) -> bool:
        """Determine if a tool response indicates success."""
        if response is None:
            return False
        if response.is_error_event():
            return False
        # Check if response data contains error indicator
        result = response.data.get("result")
        return not (isinstance(result, dict) and result.get("error"))

    def _arguments_match(self, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
        """Check if actual arguments contain all expected arguments with matching values."""
        for key, expected_val in expected.items():
            if key not in actual:
                return False
            if not self._deep_compare(expected_val, actual[key]):
                return False
        return True

    def _deep_compare(self, expected: Any, actual: Any) -> bool:
        """Recursively compare nested structures."""
        if type(expected) is not type(actual):
            return False
        if isinstance(expected, dict):
            for k, v in expected.items():
                if k not in actual:
                    return False
                if not self._deep_compare(v, actual[k]):
                    return False
            return True
        if isinstance(expected, list):
            if len(expected) != len(actual):
                return False
            return all(self._deep_compare(e, a) for e, a in zip(expected, actual, strict=False))
        return bool(expected == actual)

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        trace: ExecutionTrace | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate tool execution status against expected tool calls.

        Args:
            case: Evaluation case with expected_tool_calls.
            prediction: Not used for this evaluator.
            trace: ExecutionTrace containing actual tool call and response events.
            **kwargs: Ignored.

        Returns:
            MetricScore with ratio of successfully executed expected calls as value.
        """
        expected_calls = self._extract_expected_calls(case.expected_tool_calls)

        if trace is None:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "no_trace_provided",
                    "per_tool_results": [],
                    "executed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "missing": len(expected_calls),
                    "total_expected": len(expected_calls),
                },
                metadata={"case_id": str(case.id)},
            )

        if not expected_calls:
            # No expected calls - check if any tools were called
            tool_calls = trace.get_events_by_type(TraceEventType.TOOL_CALL)
            if tool_calls:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "unexpected_tool_calls",
                        "per_tool_results": [
                            {
                                "tool": call.data.get("tool"),
                                "executed": True,
                                "succeeded": False,
                                "expected": False,
                            }
                            for call in tool_calls
                        ],
                        "executed": len(tool_calls),
                        "succeeded": 0,
                        "failed": 0,
                        "missing": 0,
                        "total_expected": 0,
                    },
                    metadata={"case_id": str(case.id)},
                )
            return self._create_score(
                value=1.0,
                details={
                    "per_tool_results": [],
                    "executed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "missing": 0,
                    "total_expected": 0,
                },
                metadata={"case_id": str(case.id)},
            )

        # Get actual tool call/response pairs
        call_response_pairs = self._get_tool_call_response_pairs(trace)

        # Group by tool name for matching
        actual_by_tool: dict[str, list[tuple[TraceEvent, TraceEvent | None]]] = {}
        for call, resp in call_response_pairs:
            tool_name = call.data.get("tool", "")
            if tool_name:
                actual_by_tool.setdefault(tool_name, []).append((call, resp))

        per_tool_results = []
        executed_count = 0
        succeeded_count = 0
        missing_count = 0

        for expected_call in expected_calls:
            tool_name = expected_call.name
            actual_pairs = actual_by_tool.get(tool_name, [])

            if actual_pairs:
                # Use first unmatched actual call for this expected call
                call, resp = actual_pairs.pop(0)
                executed_count += 1

                success = self._is_success(resp)

                args_match = True
                if self._check_arguments:
                    actual_args = call.data.get("arguments", {})
                    args_match = self._arguments_match(expected_call.arguments, actual_args)
                    # If arguments don't match, consider execution as failed
                    if not args_match:
                        success = False

                if success:
                    succeeded_count += 1

                per_tool_results.append({
                    "tool": tool_name,
                    "executed": True,
                    "succeeded": success,
                    "arguments_match": args_match,
                    "expected_arguments": expected_call.arguments,
                    "actual_arguments": call.data.get("arguments", {}),
                    "response": resp.data if resp else None,
                    "error": resp.error if resp and resp.is_error_event() else None,
                })
            else:
                missing_count += 1
                per_tool_results.append({
                    "tool": tool_name,
                    "executed": False,
                    "succeeded": False,
                    "arguments_match": False,
                    "expected_arguments": expected_call.arguments,
                    "actual_arguments": None,
                    "response": None,
                    "error": "Tool call not found in trace",
                })

        # Check for extra actual calls not expected
        for tool_name, pairs in actual_by_tool.items():
            for call, resp in pairs:
                executed_count += 1
                success = self._is_success(resp)
                if success:
                    succeeded_count += 1
                per_tool_results.append({
                    "tool": tool_name,
                    "executed": True,
                    "succeeded": success,
                    "arguments_match": False,
                    "expected_arguments": None,
                    "actual_arguments": call.data.get("arguments", {}),
                    "response": resp.data if resp else None,
                    "error": "Unexpected tool call",
                })

        total_expected = len(expected_calls)
        score = succeeded_count / total_expected if total_expected > 0 else 1.0

        return self._create_score(
            value=score,
            details={
                "per_tool_results": per_tool_results,
                "executed": executed_count,
                "succeeded": succeeded_count,
                "failed": executed_count - succeeded_count,
                "missing": missing_count,
                "total_expected": total_expected,
            },
            metadata={"case_id": str(case.id)},
        )
