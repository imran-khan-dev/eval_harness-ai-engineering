"""Agent trajectory evaluators for comparing expected vs actual execution paths."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from eval_harness.datasets.models import EvaluationCase, ExpectedTrajectory
from eval_harness.evaluators.base import BaseEvaluator
from eval_harness.metrics.models import MetricScore, MetricType
from eval_harness.tracing.models import ExecutionTrace, TraceEventType


class TrajectoryMatchEvaluator(BaseEvaluator):
    """Trajectory match evaluator.

    Evaluates actual execution trajectory against expected trajectory
    by comparing the sequence of tool calls and steps.
    """

    def __init__(
        self,
        *,
        exact_match: bool = True,
        ignore_arguments: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize the trajectory match evaluator.

        Args:
            exact_match: If True, require exact sequence match.
                If False, use partial sequence matching (LCS-based).
            ignore_arguments: If True, only compare tool names, not arguments.
            case_sensitive: If True, tool name comparison is case-sensitive.
        """
        self._exact_match = exact_match
        self._ignore_arguments = ignore_arguments
        self._case_sensitive = case_sensitive

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TRAJECTORY_SIMILARITY

    @property
    def name(self) -> str:
        return "trajectory_match"

    def _extract_actual_tool_sequence(self, trace: ExecutionTrace) -> list[dict[str, Any]]:
        """Extract tool call sequence from trace events."""
        tool_events = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        sequence = []
        for event in tool_events:
            tool_name = event.data.get("tool", "")
            arguments = event.data.get("arguments", {})
            if self._ignore_arguments:
                sequence.append({"tool": tool_name})
            else:
                sequence.append({"tool": tool_name, "arguments": arguments})
        return sequence

    def _extract_expected_tool_sequence(self, expected: ExpectedTrajectory) -> list[dict[str, Any]]:
        """Extract tool call sequence from expected trajectory steps."""
        sequence = []
        for step in expected.steps:
            if isinstance(step, dict):
                tool_name = step.get("tool") or step.get("name") or ""
                if self._ignore_arguments:
                    sequence.append({"tool": tool_name})
                else:
                    arguments = step.get("arguments", {})
                    sequence.append({"tool": tool_name, "arguments": arguments})
            elif isinstance(step, str):
                # Simple string step - treat as tool name
                sequence.append({"tool": step})
        return sequence

    def _normalize_tool_name(self, name: str) -> str:
        """Normalize tool name based on case sensitivity setting."""
        return name if self._case_sensitive else name.lower()

    def _sequence_to_strings(self, sequence: list[dict[str, Any]]) -> list[str]:
        """Convert sequence to list of tool names for comparison."""
        return [self._normalize_tool_name(step.get("tool", "")) for step in sequence]

    def _compute_lcs_similarity(
        self,
        expected: list[str],
        actual: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Compute similarity using Longest Common Subsequence.

        Returns:
            Tuple of (similarity_score, details_dict).
        """
        matcher = SequenceMatcher(None, expected, actual)
        lcs_length = sum(block.size for block in matcher.get_matching_blocks())

        max_len = max(len(expected), len(actual))
        similarity = lcs_length / max_len if max_len > 0 else 1.0

        # Find matching blocks
        matching_blocks = []
        for block in matcher.get_matching_blocks():
            if block.size > 0:
                matching_blocks.append({
                    "expected_start": block.a,
                    "actual_start": block.b,
                    "length": block.size,
                    "matched": expected[block.a:block.a + block.size],
                })

        return similarity, {
            "lcs_length": lcs_length,
            "expected_length": len(expected),
            "actual_length": len(actual),
            "matching_blocks": matching_blocks,
            "similarity": similarity,
        }

    def _compute_exact_match(
        self,
        expected: list[str],
        actual: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Compute exact sequence match.

        Returns:
            Tuple of (score, details_dict).
        """
        if expected == actual:
            return 1.0, {
                "match": True,
                "expected": expected,
                "actual": actual,
            }

        # Find first mismatch
        first_mismatch = None
        for i, (exp, act) in enumerate(zip(expected, actual, strict=False)):
            if exp != act:
                first_mismatch = {"index": i, "expected": exp, "actual": act}
                break

        # Check for length mismatch
        if first_mismatch is None and len(expected) != len(actual):
            first_mismatch = {
                "index": min(len(expected), len(actual)),
                "expected": expected[len(actual):] if len(expected) > len(actual) else [],
                "actual": actual[len(expected):] if len(actual) > len(expected) else [],
            }

        return 0.0, {
            "match": False,
            "expected": expected,
            "actual": actual,
            "first_mismatch": first_mismatch,
        }

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        trace: ExecutionTrace | None = None,
        expected_trajectory: ExpectedTrajectory | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate trajectory match.

        Args:
            case: Evaluation case with expected_trajectory.
            prediction: Not used (uses trace and expected_trajectory).
            trace: ExecutionTrace containing actual tool call events.
            expected_trajectory: ExpectedTrajectory to compare against.
                If not provided, uses case.expected_trajectory.
            **kwargs: Ignored.

        Returns:
            MetricScore with similarity score and details.
        """
        exp_traj = expected_trajectory or case.expected_trajectory

        if trace is None:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "no_trace_provided",
                    "expected_steps": 0,
                    "actual_steps": 0,
                },
                metadata={"case_id": str(case.id)},
            )

        if exp_traj is None:
            # No expected trajectory - check if any tools were called
            actual_seq = self._extract_actual_tool_sequence(trace)
            if actual_seq:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "unexpected_tool_calls",
                        "expected_steps": 0,
                        "actual_steps": len(actual_seq),
                        "actual_tools": [s["tool"] for s in actual_seq],
                    },
                    metadata={"case_id": str(case.id)},
                )
            return self._create_score(
                value=1.0,
                details={
                    "expected_steps": 0,
                    "actual_steps": 0,
                    "match": True,
                },
                metadata={"case_id": str(case.id)},
            )

        expected_seq = self._extract_expected_tool_sequence(exp_traj)
        actual_seq = self._extract_actual_tool_sequence(trace)

        expected_tools = self._sequence_to_strings(expected_seq)
        actual_tools = self._sequence_to_strings(actual_seq)

        # Handle empty expected and empty actual
        if not expected_tools and not actual_tools:
            return self._create_score(
                value=1.0,
                details={
                    "expected_steps": 0,
                    "actual_steps": 0,
                    "match": True,
                },
                metadata={"case_id": str(case.id)},
            )

        if self._exact_match:
            score, details = self._compute_exact_match(expected_tools, actual_tools)
        else:
            score, details = self._compute_lcs_similarity(expected_tools, actual_tools)

        return self._create_score(
            value=score,
            details={
                **details,
                "expected_steps": len(expected_tools),
                "actual_steps": len(actual_tools),
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
            },
            metadata={"case_id": str(case.id)},
        )


class TrajectoryOrderEvaluator(BaseEvaluator):
    """Trajectory order evaluator.

    Evaluates whether expected tool calls occurred in the correct relative order.
    Does not require exact sequence - only checks relative ordering constraints.
    """

    def __init__(
        self,
        *,
        require_all_expected: bool = True,
        ignore_arguments: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize the trajectory order evaluator.

        Args:
            require_all_expected: If True, all expected tools must appear.
                If False, only checks order of tools that do appear.
            ignore_arguments: If True, only compare tool names.
            case_sensitive: If True, tool name comparison is case-sensitive.
        """
        self._require_all_expected = require_all_expected
        self._ignore_arguments = ignore_arguments
        self._case_sensitive = case_sensitive

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TRAJECTORY_SIMILARITY

    @property
    def name(self) -> str:
        return "trajectory_order"

    def _extract_actual_tool_sequence(self, trace: ExecutionTrace) -> list[str]:
        """Extract ordered tool names from trace events."""
        tool_events = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        tools = []
        for event in tool_events:
            tool_name = event.data.get("tool", "")
            tools.append(tool_name if self._case_sensitive else tool_name.lower())
        return tools

    def _extract_expected_tool_sequence(self, expected: ExpectedTrajectory) -> list[str]:
        """Extract expected tool names from trajectory steps."""
        tools = []
        for step in expected.steps:
            if isinstance(step, dict):
                tool_name = step.get("tool") or step.get("name") or ""
                tools.append(tool_name if self._case_sensitive else tool_name.lower())
            elif isinstance(step, str):
                tools.append(step if self._case_sensitive else step.lower())
        return tools

    def _check_order(
        self,
        expected: list[str],
        actual: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Check if expected tools appear in correct relative order in actual sequence.

        Returns:
            Tuple of (score, details_dict).
        """
        if not expected:
            return 1.0, {
                "expected_tools": [],
                "actual_tools": actual,
                "all_present": True,
                "order_correct": True,
            }

        # Find positions of expected tools in actual sequence
        positions = {}
        for i, tool in enumerate(actual):
            if tool in expected and tool not in positions:
                positions[tool] = i

        # Check which expected tools are present
        missing = [tool for tool in expected if tool not in positions]
        present = [tool for tool in expected if tool in positions]

        all_present = len(missing) == 0

        if self._require_all_expected and not all_present:
            # Missing required tools - order check fails
            return 0.0, {
                "expected_tools": expected,
                "actual_tools": actual,
                "present_tools": present,
                "missing_tools": missing,
                "all_present": False,
                "order_correct": False,
            }

        # Check relative order of present tools
        order_correct = True
        order_violations = []
        for i in range(len(present) - 1):
            tool_a = present[i]
            tool_b = present[i + 1]
            if positions[tool_a] > positions[tool_b]:
                order_correct = False
                order_violations.append({
                    "expected_before": tool_a,
                    "expected_after": tool_b,
                    "actual_position_a": positions[tool_a],
                    "actual_position_b": positions[tool_b],
                })

        # Score: 1.0 if all present and correct order, otherwise proportional
        if self._require_all_expected:
            if all_present and order_correct:
                score = 1.0
            elif not all_present:
                score = 0.0
            else:
                # All present but order wrong
                score = 0.5
        else:
            # Only score based on present tools
            if len(present) <= 1:
                score = 1.0
            else:
                correct_pairs = sum(1 for i in range(len(present) - 1)
                                   if positions[present[i]] < positions[present[i + 1]])
                total_pairs = len(present) - 1
                score = correct_pairs / total_pairs

        return score, {
            "expected_tools": expected,
            "actual_tools": actual,
            "present_tools": present,
            "missing_tools": missing,
            "all_present": all_present,
            "order_correct": order_correct,
            "order_violations": order_violations,
            "positions": {tool: positions.get(tool) for tool in expected},
        }

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        trace: ExecutionTrace | None = None,
        expected_trajectory: ExpectedTrajectory | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate trajectory order.

        Args:
            case: Evaluation case with expected_trajectory.
            prediction: Not used.
            trace: ExecutionTrace containing actual tool call events.
            expected_trajectory: ExpectedTrajectory to compare against.
            **kwargs: Ignored.

        Returns:
            MetricScore with order correctness score and details.
        """
        exp_traj = expected_trajectory or case.expected_trajectory

        if trace is None:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "no_trace_provided",
                    "expected_tools": [],
                    "actual_tools": [],
                    "all_present": False,
                    "order_correct": False,
                },
                metadata={"case_id": str(case.id)},
            )

        if exp_traj is None:
            actual_tools = self._extract_actual_tool_sequence(trace)
            if actual_tools:
                return self._create_score(
                    value=0.0,
                    details={
                        "reason": "no_expected_trajectory",
                        "expected_tools": [],
                        "actual_tools": actual_tools,
                        "all_present": True,
                        "order_correct": True,
                    },
                    metadata={"case_id": str(case.id)},
                )
            return self._create_score(
                value=1.0,
                details={
                    "expected_tools": [],
                    "actual_tools": [],
                    "all_present": True,
                    "order_correct": True,
                },
                metadata={"case_id": str(case.id)},
            )

        expected_tools = self._extract_expected_tool_sequence(exp_traj)
        actual_tools = self._extract_actual_tool_sequence(trace)

        score, details = self._check_order(expected_tools, actual_tools)

        return self._create_score(
            value=score,
            details=details,
            metadata={"case_id": str(case.id)},
        )


class TrajectoryCompletenessEvaluator(BaseEvaluator):
    """Trajectory completeness evaluator.

    Evaluates whether an agent completed the expected execution by checking
    if all expected steps were executed and if execution terminated successfully.
    """

    def __init__(
        self,
        *,
        require_successful_termination: bool = True,
        ignore_arguments: bool = True,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize the trajectory completeness evaluator.

        Args:
            require_successful_termination: If True, execution must end successfully
                (status="success" in trace) to get full score.
            ignore_arguments: If True, only compare tool names.
            case_sensitive: If True, tool name comparison is case-sensitive.
        """
        self._require_successful_termination = require_successful_termination
        self._ignore_arguments = ignore_arguments
        self._case_sensitive = case_sensitive

    @property
    def metric_type(self) -> MetricType:
        return MetricType.TRAJECTORY_SIMILARITY

    @property
    def name(self) -> str:
        return "trajectory_completeness"

    def _extract_actual_tool_set(self, trace: ExecutionTrace) -> set[str]:
        """Extract set of unique tool names from trace events."""
        tool_events = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        tools = set()
        for event in tool_events:
            tool_name = event.data.get("tool", "")
            tools.add(tool_name if self._case_sensitive else tool_name.lower())
        return tools

    def _extract_expected_tool_set(self, expected: ExpectedTrajectory) -> set[str]:
        """Extract set of unique expected tool names from trajectory steps."""
        tools = set()
        for step in expected.steps:
            if isinstance(step, dict):
                tool_name = step.get("tool") or step.get("name") or ""
                tools.add(tool_name if self._case_sensitive else tool_name.lower())
            elif isinstance(step, str):
                tools.add(step if self._case_sensitive else step.lower())
        return tools

    def _is_execution_successful(self, trace: ExecutionTrace) -> bool:
        """Check if execution terminated successfully."""
        return trace.status == "success"

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        trace: ExecutionTrace | None = None,
        expected_trajectory: ExpectedTrajectory | None = None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate trajectory completeness.

        Args:
            case: Evaluation case with expected_trajectory.
            prediction: Not used.
            trace: ExecutionTrace containing actual tool call events.
            expected_trajectory: ExpectedTrajectory to compare against.
            **kwargs: Ignored.

        Returns:
            MetricScore with completeness score and details.
        """
        exp_traj = expected_trajectory or case.expected_trajectory

        if trace is None:
            return self._create_score(
                value=0.0,
                details={
                    "reason": "no_trace_provided",
                    "expected_tools": [],
                    "actual_tools": [],
                    "completed_tools": [],
                    "missing_tools": [],
                    "execution_successful": False,
                },
                metadata={"case_id": str(case.id)},
            )

        if exp_traj is None:
            actual_tools = self._extract_actual_tool_set(trace)
            execution_successful = self._is_execution_successful(trace)
            if self._require_successful_termination and not execution_successful:
                return self._create_score(
                    value=0.5,
                    details={
                        "reason": "no_expected_trajectory_but_execution_failed",
                        "expected_tools": [],
                        "actual_tools": list(actual_tools),
                        "completed_tools": list(actual_tools),
                        "missing_tools": [],
                        "execution_successful": execution_successful,
                    },
                    metadata={"case_id": str(case.id)},
                )
            return self._create_score(
                value=1.0,
                details={
                    "expected_tools": [],
                    "actual_tools": list(actual_tools),
                    "completed_tools": list(actual_tools),
                    "missing_tools": [],
                    "execution_successful": execution_successful,
                },
                metadata={"case_id": str(case.id)},
            )

        expected_tools = self._extract_expected_tool_set(exp_traj)
        actual_tools = self._extract_actual_tool_set(trace)

        # Compute completeness
        completed_tools = expected_tools & actual_tools
        missing_tools = expected_tools - actual_tools
        extra_tools = actual_tools - expected_tools

        total_expected = len(expected_tools)
        completed_count = len(completed_tools)

        completeness = 1.0 if total_expected == 0 else completed_count / total_expected

        # Apply termination penalty if required
        execution_successful = self._is_execution_successful(trace)
        if self._require_successful_termination and not execution_successful:
            completeness *= 0.5

        return self._create_score(
            value=completeness,
            details={
                "expected_tools": list(expected_tools),
                "actual_tools": list(actual_tools),
                "completed_tools": list(completed_tools),
                "missing_tools": list(missing_tools),
                "extra_tools": list(extra_tools),
                "expected_count": total_expected,
                "completed_count": completed_count,
                "execution_successful": execution_successful,
                "trace_status": trace.status,
            },
            metadata={"case_id": str(case.id)},
        )
