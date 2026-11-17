"""Tests for agent trajectory evaluation metrics."""


import pytest

from eval_harness.datasets.models import EvaluationCase, ExpectedTrajectory
from eval_harness.evaluators import (
    TrajectoryCompletenessEvaluator,
    TrajectoryMatchEvaluator,
    TrajectoryOrderEvaluator,
)
from eval_harness.metrics.models import MetricType
from eval_harness.tracing.models import (
    ExecutionTrace,
    TraceEvent,
)


class TestTrajectoryMatchEvaluator:
    """Tests for TrajectoryMatchEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = TrajectoryMatchEvaluator()
        self.case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search", "arguments": {"query": "test"}},
                {"tool": "fetch", "arguments": {"url": "http://example.com"}},
                {"tool": "summarize", "arguments": {"text": "content"}},
            ]),
        )

    def _make_trace(self, tool_names: list[str]) -> ExecutionTrace:
        """Create a trace with tool call events in order."""
        trace = ExecutionTrace(case_id=self.case.id)
        for name in tool_names:
            trace.add_event(TraceEvent.tool_call(
                execution_id=trace.id,
                tool_name=name,
                arguments={},
            ))
        trace.finalize(status="success")
        return trace

    def test_exact_match(self) -> None:
        """Test exact trajectory match returns 1.0."""
        trace = self._make_trace(["search", "fetch", "summarize"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["match"] is True

    def test_partial_match_first_mismatch(self) -> None:
        """Test mismatch at first step."""
        trace = self._make_trace(["wrong", "fetch", "summarize"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["match"] is False
        assert score.details["first_mismatch"]["index"] == 0

    def test_missing_step(self) -> None:
        """Test missing expected step."""
        trace = self._make_trace(["search", "fetch"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["match"] is False

    def test_extra_step(self) -> None:
        """Test extra unexpected step."""
        trace = self._make_trace(["search", "fetch", "summarize", "extra"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["match"] is False

    def test_no_trace(self) -> None:
        """Test no trace provided."""
        score = self.evaluator.evaluate(self.case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_trace_provided"

    def test_no_expected_trajectory_with_actual(self) -> None:
        """Test unexpected tools when no expected trajectory."""
        case = EvaluationCase(input="test")
        trace = self._make_trace(["unexpected"])
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["reason"] == "unexpected_tool_calls"

    def test_no_expected_no_actual(self) -> None:
        """Test no expected and no actual trajectory."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["match"] is True

    def test_partial_match_mode(self) -> None:
        """Test LCS-based partial matching."""
        evaluator = TrajectoryMatchEvaluator(exact_match=False)
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search"},
                {"tool": "fetch"},
                {"tool": "summarize"},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="summarize", arguments={}
            )
        )
        trace.finalize(status="success")
        score = evaluator.evaluate(case, None, trace=trace)
        # 2 out of 3 matched in order -> 2/3 similarity
        assert score.value == pytest.approx(2/3)

    def test_ignore_arguments(self) -> None:
        """Test argument comparison can be ignored."""
        evaluator = TrajectoryMatchEvaluator(ignore_arguments=True)
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search", "arguments": {"query": "expected"}},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id,
            tool_name="search",
            arguments={"query": "different"},
        ))
        trace.finalize(status="success")
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        evaluator = TrajectoryMatchEvaluator(case_sensitive=False)
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "Search"},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.finalize(status="success")
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_metric_type_and_name(self) -> None:
        """Test metric type and name."""
        assert self.evaluator.metric_type == MetricType.TRAJECTORY_SIMILARITY
        assert self.evaluator.name == "trajectory_match"


class TestTrajectoryOrderEvaluator:
    """Tests for TrajectoryOrderEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = TrajectoryOrderEvaluator()
        self.case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search"},
                {"tool": "fetch"},
                {"tool": "summarize"},
            ]),
        )

    def _make_trace(self, tool_names: list[str]) -> ExecutionTrace:
        """Create a trace with tool call events in order."""
        trace = ExecutionTrace(case_id=self.case.id)
        for name in tool_names:
            trace.add_event(TraceEvent.tool_call(
                execution_id=trace.id,
                tool_name=name,
                arguments={},
            ))
        trace.finalize(status="success")
        return trace

    def test_correct_order(self) -> None:
        """Test correct relative order returns 1.0."""
        trace = self._make_trace(["search", "fetch", "summarize"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["order_correct"] is True
        assert score.details["all_present"] is True

    def test_incorrect_order(self) -> None:
        """Test incorrect order returns 0.5 (all present but wrong order)."""
        trace = self._make_trace(["fetch", "search", "summarize"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.5
        assert score.details["order_correct"] is False
        assert len(score.details["order_violations"]) > 0

    def test_missing_expected_tool(self) -> None:
        """Test missing expected tool returns 0.0 when require_all_expected=True."""
        trace = self._make_trace(["search", "summarize"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["all_present"] is False
        assert "fetch" in score.details["missing_tools"]

    def test_require_all_expected_false(self) -> None:
        """Test partial order scoring when not all required."""
        evaluator = TrajectoryOrderEvaluator(require_all_expected=False)
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search"},
                {"tool": "fetch"},
                {"tool": "summarize"},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="summarize", arguments={}
            )
        )
        trace.finalize(status="success")
        score = evaluator.evaluate(case, None, trace=trace)
        # Only search and summarize present, in correct relative order
        assert score.value == 1.0

    def test_no_trace(self) -> None:
        """Test no trace provided."""
        score = self.evaluator.evaluate(self.case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_trace_provided"

    def test_no_expected_trajectory(self) -> None:
        """Test no expected trajectory with actual tools."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.finalize(status="success")
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["reason"] == "no_expected_trajectory"

    def test_empty_expected_and_actual(self) -> None:
        """Test empty expected and actual."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["all_present"] is True

    def test_duplicate_tool_calls(self) -> None:
        """Test duplicate tool calls - only first occurrence matters."""
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search"},
                {"tool": "fetch"},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="fetch", arguments={}
            )
        )
        trace.finalize(status="success")
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_case_insensitive(self) -> None:
        """Test case insensitive ordering."""
        evaluator = TrajectoryOrderEvaluator(case_sensitive=False)
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "Search"},
                {"tool": "Fetch"},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="fetch", arguments={}
            )
        )
        trace.finalize(status="success")
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_metric_type_and_name(self) -> None:
        """Test metric type and name."""
        assert self.evaluator.metric_type == MetricType.TRAJECTORY_SIMILARITY
        assert self.evaluator.name == "trajectory_order"


class TestTrajectoryCompletenessEvaluator:
    """Tests for TrajectoryCompletenessEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = TrajectoryCompletenessEvaluator()
        self.case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search"},
                {"tool": "fetch"},
                {"tool": "summarize"},
            ]),
        )

    def _make_trace(self, tool_names: list[str], status: str = "success") -> ExecutionTrace:
        """Create a trace with tool call events."""
        trace = ExecutionTrace(case_id=self.case.id)
        for name in tool_names:
            trace.add_event(TraceEvent.tool_call(
                execution_id=trace.id,
                tool_name=name,
                arguments={},
            ))
        trace.finalize(status=status)
        return trace

    def test_all_completed_successful(self) -> None:
        """Test all expected tools completed with successful termination."""
        trace = self._make_trace(["search", "fetch", "summarize"], status="success")
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["completed_count"] == 3
        assert score.details["missing_tools"] == []
        assert score.details["execution_successful"] is True

    def test_partial_completion(self ) -> None:
        """Test partial completion of expected tools."""
        trace = self._make_trace(["search", "fetch"], status="success")
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == pytest.approx(2/3)
        assert score.details["completed_count"] == 2
        assert "summarize" in score.details["missing_tools"]

    def test_no_completion(self) -> None:
        """Test no expected tools completed."""
        trace = self._make_trace(["other_tool"], status="success")
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["completed_count"] == 0
        assert len(score.details["missing_tools"]) == 3

    def test_failed_termination_penalty(self) -> None:
        """Test failed execution status reduces score."""
        trace = self._make_trace(["search", "fetch", "summarize"], status="error")
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        # 1.0 * 0.5 = 0.5
        assert score.value == pytest.approx(0.5)
        assert score.details["execution_successful"] is False

    def test_failed_termination_no_penalty(self) -> None:
        """Test no penalty when require_successful_termination=False."""
        evaluator = TrajectoryCompletenessEvaluator(require_successful_termination=False)
        trace = self._make_trace(["search", "fetch", "summarize"], status="error")
        score = evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["execution_successful"] is False

    def test_extra_tools(self) -> None:
        """Test extra tools don't affect completeness score."""
        trace = self._make_trace(["search", "fetch", "summarize", "extra1", "extra2"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert "extra1" in score.details["extra_tools"]
        assert "extra2" in score.details["extra_tools"]

    def test_no_trace(self) -> None:
        """Test no trace provided."""
        score = self.evaluator.evaluate(self.case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_trace_provided"

    def test_no_expected_trajectory(self) -> None:
        """Test no expected trajectory."""
        case = EvaluationCase(input="test")
        trace = self._make_trace(["search", "fetch"])
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_empty_expected_and_actual(self) -> None:
        """Test empty expected and actual."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_case_insensitive(self) -> None:
        """Test case insensitive comparison."""
        evaluator = TrajectoryCompletenessEvaluator(case_sensitive=False)
        case = EvaluationCase(
            input="test",
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "Search"},
                {"tool": "Fetch"},
            ]),
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="search", arguments={}
            )
        )
        trace.add_event(
            TraceEvent.tool_call(
                execution_id=trace.id, tool_name="fetch", arguments={}
            )
        )
        trace.finalize(status="success")
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_metric_type_and_name(self) -> None:
        """Test metric type and name."""
        assert self.evaluator.metric_type == MetricType.TRAJECTORY_SIMILARITY
        assert self.evaluator.name == "trajectory_completeness"
