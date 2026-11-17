"""Tests for tool evaluation metrics."""

import uuid
from typing import Any

import pytest

from eval_harness.datasets.models import EvaluationCase, ExpectedToolCall
from eval_harness.evaluators import (
    ToolArgumentEvaluator,
    ToolExecutionEvaluator,
    ToolSelectionEvaluator,
)
from eval_harness.metrics.models import MetricType
from eval_harness.tracing.models import (
    ExecutionTrace,
    TraceEvent,
)


class TestToolSelectionEvaluator:
    """Tests for ToolSelectionEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = ToolSelectionEvaluator()
        self.case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "test"}),
                ExpectedToolCall(name="fetch", arguments={"url": "http://example.com"}),
                ExpectedToolCall(name="summarize", arguments={"text": "content"}),
            ],
        )

    def _make_trace(self, tool_names: list[str]) -> ExecutionTrace:
        """Create a trace with tool call events."""
        trace = ExecutionTrace(case_id=self.case.id)
        for name in tool_names:
            trace.add_event(TraceEvent.tool_call(
                execution_id=trace.id,
                tool_name=name,
                arguments={},
            ))
        trace.finalize()
        return trace

    def test_all_tools_match(self) -> None:
        """Test all expected tools present in trace."""
        trace = self._make_trace(["search", "fetch", "summarize"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["precision"] == 1.0
        assert score.details["recall"] == 1.0
        assert score.details["f1"] == 1.0

    def test_partial_match(self) -> None:
        """Test partial tool match."""
        trace = self._make_trace(["search", "fetch"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.details["precision"] == 1.0
        assert score.details["recall"] == pytest.approx(2/3)
        assert score.details["f1"] == pytest.approx(0.8)

    def test_extra_tool_in_trace(self) -> None:
        """Test unexpected tool in trace reduces precision."""
        trace = self._make_trace(["search", "fetch", "summarize", "extra_tool"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.details["precision"] == pytest.approx(3/4)
        assert score.details["recall"] == 1.0

    def test_missing_tool(self) -> None:
        """Test missing tool reduces recall."""
        trace = self._make_trace(["search"])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.details["precision"] == 1.0
        assert score.details["recall"] == pytest.approx(1/3)

    def test_no_trace_provided(self) -> None:
        """Test evaluation without trace returns 0."""
        score = self.evaluator.evaluate(self.case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_trace_provided"

    def test_empty_expected_and_actual(self) -> None:
        """Test empty expected and actual returns 1.0."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["precision"] == 1.0
        assert score.details["recall"] == 1.0

    def test_empty_expected_with_actual(self) -> None:
        """Test unexpected tools when none expected."""
        case = EvaluationCase(input="test")
        trace = self._make_trace(["unexpected_tool"])
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.0

    def test_duplicate_handling_disabled(self) -> None:
        """Test duplicate tool calls counted once by default."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={}),
                ExpectedToolCall(name="search", arguments={}),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="search", arguments={}
        ))
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        # With set-based matching, one match covers both expected
        assert score.details["recall"] == 1.0

    def test_duplicate_handling_enabled(self) -> None:
        """Test duplicate counting with count_duplicates=True."""
        evaluator = ToolSelectionEvaluator(count_duplicates=True)
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={}),
                ExpectedToolCall(name="search", arguments={}),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="search", arguments={}
        ))
        trace.finalize()
        score = evaluator.evaluate(case, None, trace=trace)
        # Only one actual call for two expected
        assert score.details["recall"] == 0.5

    def test_required_only(self) -> None:
        """Test required_only filters optional calls."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="required_tool", arguments={}, required=True),
                ExpectedToolCall(name="optional_tool", arguments={}, required=False),
            ],
        )
        evaluator = ToolSelectionEvaluator(required_only=True)
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="required_tool", arguments={}
        ))
        trace.finalize()
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0
        assert "optional_tool" not in score.details.get("expected_tools", [])

    def test_metric_type_and_name(self) -> None:
        """Test metric type and name are correct."""
        assert self.evaluator.metric_type == MetricType.TOOL_SELECTION_ACCURACY
        assert self.evaluator.name == "tool_selection_accuracy"


class TestToolArgumentEvaluator:
    """Tests for ToolArgumentEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = ToolArgumentEvaluator()
        self.case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "test", "limit": 10}),
                ExpectedToolCall(name="fetch", arguments={"url": "http://example.com"}),
            ],
        )

    def _make_trace(self, calls: list[dict[str, Any]]) -> ExecutionTrace:
        """Create a trace with tool call events."""
        trace = ExecutionTrace(case_id=self.case.id)
        for call in calls:
            trace.add_event(TraceEvent.tool_call(
                execution_id=trace.id,
                tool_name=call["tool"],
                arguments=call.get("arguments", {}),
            ))
        trace.finalize()
        return trace

    def test_exact_match(self) -> None:
        """Test exact argument match returns 1.0."""
        trace = self._make_trace([
            {"tool": "search", "arguments": {"query": "test", "limit": 10}},
            {"tool": "fetch", "arguments": {"url": "http://example.com"}},
        ])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["matched_calls"] == 2

    def test_argument_mismatch(self) -> None:
        """Test mismatched arguments."""
        trace = self._make_trace([
            {"tool": "search", "arguments": {"query": "wrong", "limit": 10}},
            {"tool": "fetch", "arguments": {"url": "http://example.com"}},
        ])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.5
        assert score.details["matched_calls"] == 1

    def test_missing_argument(self) -> None:
        """Test missing argument in actual call."""
        trace = self._make_trace([
            {"tool": "search", "arguments": {"query": "test"}},  # missing limit
            {"tool": "fetch", "arguments": {"url": "http://example.com"}},
        ])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.5
        assert "limit" in score.details["per_tool_results"]["search"][0]["missing_keys"]

    def test_extra_argument(self) -> None:
        """Test extra argument in actual call causes mismatch with exact_match=True."""
        trace = self._make_trace([
            {"tool": "search", "arguments": {"query": "test", "limit": 10, "extra": "ignored"}},
            {"tool": "fetch", "arguments": {"url": "http://example.com"}},
        ])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        # exact_match=True means extra keys cause mismatch
        assert score.value == 0.5

    def test_nested_arguments(self) -> None:
        """Test nested dictionary arguments."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={
                    "params": {"query": "test", "filters": {"type": "article"}}
                }),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id,
            tool_name="search",
            arguments={"params": {"query": "test", "filters": {"type": "article"}}},
        ))
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_nested_argument_mismatch(self) -> None:
        """Test nested argument mismatch."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={
                    "params": {"query": "test", "filters": {"type": "article"}}
                }),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id,
            tool_name="search",
            arguments={"params": {"query": "test", "filters": {"type": "video"}}},
        ))
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.0

    def test_duplicate_expected_calls(self) -> None:
        """Test multiple expected calls to same tool."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "first"}),
                ExpectedToolCall(name="search", arguments={"query": "second"}),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="search", arguments={"query": "first"}
        ))
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="search", arguments={"query": "second"}
        ))
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_partial_duplicate_match(self) -> None:
        """Test partial match for duplicate expected calls."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "first"}),
                ExpectedToolCall(name="search", arguments={"query": "second"}),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="search", arguments={"query": "first"}
        ))
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.5

    def test_extra_actual_call(self) -> None:
        """Test extra actual call beyond expected."""
        trace = self._make_trace([
            {"tool": "search", "arguments": {"query": "test", "limit": 10}},
            {"tool": "fetch", "arguments": {"url": "http://example.com"}},
            {"tool": "search", "arguments": {"query": "extra"}},  # extra
        ])
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0  # Only expected calls evaluated

    def test_no_trace(self) -> None:
        """Test no trace provided."""
        score = self.evaluator.evaluate(self.case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_trace_provided"

    def test_no_expected_calls_with_actual(self) -> None:
        """Test unexpected calls when no expected calls."""
        case = EvaluationCase(input="test")
        trace = self._make_trace([{"tool": "search", "arguments": {}}])
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["reason"] == "unexpected_tool_calls"

    def test_no_expected_no_actual(self) -> None:
        """Test no expected and no actual calls."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_subset_match_mode(self) -> None:
        """Test subset match mode (exact_match=False)."""
        evaluator = ToolArgumentEvaluator(exact_match=False)
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "test"}),
            ],
        )
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id,
            tool_name="search",
            arguments={"query": "test", "limit": 10, "extra": "ignored"},
        ))
        trace.finalize()
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_required_only(self) -> None:
        """Test required_only filters optional calls."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="required", arguments={"arg": 1}, required=True),
                ExpectedToolCall(name="optional", arguments={"arg": 2}, required=False),
            ],
        )
        evaluator = ToolArgumentEvaluator(required_only=True)
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id, tool_name="required", arguments={"arg": 1}
        ))
        trace.finalize()
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_metric_type_and_name(self) -> None:
        """Test metric type and name."""
        assert self.evaluator.metric_type == MetricType.TOOL_CALL_CORRECTNESS
        assert self.evaluator.name == "tool_argument_correctness"


class TestToolExecutionEvaluator:
    """Tests for ToolExecutionEvaluator."""

    def setup_method(self) -> None:
        self.evaluator = ToolExecutionEvaluator()
        self.case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "test"}),
                ExpectedToolCall(name="fetch", arguments={"url": "http://example.com"}),
            ],
        )

    def _make_trace(
        self,
        calls: list[dict[str, Any]],
        responses: list[dict[str, Any]] | None = None,
    ) -> ExecutionTrace:
        """Create a trace with tool call and response events."""
        trace = ExecutionTrace(case_id=self.case.id)
        for _i, call in enumerate(calls):
            corr_id = call.get("correlation_id", uuid.uuid4())
            trace.add_event(TraceEvent.tool_call(
                execution_id=trace.id,
                tool_name=call["tool"],
                arguments=call.get("arguments", {}),
                correlation_id=corr_id,
            ))
        if responses:
            for _i, resp in enumerate(responses):
                corr_id = resp.get("correlation_id", uuid.uuid4())
                trace.add_event(TraceEvent.tool_response(
                    execution_id=trace.id,
                    tool_name=resp["tool"],
                    result=resp.get("result", {}),
                    correlation_id=corr_id,
                ))
        trace.finalize()
        return trace

    def test_all_executed_and_succeeded(self) -> None:
        """Test all expected tools executed and succeeded."""
        corr_0 = uuid.uuid4()
        corr_1 = uuid.uuid4()
        trace = self._make_trace(
            calls=[
                {
                    "tool": "search",
                    "arguments": {"query": "test"},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "arguments": {"url": "http://example.com"},
                    "correlation_id": corr_1,
                },
            ],
            responses=[
                {
                    "tool": "search",
                    "result": {"data": "results"},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "result": {"content": "page"},
                    "correlation_id": corr_1,
                },
            ],
        )
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 1.0
        assert score.details["succeeded"] == 2
        assert score.details["missing"] == 0

    def test_missing_execution(self) -> None:
        """Test missing tool execution."""
        corr_0 = uuid.uuid4()
        trace = self._make_trace(
            calls=[
                {
                    "tool": "search",
                    "arguments": {"query": "test"},
                    "correlation_id": corr_0,
                },
            ],
            responses=[
                {
                    "tool": "search",
                    "result": {"data": "results"},
                    "correlation_id": corr_0,
                },
            ],
        )
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.5
        assert score.details["missing"] == 1

    def test_failed_execution(self) -> None:
        """Test tool execution that failed."""
        corr_0 = uuid.uuid4()
        corr_1 = uuid.uuid4()
        trace = self._make_trace(
            calls=[
                {
                    "tool": "search",
                    "arguments": {"query": "test"},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "arguments": {"url": "http://example.com"},
                    "correlation_id": corr_1,
                },
            ],
            responses=[
                {
                    "tool": "search",
                    "result": {"data": "results"},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "result": {"error": "timeout"},
                    "correlation_id": corr_1,
                },
            ],
        )
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.5
        assert score.details["failed"] == 1

    def test_error_event(self) -> None:
        """Test error event marks execution as failed."""
        trace = ExecutionTrace(case_id=self.case.id)
        corr_0 = uuid.uuid4()
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id,
            tool_name="search",
            arguments={"query": "test"},
            correlation_id=corr_0,
        ))
        trace.add_event(TraceEvent.create_error(
            execution_id=trace.id,
            error="Tool failed",
            correlation_id=corr_0,
        ))
        trace.finalize()
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        # Two expected calls: search (failed) and fetch (missing) -> 0/2 = 0.0
        assert score.value == 0.0
        assert any(r["error"] == "Tool failed" for r in score.details["per_tool_results"])

    def test_no_trace(self) -> None:
        """Test no trace provided."""
        score = self.evaluator.evaluate(self.case, None)
        assert score.value == 0.0
        assert score.details["reason"] == "no_trace_provided"
        assert score.details["missing"] == 2

    def test_unexpected_tool_call(self) -> None:
        """Test unexpected tool call in trace."""
        corr_0 = uuid.uuid4()
        corr_1 = uuid.uuid4()
        corr_2 = uuid.uuid4()
        trace = self._make_trace(
            calls=[
                {
                    "tool": "search",
                    "arguments": {"query": "test"},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "arguments": {"url": "http://example.com"},
                    "correlation_id": corr_1,
                },
                {
                    "tool": "unexpected",
                    "arguments": {},
                    "correlation_id": corr_2,
                },
            ],
            responses=[
                {
                    "tool": "search",
                    "result": {},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "result": {},
                    "correlation_id": corr_1,
                },
                {
                    "tool": "unexpected",
                    "result": {},
                    "correlation_id": corr_2,
                },
            ],
        )
        score = self.evaluator.evaluate(self.case, None, trace=trace)
        assert score.details["executed"] == 3
        assert any(r["error"] == "Unexpected tool call" for r in score.details["per_tool_results"])

    def test_no_expected_with_actual(self) -> None:
        """Test unexpected calls when no expected."""
        case = EvaluationCase(input="test")
        corr_0 = uuid.uuid4()
        trace = self._make_trace(
            calls=[
                {
                    "tool": "search",
                    "arguments": {},
                    "correlation_id": corr_0,
                },
            ],
            responses=[
                {
                    "tool": "search",
                    "result": {},
                    "correlation_id": corr_0,
                },
            ],
        )
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 0.0
        assert score.details["reason"] == "unexpected_tool_calls"

    def test_no_expected_no_actual(self) -> None:
        """Test no expected and no actual."""
        case = EvaluationCase(input="test")
        trace = ExecutionTrace(case_id=case.id)
        trace.finalize()
        score = self.evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_argument_checking(self) -> None:
        """Test argument checking when enabled."""
        evaluator = ToolExecutionEvaluator(check_arguments=True)
        corr_0 = uuid.uuid4()
        corr_1 = uuid.uuid4()
        trace = self._make_trace(
            calls=[
                {
                    "tool": "search",
                    "arguments": {"query": "test"},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "arguments": {"url": "http://wrong.com"},  # wrong arg
                    "correlation_id": corr_1,
                },
            ],
            responses=[
                {
                    "tool": "search",
                    "result": {},
                    "correlation_id": corr_0,
                },
                {
                    "tool": "fetch",
                    "result": {},
                    "correlation_id": corr_1,
                },
            ],
        )
        score = evaluator.evaluate(self.case, None, trace=trace)
        assert score.value == 0.5
        assert not score.details["per_tool_results"][1]["arguments_match"]

    def test_required_only(self) -> None:
        """Test required_only filters optional calls."""
        case = EvaluationCase(
            input="test",
            expected_tool_calls=[
                ExpectedToolCall(name="required", arguments={}, required=True),
                ExpectedToolCall(name="optional", arguments={}, required=False),
            ],
        )
        evaluator = ToolExecutionEvaluator(required_only=True)
        trace = ExecutionTrace(case_id=case.id)
        corr_0 = uuid.uuid4()
        trace.add_event(TraceEvent.tool_call(
            execution_id=trace.id,
            tool_name="required",
            arguments={},
            correlation_id=corr_0,
        ))
        trace.add_event(TraceEvent.tool_response(
            execution_id=trace.id,
            tool_name="required",
            result={},
            correlation_id=corr_0,
        ))
        trace.finalize()
        score = evaluator.evaluate(case, None, trace=trace)
        assert score.value == 1.0

    def test_metric_type_and_name(self) -> None:
        """Test metric type and name."""
        assert self.evaluator.metric_type == MetricType.TOOL_CALL_CORRECTNESS
        assert self.evaluator.name == "tool_execution_correctness"
