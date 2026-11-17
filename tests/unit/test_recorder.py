"""Tests for InMemoryTraceRecorder."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from eval_harness.tracing.models import (
    TraceEvent,
    TraceEventType,
)
from eval_harness.tracing.recorder import (
    ExecutionAlreadyStartedError,
    ExecutionCompletedError,
    ExecutionNotFoundError,
    InMemoryTraceRecorder,
)


class TestInMemoryTraceRecorder:
    """Tests for InMemoryTraceRecorder."""

    def test_start_execution(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        trace = recorder.start_execution(execution_id, case_id)

        assert trace.id == execution_id
        assert trace.case_id == case_id
        assert trace.status == "in_progress"
        assert recorder.has_execution(execution_id)

    def test_start_execution_with_run_id(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()
        run_id = uuid4()

        trace = recorder.start_execution(execution_id, case_id, run_id=run_id)

        assert trace.run_id == run_id

    def test_start_execution_with_metadata(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()
        metadata = {"source": "test", "version": "1.0"}

        trace = recorder.start_execution(execution_id, case_id, metadata=metadata)

        assert trace.metadata == metadata

    def test_duplicate_execution_start_raises(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        with pytest.raises(ExecutionAlreadyStartedError) as exc_info:
            recorder.start_execution(execution_id, case_id)

        assert exc_info.value.execution_id == execution_id

    def test_end_execution(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)
        trace = recorder.end_execution(execution_id, status="success")

        assert trace.status == "success"
        assert trace.end_time is not None
        assert trace.total_duration_ms is not None

    def test_end_execution_with_error(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)
        trace = recorder.end_execution(execution_id, status="error", error="Something failed")

        assert trace.status == "error"
        assert trace.error == "Something failed"

    def test_end_nonexistent_execution_raises(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()

        with pytest.raises(ExecutionNotFoundError) as exc_info:
            recorder.end_execution(execution_id)

        assert exc_info.value.execution_id == execution_id

    def test_record_event(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        event = TraceEvent(
            event_type=TraceEventType.LLM_CALL,
            execution_id=execution_id,
            data={"prompt": "Hello"},
        )
        recorder.record_event(event)

        events = recorder.get_events(execution_id)
        assert len(events) == 1
        assert events[0].event_type == TraceEventType.LLM_CALL
        assert events[0].data["prompt"] == "Hello"

    def test_record_multiple_events_in_order(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        event1 = TraceEvent(event_type=TraceEventType.EXECUTION_START, execution_id=execution_id)
        event2 = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)
        event3 = TraceEvent(event_type=TraceEventType.LLM_RESPONSE, execution_id=execution_id)
        event4 = TraceEvent(event_type=TraceEventType.EXECUTION_END, execution_id=execution_id)

        recorder.record_event(event1)
        recorder.record_event(event2)
        recorder.record_event(event3)
        recorder.record_event(event4)

        events = recorder.get_events(execution_id)
        assert len(events) == 4
        assert events[0].event_type == TraceEventType.EXECUTION_START
        assert events[1].event_type == TraceEventType.LLM_CALL
        assert events[2].event_type == TraceEventType.LLM_RESPONSE
        assert events[3].event_type == TraceEventType.EXECUTION_END

    def test_record_event_unknown_execution_raises(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()

        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)

        with pytest.raises(ExecutionNotFoundError) as exc_info:
            recorder.record_event(event)

        assert exc_info.value.execution_id == execution_id

    def test_record_event_no_execution_id_raises(self) -> None:
        recorder = InMemoryTraceRecorder()

        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=None)

        with pytest.raises(ValueError, match="must have an execution_id"):
            recorder.record_event(event)

    def test_record_event_after_completion_raises(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)
        recorder.end_execution(execution_id, status="success")

        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)

        with pytest.raises(ExecutionCompletedError) as exc_info:
            recorder.record_event(event)

        assert exc_info.value.execution_id == execution_id

    def test_get_execution_returns_copy(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        trace1 = recorder.get_execution(execution_id)
        trace2 = recorder.get_execution(execution_id)

        # Should be separate copies
        assert trace1 is not trace2
        assert trace1.id == trace2.id == execution_id

        # Modifying one should not affect the other
        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)
        trace1.events.append(event)
        trace2_copy = recorder.get_execution(execution_id)
        assert len(trace2_copy.events) == 0

    def test_get_execution_unknown_raises(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()

        with pytest.raises(ExecutionNotFoundError):
            recorder.get_execution(execution_id)

    def test_get_all_executions(self) -> None:
        recorder = InMemoryTraceRecorder()

        exec_id1 = uuid4()
        exec_id2 = uuid4()
        case_id = uuid4()

        recorder.start_execution(exec_id1, case_id)
        recorder.start_execution(exec_id2, case_id)

        executions = recorder.get_all_executions()

        assert len(executions) == 2
        exec_ids = {e.id for e in executions}
        assert exec_ids == {exec_id1, exec_id2}

    def test_get_all_executions_returns_copies(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        executions1 = recorder.get_all_executions()
        executions2 = recorder.get_all_executions()

        # Should be separate copies
        assert executions1[0] is not executions2[0]

    def test_get_events_returns_copy(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)
        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)
        recorder.record_event(event)

        events1 = recorder.get_events(execution_id)
        events2 = recorder.get_events(execution_id)

        assert events1 is not events2
        assert len(events1) == len(events2) == 1

    def test_get_events_unknown_raises(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()

        with pytest.raises(ExecutionNotFoundError):
            recorder.get_events(execution_id)

    def test_has_execution(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        assert recorder.has_execution(execution_id) is False

        recorder.start_execution(execution_id, case_id)
        assert recorder.has_execution(execution_id) is True

    def test_is_completed(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        assert recorder.is_completed(execution_id) is False

        recorder.start_execution(execution_id, case_id)
        assert recorder.is_completed(execution_id) is False

        recorder.end_execution(execution_id, status="success")
        assert recorder.is_completed(execution_id) is True

    def test_execution_with_multiple_events_and_aggregates(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        # Add LLM call with token usage
        from eval_harness.tracing.models import CostEstimate, TokenUsage

        token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        cost = CostEstimate(input_cost_usd=0.01, output_cost_usd=0.02)

        event1 = TraceEvent.llm_call(execution_id, "prompt", "gpt-4")
        event2 = TraceEvent.llm_response(
            execution_id, "response", "gpt-4", token_usage=token_usage, cost=cost
        )

        recorder.record_event(event1)
        recorder.record_event(event2)

        trace = recorder.get_execution(execution_id)
        assert trace.total_tokens.prompt_tokens == 100
        assert trace.total_tokens.completion_tokens == 50
        assert trace.total_cost.input_cost_usd == 0.01
        assert trace.total_cost.output_cost_usd == 0.02

    def test_multiple_independent_executions(self) -> None:
        recorder = InMemoryTraceRecorder()

        exec_id1 = uuid4()
        exec_id2 = uuid4()
        case_id1 = uuid4()
        case_id2 = uuid4()

        recorder.start_execution(exec_id1, case_id1)
        recorder.start_execution(exec_id2, case_id2)

        # Record events for first execution
        recorder.record_event(TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=exec_id1))
        recorder.record_event(TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=exec_id1))

        # Record events for second execution
        recorder.record_event(TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=exec_id2))

        events1 = recorder.get_events(exec_id1)
        events2 = recorder.get_events(exec_id2)

        assert len(events1) == 2
        assert len(events2) == 1

    def test_record_event_after_error_status(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)
        recorder.end_execution(execution_id, status="error", error="Failed")

        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)

        with pytest.raises(ExecutionCompletedError):
            recorder.record_event(event)

    def test_record_event_after_partial_status(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)
        recorder.end_execution(execution_id, status="partial")

        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)

        with pytest.raises(ExecutionCompletedError):
            recorder.record_event(event)

    def test_get_execution_preserves_events_order(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        # Record events out of chronological order to verify they stay in recorded order
        recorder.record_event(
            TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)
        )
        recorder.record_event(
            TraceEvent(event_type=TraceEventType.EXECUTION_START, execution_id=execution_id)
        )
        recorder.record_event(
            TraceEvent(event_type=TraceEventType.LLM_RESPONSE, execution_id=execution_id)
        )

        events = recorder.get_events(execution_id)
        # Order should be the order they were recorded
        assert events[0].event_type == TraceEventType.LLM_CALL
        assert events[1].event_type == TraceEventType.EXECUTION_START
        assert events[2].event_type == TraceEventType.LLM_RESPONSE

    def test_execution_trace_aggregates_on_event_recording(self) -> None:
        recorder = InMemoryTraceRecorder()
        execution_id = uuid4()
        case_id = uuid4()

        recorder.start_execution(execution_id, case_id)

        from eval_harness.tracing.models import CostEstimate, LatencyMetrics, TokenUsage

        # Record LLM call
        recorder.record_event(
            TraceEvent.llm_call(execution_id, "prompt", "gpt-4")
        )

        # Record LLM response with tokens and cost
        token_usage = TokenUsage(prompt_tokens=200, completion_tokens=100)
        cost = CostEstimate(input_cost_usd=0.02, output_cost_usd=0.04)
        latency = LatencyMetrics(start_time=datetime.now(UTC))

        recorder.record_event(
            TraceEvent.llm_response(
                execution_id,
                "response",
                "gpt-4",
                token_usage=token_usage,
                cost=cost,
                latency=latency,
            )
        )

        # Record tool call
        recorder.record_event(
            TraceEvent.tool_call(execution_id, "search", {"query": "test"})
        )

        trace = recorder.get_execution(execution_id)

        # Verify aggregates
        assert trace.total_tokens.prompt_tokens == 200
        assert trace.total_tokens.completion_tokens == 100
        assert trace.total_cost.input_cost_usd == 0.02
        assert trace.total_cost.output_cost_usd == 0.04

        # Verify events
        assert len(trace.events) == 3
        assert len(trace.get_llm_events()) == 2
        assert len(trace.get_tool_events()) == 1
