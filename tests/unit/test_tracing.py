"""Tests for tracing models."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from eval_harness.tracing.models import (
    CostEstimate,
    ExecutionTrace,
    LatencyMetrics,
    TokenUsage,
    TraceEvent,
    TraceEventType,
)


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_with_values(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_explicit_total(self) -> None:
        # total_tokens is always computed from prompt + completion on validation
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150
        # Changing prompt_tokens doesn't auto-update total_tokens (validate_assignment=False)
        # Create a new instance for updated values
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=50)
        assert usage2.total_tokens == 250

    def test_addition(self) -> None:
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100)
        combined = usage1 + usage2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450

    def test_invalid_negative(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)


class TestLatencyMetrics:
    """Tests for LatencyMetrics model."""

    def test_basic(self) -> None:
        start = datetime.now(UTC)
        latency = LatencyMetrics(start_time=start)
        assert latency.start_time == start
        assert latency.end_time is None
        assert latency.duration_ms is None

    def test_with_end_time(self) -> None:
        start = datetime.now(UTC)
        end = start + timedelta(milliseconds=150)
        latency = LatencyMetrics(start_time=start, end_time=end)
        assert latency.duration_ms == 150.0

    def test_finalize(self) -> None:
        start = datetime.now(UTC)
        latency = LatencyMetrics(start_time=start)
        latency.finalize()
        assert latency.end_time is not None
        assert latency.duration_ms is not None
        assert latency.duration_ms >= 0

    def test_finalize_with_time(self) -> None:
        start = datetime.now(UTC)
        end = start + timedelta(milliseconds=200)
        latency = LatencyMetrics(start_time=start)
        latency.finalize(end_time=end)
        assert latency.end_time == end
        assert latency.duration_ms == 200.0

    def test_queue_and_processing_time(self) -> None:
        latency = LatencyMetrics(
            start_time=datetime.now(UTC),
            queue_time_ms=10.0,
            processing_time_ms=90.0,
        )
        assert latency.queue_time_ms == 10.0
        assert latency.processing_time_ms == 90.0


class TestCostEstimate:
    """Tests for CostEstimate model."""

    def test_defaults(self) -> None:
        cost = CostEstimate()
        assert cost.input_cost_usd == 0.0
        assert cost.output_cost_usd == 0.0
        assert cost.total_cost_usd == 0.0

    def test_with_values(self) -> None:
        cost = CostEstimate(input_cost_usd=0.01, output_cost_usd=0.02)
        assert cost.input_cost_usd == 0.01
        assert cost.output_cost_usd == 0.02
        assert cost.total_cost_usd == 0.03

    def test_explicit_total(self) -> None:
        # total_cost_usd is always computed from input + output on validation
        cost = CostEstimate(input_cost_usd=0.01, output_cost_usd=0.02)
        assert cost.total_cost_usd == 0.03
        # Changing input_cost_usd doesn't auto-update total_cost_usd (validate_assignment=False)
        # Create a new instance for updated values
        cost2 = CostEstimate(input_cost_usd=0.05, output_cost_usd=0.02)
        assert cost2.total_cost_usd == 0.07

    def test_addition(self) -> None:
        cost1 = CostEstimate(input_cost_usd=0.01, output_cost_usd=0.02)
        cost2 = CostEstimate(input_cost_usd=0.03, output_cost_usd=0.04)
        combined = cost1 + cost2
        assert combined.input_cost_usd == 0.04
        assert combined.output_cost_usd == 0.06
        assert combined.total_cost_usd == 0.10

    def test_with_model_info(self) -> None:
        cost = CostEstimate(
            input_cost_usd=0.01,
            output_cost_usd=0.02,
            model_name="gpt-4",
            pricing_version="2024-01",
        )
        assert cost.model_name == "gpt-4"
        assert cost.pricing_version == "2024-01"

    def test_invalid_negative(self) -> None:
        with pytest.raises(ValidationError):
            CostEstimate(input_cost_usd=-0.01)


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_minimal_event(self) -> None:
        event = TraceEvent(event_type=TraceEventType.LLM_CALL)
        assert isinstance(event.id, UUID)
        assert event.event_type == TraceEventType.LLM_CALL
        assert isinstance(event.timestamp, datetime)
        assert event.data == {}
        assert event.tags == []

    def test_full_event(self) -> None:
        event = TraceEvent(
            event_type=TraceEventType.TOOL_CALL,
            parent_id=uuid4(),
            correlation_id=uuid4(),
            data={"tool": "search", "args": {"q": "test"}},
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            latency=LatencyMetrics(start_time=datetime.now(UTC)),
            cost=CostEstimate(input_cost_usd=0.001),
            tags=["tool", "search"],
            metadata={"source": "agent"},
        )
        assert event.event_type == TraceEventType.TOOL_CALL
        assert event.parent_id is not None
        assert event.correlation_id is not None
        assert event.data["tool"] == "search"
        assert event.token_usage is not None
        assert event.latency is not None
        assert event.cost is not None
        assert "tool" in event.tags
        assert event.metadata["source"] == "agent"

    def test_event_types(self) -> None:
        for event_type in TraceEventType:
            event = TraceEvent(event_type=event_type)
            assert event.event_type == event_type


class TestExecutionTrace:
    """Tests for ExecutionTrace model."""

    def test_minimal_trace(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        assert isinstance(trace.id, UUID)
        assert trace.case_id is not None
        assert trace.events == []
        assert trace.status == "in_progress"
        assert trace.total_tokens.prompt_tokens == 0

    def test_add_event(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        event = TraceEvent(
            event_type=TraceEventType.LLM_CALL,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            cost=CostEstimate(input_cost_usd=0.01, output_cost_usd=0.02),
        )
        trace.add_event(event)
        assert len(trace.events) == 1
        assert trace.total_tokens.prompt_tokens == 100
        assert trace.total_tokens.completion_tokens == 50
        assert trace.total_cost.input_cost_usd == 0.01
        assert trace.total_cost.output_cost_usd == 0.02

    def test_finalize_success(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.finalize(status="success")
        assert trace.status == "success"
        assert trace.end_time is not None
        assert trace.total_duration_ms is not None
        assert trace.total_duration_ms >= 0

    def test_finalize_error(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.finalize(status="error", error="Something failed")
        assert trace.status == "error"
        assert trace.error == "Something failed"

    def test_get_events_by_type(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_CALL))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_RESPONSE))

        llm_events = trace.get_events_by_type(TraceEventType.LLM_CALL)
        assert len(llm_events) == 1

        tool_events = trace.get_events_by_type(TraceEventType.TOOL_CALL)
        assert len(tool_events) == 1

    def test_get_child_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        parent_id = uuid4()
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL, id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_CALL, parent_id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_RESPONSE, parent_id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_RESPONSE))

        children = trace.get_child_events(parent_id)
        assert len(children) == 2
        assert all(c.parent_id == parent_id for c in children)

    def test_run_id(self) -> None:
        run_id = uuid4()
        trace = ExecutionTrace(case_id=uuid4(), run_id=run_id)
        assert trace.run_id == run_id
