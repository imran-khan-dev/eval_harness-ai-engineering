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


class TestTraceEventFactoryMethods:
    """Tests for TraceEvent factory methods."""

    def test_llm_call_factory(self) -> None:
        execution_id = uuid4()
        event = TraceEvent.llm_call(
            execution_id=execution_id,
            prompt="Hello, world!",
            model="gpt-4",
            parent_id=uuid4(),
            correlation_id=uuid4(),
        )
        assert event.event_type == TraceEventType.LLM_CALL
        assert event.execution_id == execution_id
        assert event.data["prompt"] == "Hello, world!"
        assert event.data["model"] == "gpt-4"
        assert event.parent_id is not None
        assert event.correlation_id is not None

    def test_llm_response_factory(self) -> None:
        execution_id = uuid4()
        token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        cost = CostEstimate(input_cost_usd=0.01, output_cost_usd=0.02)
        latency = LatencyMetrics(start_time=datetime.now(UTC))
        event = TraceEvent.llm_response(
            execution_id=execution_id,
            response="Hello!",
            model="gpt-4",
            token_usage=token_usage,
            cost=cost,
            latency=latency,
            parent_id=uuid4(),
            correlation_id=uuid4(),
        )
        assert event.event_type == TraceEventType.LLM_RESPONSE
        assert event.execution_id == execution_id
        assert event.data["response"] == "Hello!"
        assert event.data["model"] == "gpt-4"
        assert event.token_usage is not None
        assert event.token_usage.prompt_tokens == 100
        assert event.cost is not None
        assert event.cost.total_cost_usd == 0.03
        assert event.latency is not None

    def test_tool_call_factory(self) -> None:
        execution_id = uuid4()
        event = TraceEvent.tool_call(
            execution_id=execution_id,
            tool_name="search",
            arguments={"query": "test"},
            parent_id=uuid4(),
            correlation_id=uuid4(),
        )
        assert event.event_type == TraceEventType.TOOL_CALL
        assert event.execution_id == execution_id
        assert event.data["tool"] == "search"
        assert event.data["arguments"] == {"query": "test"}

    def test_tool_response_factory(self) -> None:
        execution_id = uuid4()
        latency = LatencyMetrics(start_time=datetime.now(UTC))
        event = TraceEvent.tool_response(
            execution_id=execution_id,
            tool_name="search",
            result={"results": ["item1", "item2"]},
            latency=latency,
        )
        assert event.event_type == TraceEventType.TOOL_RESPONSE
        assert event.execution_id == execution_id
        assert event.data["tool"] == "search"
        assert event.data["result"] == {"results": ["item1", "item2"]}
        assert event.latency is not None

    def test_agent_iteration_factory(self) -> None:
        execution_id = uuid4()
        event = TraceEvent.agent_iteration(
            execution_id=execution_id,
            iteration=1,
            action="think",
            parent_id=uuid4(),
            correlation_id=uuid4(),
        )
        assert event.event_type == TraceEventType.AGENT_ITERATION
        assert event.execution_id == execution_id
        assert event.data["iteration"] == 1
        assert event.data["action"] == "think"

    def test_execution_start_factory(self) -> None:
        execution_id = uuid4()
        case_id = uuid4()
        event = TraceEvent.execution_start(
            execution_id=execution_id,
            case_id=case_id,
            correlation_id=uuid4(),
        )
        assert event.event_type == TraceEventType.EXECUTION_START
        assert event.execution_id == execution_id
        assert event.data["case_id"] == str(case_id)

    def test_execution_end_factory(self) -> None:
        execution_id = uuid4()
        parent_id = uuid4()
        event = TraceEvent.execution_end(
            execution_id=execution_id,
            status="success",
            parent_id=parent_id,
            correlation_id=uuid4(),
            duration_ms=150.5,
        )
        assert event.event_type == TraceEventType.EXECUTION_END
        assert event.execution_id == execution_id
        assert event.parent_id == parent_id
        assert event.duration_ms == 150.5
        assert event.data["status"] == "success"

    def test_error_factory(self) -> None:
        execution_id = uuid4()
        event = TraceEvent.create_error(
            execution_id=execution_id,
            error="Connection timeout",
            parent_id=uuid4(),
            correlation_id=uuid4(),
        )
        assert event.event_type == TraceEventType.ERROR
        assert event.execution_id == execution_id
        assert event.error == "Connection timeout"


class TestTraceEventHelperMethods:
    """Tests for TraceEvent helper methods."""

    def test_is_start_event(self) -> None:
        event = TraceEvent.execution_start(execution_id=uuid4(), case_id=uuid4())
        assert event.is_start_event() is True

        event2 = TraceEvent(event_type=TraceEventType.LLM_CALL)
        assert event2.is_start_event() is False

    def test_is_end_event(self) -> None:
        event = TraceEvent.execution_end(execution_id=uuid4(), status="success")
        assert event.is_end_event() is True

        event2 = TraceEvent(event_type=TraceEventType.LLM_CALL)
        assert event2.is_end_event() is False

    def test_is_error_event(self) -> None:
        event = TraceEvent.create_error(execution_id=uuid4(), error="test error")
        assert event.is_error_event() is True

        event2 = TraceEvent(event_type=TraceEventType.LLM_CALL)
        assert event2.is_error_event() is False


class TestTraceEventNewFields:
    """Tests for new TraceEvent fields."""

    def test_execution_id_field(self) -> None:
        execution_id = uuid4()
        event = TraceEvent(event_type=TraceEventType.LLM_CALL, execution_id=execution_id)
        assert event.execution_id == execution_id

    def test_duration_ms_field(self) -> None:
        event = TraceEvent(event_type=TraceEventType.LLM_CALL, duration_ms=123.45)
        assert event.duration_ms == 123.45

    def test_duration_ms_validation(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(event_type=TraceEventType.LLM_CALL, duration_ms=-10)

    def test_error_field(self) -> None:
        event = TraceEvent(event_type=TraceEventType.ERROR, error="Something went wrong")
        assert event.error == "Something went wrong"

    def test_new_event_types_exist(self) -> None:
        assert TraceEventType.AGENT_ITERATION == "agent_iteration"
        assert TraceEventType.EXECUTION_START == "execution_start"
        assert TraceEventType.EXECUTION_END == "execution_end"


class TestExecutionTraceNewMethods:
    """Tests for new ExecutionTrace methods."""

    def test_get_llm_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_RESPONSE))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_CALL))

        llm_events = trace.get_llm_events()
        assert len(llm_events) == 2
        llm_types = (TraceEventType.LLM_CALL, TraceEventType.LLM_RESPONSE)
        assert all(e.event_type in llm_types for e in llm_events)

    def test_get_tool_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_CALL))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_RESPONSE))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL))

        tool_events = trace.get_tool_events()
        assert len(tool_events) == 2
        tool_types = (TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESPONSE)
        assert all(e.event_type in tool_types for e in tool_events)

    def test_get_agent_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.add_event(TraceEvent(event_type=TraceEventType.AGENT_STEP))
        trace.add_event(TraceEvent(event_type=TraceEventType.AGENT_ITERATION))
        trace.add_event(TraceEvent(event_type=TraceEventType.AGENT_DECISION))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL))

        agent_events = trace.get_agent_events()
        assert len(agent_events) == 3
        assert all(
            e.event_type
            in (
                TraceEventType.AGENT_STEP,
                TraceEventType.AGENT_ITERATION,
                TraceEventType.AGENT_DECISION,
            )
            for e in agent_events
        )

    def test_get_error_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.add_event(TraceEvent(event_type=TraceEventType.ERROR))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL))

        error_events = trace.get_error_events()
        assert len(error_events) == 1
        assert error_events[0].event_type == TraceEventType.ERROR

    def test_get_execution_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        trace.add_event(TraceEvent(event_type=TraceEventType.EXECUTION_START))
        trace.add_event(TraceEvent(event_type=TraceEventType.EXECUTION_END))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL))

        exec_events = trace.get_execution_events()
        assert len(exec_events) == 2
        assert all(
            e.event_type in (TraceEventType.EXECUTION_START, TraceEventType.EXECUTION_END)
            for e in exec_events
        )

    def test_get_root_events(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        parent_id = uuid4()
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL, id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_CALL, parent_id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_RESPONSE, parent_id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_RESPONSE))

        root_events = trace.get_root_events()
        assert len(root_events) == 2  # parent_id is None for root events

    def test_get_event_tree(self) -> None:
        trace = ExecutionTrace(case_id=uuid4())
        parent_id = uuid4()
        child_id = uuid4()
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_CALL, id=parent_id))
        trace.add_event(
            TraceEvent(
                event_type=TraceEventType.TOOL_CALL, id=child_id, parent_id=parent_id
            )
        )
        trace.add_event(TraceEvent(event_type=TraceEventType.TOOL_RESPONSE, parent_id=parent_id))
        trace.add_event(TraceEvent(event_type=TraceEventType.LLM_RESPONSE))

        tree = trace.get_event_tree()
        assert parent_id in tree
        assert len(tree[parent_id]) == 2
        assert all(e.parent_id == parent_id for e in tree[parent_id])


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
