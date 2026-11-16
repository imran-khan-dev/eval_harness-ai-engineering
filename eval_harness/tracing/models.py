"""Tracing models for execution traces and events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class TraceEventType(str, Enum):
    """Types of trace events."""

    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    AGENT_STEP = "agent_step"
    AGENT_DECISION = "agent_decision"
    ERROR = "error"
    CUSTOM = "custom"


class TokenUsage(BaseModel):
    """Token usage for an LLM call."""

    prompt_tokens: int = Field(default=0, ge=0, description="Input/prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Output/completion tokens")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens")

    @model_validator(mode="after")
    def compute_total(self) -> TokenUsage:
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    model_config = {
        "validate_assignment": False,
        "extra": "forbid",
    }


class LatencyMetrics(BaseModel):
    """Latency measurements for an operation."""

    start_time: datetime = Field(description="Operation start time")
    end_time: datetime | None = Field(None, description="Operation end time")
    duration_ms: float | None = Field(default=None, ge=0, description="Duration in milliseconds")
    queue_time_ms: float = Field(default=0.0, ge=0, description="Queue/wait time in ms")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time in ms")

    @model_validator(mode="after")
    def compute_duration(self) -> LatencyMetrics:
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        return self

    def finalize(self, end_time: datetime | None = None) -> None:
        """Mark the operation as complete."""
        self.end_time = end_time or datetime.now(timezone.utc)
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    model_config = {
        "validate_assignment": False,
        "extra": "forbid",
    }


class CostEstimate(BaseModel):
    """Estimated cost for an LLM operation."""

    input_cost_usd: float = Field(default=0.0, ge=0, description="Input token cost in USD")
    output_cost_usd: float = Field(default=0.0, ge=0, description="Output token cost in USD")
    total_cost_usd: float = Field(default=0.0, ge=0, description="Total cost in USD")
    model_name: str | None = Field(None, description="Model used for cost calculation")
    pricing_version: str | None = Field(None, description="Pricing data version")

    @model_validator(mode="after")
    def compute_total(self) -> CostEstimate:
        self.total_cost_usd = self.input_cost_usd + self.output_cost_usd
        return self

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            model_name=self.model_name or other.model_name,
            pricing_version=self.pricing_version or other.pricing_version,
        )

    model_config = {
        "validate_assignment": False,
        "extra": "forbid",
    }


class TraceEvent(BaseModel):
    """A single event in an execution trace.

    Attributes:
        id: Unique event identifier.
        event_type: Type of the trace event.
        timestamp: When the event occurred.
        parent_id: Optional parent event ID for nesting.
        correlation_id: Optional correlation ID for grouping.
        data: Event-specific data payload.
        token_usage: Optional token usage for LLM events.
        latency: Optional latency metrics.
        cost: Optional cost estimate.
        tags: Tags for filtering events.
        metadata: Additional arbitrary metadata.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    event_type: TraceEventType = Field(..., description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Event timestamp"
    )
    parent_id: UUID | None = Field(None, description="Parent event ID")
    correlation_id: UUID | None = Field(None, description="Correlation ID for grouping")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data payload")
    token_usage: TokenUsage | None = Field(None, description="Token usage for LLM events")
    latency: LatencyMetrics | None = Field(None, description="Latency metrics")
    cost: CostEstimate | None = Field(None, description="Cost estimate")
    tags: list[str] = Field(default_factory=list, description="Event tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }


class ExecutionTrace(BaseModel):
    """A complete execution trace for an evaluation case.

    Attributes:
        id: Unique trace identifier.
        case_id: ID of the evaluation case this trace belongs to.
        run_id: Optional run/session identifier.
        events: Ordered list of trace events.
        start_time: Trace start timestamp.
        end_time: Trace end timestamp.
        total_tokens: Aggregated token usage.
        total_cost: Aggregated cost estimate.
        total_duration_ms: Total trace duration.
        status: Trace status (success, error, partial).
        error: Optional error message if failed.
        metadata: Additional arbitrary metadata.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique trace identifier")
    case_id: UUID = Field(..., description="Evaluation case ID")
    run_id: UUID | None = Field(None, description="Run/session ID")
    events: list[TraceEvent] = Field(default_factory=list, description="Trace events")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Trace start time"
    )
    end_time: datetime | None = Field(None, description="Trace end time")
    total_tokens: TokenUsage = Field(
        default_factory=lambda: TokenUsage(), description="Aggregated token usage"
    )
    total_cost: CostEstimate = Field(
        default_factory=lambda: CostEstimate(
            input_cost_usd=0.0,
            output_cost_usd=0.0,
            total_cost_usd=0.0,
            model_name=None,
            pricing_version=None,
        ),
        description="Aggregated cost estimate",
    )
    total_duration_ms: float | None = Field(None, ge=0, description="Total duration")
    status: str = Field(default="in_progress", description="Trace status")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Trace metadata")

    def add_event(self, event: TraceEvent) -> None:
        """Add an event to the trace and update aggregates."""
        self.events.append(event)
        if event.token_usage:
            self.total_tokens += event.token_usage
        if event.cost:
            self.total_cost += event.cost

    def finalize(self, status: str = "success", error: str | None = None) -> None:
        """Mark the trace as complete."""
        self.end_time = datetime.now(timezone.utc)
        self.status = status
        self.error = error
        if self.start_time and self.end_time:
            self.total_duration_ms = (
                self.end_time - self.start_time
            ).total_seconds() * 1000

    def get_events_by_type(self, event_type: TraceEventType) -> list[TraceEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_child_events(self, parent_id: UUID) -> list[TraceEvent]:
        """Get all child events of a parent event."""
        return [e for e in self.events if e.parent_id == parent_id]

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }
