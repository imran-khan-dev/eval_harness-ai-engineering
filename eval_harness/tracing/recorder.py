"""In-memory trace recorder for execution traces."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from eval_harness.tracing.models import ExecutionTrace, TraceEvent


class ExecutionNotFoundError(Exception):
    """Raised when an execution is not found."""

    def __init__(self, execution_id: UUID) -> None:
        self.execution_id = execution_id
        super().__init__(f"Execution not found: {execution_id}")


class ExecutionAlreadyStartedError(Exception):
    """Raised when attempting to start an already started execution."""

    def __init__(self, execution_id: UUID) -> None:
        self.execution_id = execution_id
        super().__init__(f"Execution already started: {execution_id}")


class ExecutionCompletedError(Exception):
    """Raised when attempting to modify a completed execution."""

    def __init__(self, execution_id: UUID) -> None:
        self.execution_id = execution_id
        super().__init__(f"Execution already completed: {execution_id}")


class InMemoryTraceRecorder:
    """In-memory recorder for execution traces.

    Manages the lifecycle of execution traces and their events.
    All operations are in-memory and not thread-safe.
    """

    def __init__(self) -> None:
        self._executions: dict[UUID, ExecutionTrace] = {}

    def start_execution(
        self,
        execution_id: UUID,
        case_id: UUID,
        run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionTrace:
        """Start a new execution trace.

        Args:
            execution_id: Unique identifier for this execution.
            case_id: Evaluation case ID this execution belongs to.
            run_id: Optional run/session identifier.
            metadata: Optional metadata for the trace.

        Returns:
            The created ExecutionTrace.

        Raises:
            ExecutionAlreadyStartedError: If execution_id already exists.
        """
        if execution_id in self._executions:
            raise ExecutionAlreadyStartedError(execution_id)

        trace = ExecutionTrace(
            id=execution_id,
            case_id=case_id,
            run_id=run_id,
            metadata=metadata or {},
            end_time=None,
            total_duration_ms=None,
            error=None,
        )
        self._executions[execution_id] = trace
        return trace

    def end_execution(
        self,
        execution_id: UUID,
        status: str = "success",
        error: str | None = None,
    ) -> ExecutionTrace:
        """End an execution trace.

        Args:
            execution_id: Execution to end.
            status: Final status (success, error, partial).
            error: Optional error message.

        Returns:
            The finalized ExecutionTrace.

        Raises:
            ExecutionNotFoundError: If execution_id doesn't exist.
        """
        trace = self._get_execution(execution_id)
        trace.finalize(status=status, error=error)
        return trace

    def record_event(self, event: TraceEvent) -> None:
        """Record a trace event to its execution.

        Args:
            event: The event to record.

        Raises:
            ExecutionNotFoundError: If event.execution_id doesn't exist.
            ExecutionCompletedError: If the execution is already finalized.
        """
        if event.execution_id is None:
            raise ValueError("Event must have an execution_id")

        trace = self._get_execution(event.execution_id)

        if trace.status != "in_progress":
            raise ExecutionCompletedError(event.execution_id)

        trace.add_event(event)

    def get_execution(self, execution_id: UUID) -> ExecutionTrace:
        """Get an execution trace by ID.

        Returns a copy to prevent accidental modification.

        Args:
            execution_id: Execution to retrieve.

        Returns:
            A copy of the ExecutionTrace.

        Raises:
            ExecutionNotFoundError: If execution_id doesn't exist.
        """
        trace = self._get_execution(execution_id)
        return self._copy_trace(trace)

    def get_all_executions(self) -> list[ExecutionTrace]:
        """Get all recorded executions.

        Returns copies to prevent accidental modification.

        Returns:
            List of ExecutionTrace copies.
        """
        return [self._copy_trace(trace) for trace in self._executions.values()]

    def get_events(self, execution_id: UUID) -> list[TraceEvent]:
        """Get all events for an execution in recorded order.

        Returns a copy to prevent accidental modification.

        Args:
            execution_id: Execution to retrieve events for.

        Returns:
            List of TraceEvent copies in recorded order.

        Raises:
            ExecutionNotFoundError: If execution_id doesn't exist.
        """
        trace = self._get_execution(execution_id)
        return list(trace.events)

    def has_execution(self, execution_id: UUID) -> bool:
        """Check if an execution exists."""
        return execution_id in self._executions

    def is_completed(self, execution_id: UUID) -> bool:
        """Check if an execution is completed."""
        if execution_id not in self._executions:
            return False
        return self._executions[execution_id].status != "in_progress"

    def _get_execution(self, execution_id: UUID) -> ExecutionTrace:
        """Get internal execution trace (not a copy)."""
        if execution_id not in self._executions:
            raise ExecutionNotFoundError(execution_id)
        return self._executions[execution_id]

    def _copy_trace(self, trace: ExecutionTrace) -> ExecutionTrace:
        """Create a deep copy of an ExecutionTrace."""
        import copy

        return copy.deepcopy(trace)
