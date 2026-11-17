"""Agent trajectory reconstruction from trace events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from eval_harness.tracing.models import ExecutionTrace, TraceEvent, TraceEventType


@dataclass(frozen=True)
class TrajectoryNode:
    """A node in the reconstructed event tree.

    Attributes:
        event: The trace event this node represents.
        children: Child nodes in the tree.
        depth: Depth in the tree (0 for root).
    """

    event: TraceEvent
    children: tuple[TrajectoryNode, ...]
    depth: int


@dataclass(frozen=True)
class TrajectoryValidationResult:
    """Result of trajectory validation.

    Attributes:
        is_valid: Whether the trajectory is valid.
        orphan_events: Events with non-existent parent IDs.
        circular_references: List of event IDs involved in cycles.
        root_events: Root events (including orphans treated as roots).
    """

    is_valid: bool
    orphan_events: tuple[TraceEvent, ...]
    circular_references: tuple[tuple[UUID, ...], ...]
    root_events: tuple[TraceEvent, ...]


class TrajectoryReconstructor:
    """Reconstructs agent execution trajectories from trace events."""

    @staticmethod
    def build_tree(trace: ExecutionTrace) -> tuple[TrajectoryNode, ...]:
        """Build a tree of events from an execution trace.

        Args:
            trace: The execution trace containing events.

        Returns:
            Tuple of root TrajectoryNodes.
        """
        children_map: dict[UUID, list[TraceEvent]] = defaultdict(list)
        for event in trace.events:
            if event.parent_id:
                children_map[event.parent_id].append(event)

        root_events = [e for e in trace.events if e.parent_id is None]

        def build_node(event: TraceEvent, depth: int) -> TrajectoryNode:
            children = tuple(
                build_node(child, depth + 1)
                for child in children_map.get(event.id, [])
            )
            return TrajectoryNode(event=event, children=children, depth=depth)

        return tuple(build_node(root, 0) for root in root_events)

    @staticmethod
    def get_events_by_iteration(
        trace: ExecutionTrace, iteration: int
    ) -> list[TraceEvent]:
        """Get all events belonging to a specific agent iteration.

        Args:
            trace: The execution trace.
            iteration: The iteration number to filter by.

        Returns:
            List of events in that iteration (including the iteration event).
        """
        children_map: dict[UUID, list[TraceEvent]] = defaultdict(list)
        for event in trace.events:
            if event.parent_id:
                children_map[event.parent_id].append(event)

        iteration_root = None
        for event in trace.events:
            if (
                event.event_type == TraceEventType.AGENT_ITERATION
                and event.data.get("iteration") == iteration
            ):
                iteration_root = event
                break

        if iteration_root is None:
            return []

        result = [iteration_root]

        def collect_descendants(event_id: UUID) -> None:
            for child in children_map.get(event_id, []):
                result.append(child)
                collect_descendants(child.id)

        collect_descendants(iteration_root.id)
        return result

    @staticmethod
    def validate_trajectory(trace: ExecutionTrace) -> TrajectoryValidationResult:
        """Validate the trajectory for orphans and cycles.

        Args:
            trace: The execution trace to validate.

        Returns:
            TrajectoryValidationResult with validation details.
        """
        event_map = {event.id: event for event in trace.events}

        orphan_events = [
            event
            for event in trace.events
            if event.parent_id is not None and event.parent_id not in event_map
        ]

        circular_references = TrajectoryReconstructor._detect_cycles(trace.events)

        root_events = [
            event
            for event in trace.events
            if event.parent_id is None or event.parent_id not in event_map
        ]

        is_valid = len(orphan_events) == 0 and len(circular_references) == 0

        return TrajectoryValidationResult(
            is_valid=is_valid,
            orphan_events=tuple(orphan_events),
            circular_references=tuple(circular_references),
            root_events=tuple(root_events),
        )

    @staticmethod
    def _detect_cycles(events: list[TraceEvent]) -> list[tuple[UUID, ...]]:
        """Detect cycles in the event graph.

        Args:
            events: List of trace events.

        Returns:
            List of cycles, each represented as a tuple of event IDs.
        """
        event_map = {event.id: event for event in events}
        visited: set[UUID] = set()
        rec_stack: set[UUID] = set()
        cycles: list[tuple[UUID, ...]] = []
        path: list[UUID] = []

        def dfs(event_id: UUID) -> None:
            visited.add(event_id)
            rec_stack.add(event_id)
            path.append(event_id)

            event = event_map.get(event_id)
            if event and event.parent_id:
                parent_id = event.parent_id
                if parent_id not in visited:
                    dfs(parent_id)
                elif parent_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(parent_id)
                    cycles.append(tuple(path[cycle_start:]))

            rec_stack.remove(event_id)
            path.pop()

        for event in events:
            if event.id not in visited:
                dfs(event.id)

        return cycles

    @staticmethod
    def find_path_to_root(trace: ExecutionTrace, event_id: UUID) -> list[TraceEvent]:
        """Find the path from an event to its root.

        Args:
            trace: The execution trace.
            event_id: The event ID to start from.

        Returns:
            List of events from the event to root (inclusive), or empty if not found.
        """
        event_map = {event.id: event for event in trace.events}
        event = event_map.get(event_id)
        if event is None:
            return []

        path = [event]
        current: TraceEvent | None = event
        while current is not None and current.parent_id is not None:
            current = event_map.get(current.parent_id)
            if current is None:
                break
            path.append(current)

        return path

    @staticmethod
    def get_max_depth(trace: ExecutionTrace) -> int:
        """Get the maximum depth of the event tree.

        Args:
            trace: The execution trace.

        Returns:
            Maximum depth (0 for single event, -1 for empty).
        """
        roots = [e for e in trace.events if e.parent_id is None]
        if not roots:
            return -1

        children_map: dict[UUID, list[TraceEvent]] = defaultdict(list)
        for event in trace.events:
            if event.parent_id:
                children_map[event.parent_id].append(event)

        def get_depth(event_id: UUID) -> int:
            children = children_map.get(event_id, [])
            if not children:
                return 0
            return 1 + max(get_depth(child.id) for child in children)

        return max(get_depth(root.id) for root in roots)

    @staticmethod
    def get_siblings(trace: ExecutionTrace, event_id: UUID) -> list[TraceEvent]:
        """Get sibling events of a given event.

        Args:
            trace: The execution trace.
            event_id: The event ID.

        Returns:
            List of sibling events (same parent, excluding self).
        """
        event_map = {event.id: event for event in trace.events}
        event = event_map.get(event_id)
        if event is None or event.parent_id is None:
            return []

        siblings = [
            e
            for e in trace.events
            if e.parent_id == event.parent_id and e.id != event_id
        ]
        return siblings

    @staticmethod
    def get_descendants(trace: ExecutionTrace, event_id: UUID) -> list[TraceEvent]:
        """Get all descendant events of a given event.

        Args:
            trace: The execution trace.
            event_id: The event ID.

        Returns:
            List of all descendant events in recording order.
        """
        children_map: dict[UUID, list[TraceEvent]] = defaultdict(list)
        for event in trace.events:
            if event.parent_id:
                children_map[event.parent_id].append(event)

        result: list[TraceEvent] = []

        def collect(event_id: UUID) -> None:
            for child in children_map.get(event_id, []):
                result.append(child)
                collect(child.id)

        collect(event_id)
        return result

    @staticmethod
    def pretty_print_tree(
        trace: ExecutionTrace, max_depth: int | None = None
    ) -> str:
        """Pretty print the event tree.

        Args:
            trace: The execution trace.
            max_depth: Maximum depth to print (None for unlimited).

        Returns:
            Formatted string representation of the tree.
        """
        roots = [e for e in trace.events if e.parent_id is None]

        children_map: dict[UUID, list[TraceEvent]] = defaultdict(list)
        for event in trace.events:
            if event.parent_id:
                children_map[event.parent_id].append(event)

        def print_node(node: TraceEvent, prefix: str, is_last: bool, depth: int) -> str:
            if max_depth is not None and depth > max_depth:
                return ""

            lines = []
            connector = "└── " if is_last else "├── "
            event = node
            et = event.event_type
            event_type_str = et.value if hasattr(et, "value") else str(et)
            event_info = f"{event_type_str} ({str(event.id)[:8]})"
            if event.data:
                key_data = []
                for k, v in event.data.items():
                    if isinstance(v, str) and len(v) > 50:
                        v = v[:47] + "..."
                    key_data.append(f"{k}={v}")
                if key_data:
                    event_info += f" [{', '.join(key_data)}]"
            lines.append(f"{prefix}{connector}{event_info}")

            children = children_map.get(event.id, [])
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                lines.append(
                    print_node(child, child_prefix, i == len(children) - 1, depth + 1)
                )

            return "\n".join(filter(None, lines))

        if not roots:
            return "(empty trace)"

        output_lines = []
        for i, root in enumerate(roots):
            output_lines.append(print_node(root, "", i == len(roots) - 1, 0))

        return "\n".join(filter(None, output_lines))


def reconstruct_trajectory(trace: ExecutionTrace) -> tuple[TrajectoryNode, ...]:
    """Convenience function to reconstruct trajectory from trace.

    Args:
        trace: The execution trace.

    Returns:
        Tuple of root TrajectoryNodes.
    """
    return TrajectoryReconstructor.build_tree(trace)


def validate_trajectory(trace: ExecutionTrace) -> TrajectoryValidationResult:
    """Convenience function to validate trajectory.

    Args:
        trace: The execution trace.

    Returns:
        TrajectoryValidationResult with validation details.
    """
    return TrajectoryReconstructor.validate_trajectory(trace)
