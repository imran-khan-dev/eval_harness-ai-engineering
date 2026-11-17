"""Tests for agent trajectory reconstruction."""

from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from eval_harness.tracing.models import ExecutionTrace, TraceEvent, TraceEventType
from eval_harness.tracing.trajectory import (
    TrajectoryNode,
    TrajectoryReconstructor,
    TrajectoryValidationResult,
    reconstruct_trajectory,
    validate_trajectory,
)


# Helper to create trace with required case_id
def make_trace(events: list[TraceEvent] | None = None) -> ExecutionTrace:
    """Create an ExecutionTrace with a dummy case_id."""
    return ExecutionTrace(case_id=uuid4(), events=events or [])


class TestTrajectoryNode:
    """Tests for TrajectoryNode dataclass."""

    def test_creation(self) -> None:
        """Test basic node creation."""
        event = TraceEvent(
            id=uuid4(),
            event_type=TraceEventType.LLM_CALL,
        )
        node = TrajectoryNode(event=event, children=(), depth=0)
        assert node.event == event
        assert node.children == ()
        assert node.depth == 0

    def test_with_children(self) -> None:
        """Test node with children."""
        parent_event = TraceEvent(
            id=uuid4(),
            event_type=TraceEventType.AGENT_ITERATION,
            data={"iteration": 1},
        )
        child_event = TraceEvent(
            id=uuid4(),
            event_type=TraceEventType.LLM_CALL,
            parent_id=parent_event.id,
        )
        child_node = TrajectoryNode(event=child_event, children=(), depth=1)
        parent_node = TrajectoryNode(
            event=parent_event, children=(child_node,), depth=0
        )
        assert len(parent_node.children) == 1
        assert parent_node.children[0].event == child_event
        assert parent_node.children[0].depth == 1

    def test_immutable(self) -> None:
        """Test that node is immutable (frozen)."""
        event = TraceEvent(id=uuid4(), event_type=TraceEventType.LLM_CALL)
        node = TrajectoryNode(event=event, children=(), depth=0)
        with pytest.raises(FrozenInstanceError):
            node.depth = 1  # type: ignore


class TestTrajectoryValidationResult:
    """Tests for TrajectoryValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test valid trajectory result."""
        result = TrajectoryValidationResult(
            is_valid=True,
            orphan_events=(),
            circular_references=(),
            root_events=(),
        )
        assert result.is_valid is True
        assert result.orphan_events == ()
        assert result.circular_references == ()
        assert result.root_events == ()

    def test_invalid_result(self) -> None:
        """Test invalid trajectory result."""
        event = TraceEvent(id=uuid4(), event_type=TraceEventType.LLM_CALL)
        result = TrajectoryValidationResult(
            is_valid=False,
            orphan_events=(event,),
            circular_references=((event.id, event.id),),
            root_events=(event,),
        )
        assert result.is_valid is False
        assert len(result.orphan_events) == 1
        assert len(result.circular_references) == 1
        assert len(result.root_events) == 1


class TestTrajectoryReconstructor:
    """Tests for TrajectoryReconstructor class."""

    def test_build_tree_empty(self) -> None:
        """Test building tree from empty trace."""
        trace = make_trace()
        roots = TrajectoryReconstructor.build_tree(trace)
        assert roots == ()

    def test_build_tree_single_root(self) -> None:
        """Test building tree with single root event."""
        event = TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START)
        trace = make_trace(events=[event])
        roots = TrajectoryReconstructor.build_tree(trace)
        assert len(roots) == 1
        assert roots[0].event == event
        assert roots[0].depth == 0
        assert roots[0].children == ()

    def test_build_tree_with_children(self) -> None:
        """Test building tree with parent-child relationships."""
        root_id = uuid4()
        child_id = uuid4()
        root = TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START)
        child = TraceEvent(
            id=child_id,
            event_type=TraceEventType.LLM_CALL,
            parent_id=root_id,
        )
        trace = make_trace(events=[root, child])
        roots = TrajectoryReconstructor.build_tree(trace)
        assert len(roots) == 1
        assert len(roots[0].children) == 1
        assert roots[0].children[0].event == child
        assert roots[0].children[0].depth == 1

    def test_build_tree_multiple_roots(self) -> None:
        """Test building tree with multiple root events."""
        root1 = TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START)
        root2 = TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START)
        trace = make_trace(events=[root1, root2])
        roots = TrajectoryReconstructor.build_tree(trace)
        assert len(roots) == 2

    def test_build_tree_complex(self) -> None:
        """Test building a complex tree."""
        root_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()
        grandchild_id = uuid4()

        root = TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START)
        child1 = TraceEvent(
            id=child1_id,
            event_type=TraceEventType.AGENT_ITERATION,
            parent_id=root_id,
            data={"iteration": 1},
        )
        child2 = TraceEvent(
            id=child2_id,
            event_type=TraceEventType.AGENT_ITERATION,
            parent_id=root_id,
            data={"iteration": 2},
        )
        grandchild = TraceEvent(
            id=grandchild_id,
            event_type=TraceEventType.LLM_CALL,
            parent_id=child1_id,
        )

        trace = make_trace(events=[root, child1, child2, grandchild])
        roots = TrajectoryReconstructor.build_tree(trace)

        assert len(roots) == 1
        assert len(roots[0].children) == 2
        child_nodes = roots[0].children
        assert any(c.event.id == child1_id for c in child_nodes)
        assert any(c.event.id == child2_id for c in child_nodes)

        # Find child1 and check its child
        child1_node = next(c for c in child_nodes if c.event.id == child1_id)
        assert len(child1_node.children) == 1
        assert child1_node.children[0].event.id == grandchild_id
        assert child1_node.children[0].depth == 2

    def test_get_events_by_iteration_found(self) -> None:
        """Test getting events by iteration when iteration exists."""
        iter1_id = uuid4()
        iter2_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(
                    id=iter1_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    data={"iteration": 1},
                ),
                TraceEvent(
                    id=child1_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=iter1_id,
                ),
                TraceEvent(
                    id=iter2_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    data={"iteration": 2},
                ),
                TraceEvent(
                    id=child2_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=iter2_id,
                ),
            ]
        )

        iter1_events = TrajectoryReconstructor.get_events_by_iteration(trace, 1)
        assert len(iter1_events) == 2
        assert iter1_events[0].id == iter1_id
        assert iter1_events[1].id == child1_id

        iter2_events = TrajectoryReconstructor.get_events_by_iteration(trace, 2)
        assert len(iter2_events) == 2
        assert iter2_events[0].id == iter2_id
        assert iter2_events[1].id == child2_id

    def test_get_events_by_iteration_not_found(self) -> None:
        """Test getting events by iteration when iteration doesn't exist."""
        trace = make_trace(
            events=[
                TraceEvent(
                    id=uuid4(),
                    event_type=TraceEventType.AGENT_ITERATION,
                    data={"iteration": 1},
                ),
            ]
        )
        events = TrajectoryReconstructor.get_events_by_iteration(trace, 999)
        assert events == []

    def test_get_events_by_iteration_nested(self) -> None:
        """Test getting events with nested descendants."""
        iter_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(
                    id=iter_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    data={"iteration": 1},
                ),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.TOOL_CALL,
                    parent_id=iter_id,
                ),
                TraceEvent(
                    id=grandchild_id,
                    event_type=TraceEventType.TOOL_RESPONSE,
                    parent_id=child_id,
                ),
            ]
        )

        events = TrajectoryReconstructor.get_events_by_iteration(trace, 1)
        assert len(events) == 3
        ids = {e.id for e in events}
        assert ids == {iter_id, child_id, grandchild_id}

    def test_validate_trajectory_valid(self) -> None:
        """Test validating a valid trajectory."""
        root_id = uuid4()
        child_id = uuid4()
        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=root_id,
                ),
            ]
        )

        result = TrajectoryReconstructor.validate_trajectory(trace)
        assert result.is_valid is True
        assert result.orphan_events == ()
        assert result.circular_references == ()
        assert len(result.root_events) == 1
        assert result.root_events[0].id == root_id

    def test_validate_trajectory_orphan(self) -> None:
        """Test validating trajectory with orphan event."""
        root_id = uuid4()
        orphan_id = uuid4()
        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=orphan_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=uuid4(),  # Non-existent parent
                ),
            ]
        )

        result = TrajectoryReconstructor.validate_trajectory(trace)
        assert result.is_valid is False
        assert len(result.orphan_events) == 1
        assert result.orphan_events[0].id == orphan_id
        # Orphan is treated as root
        assert len(result.root_events) == 2

    def test_validate_trajectory_cycle(self) -> None:
        """Test validating trajectory with cycle."""
        a_id = uuid4()
        b_id = uuid4()

        # Create events that form a cycle: A -> B -> A
        a = TraceEvent(
            id=a_id,
            event_type=TraceEventType.LLM_CALL,
            parent_id=b_id,
        )
        b = TraceEvent(
            id=b_id,
            event_type=TraceEventType.LLM_RESPONSE,
            parent_id=a_id,
        )
        trace = make_trace(events=[a, b])

        result = TrajectoryReconstructor.validate_trajectory(trace)
        assert result.is_valid is False
        assert len(result.circular_references) == 1

    def test_find_path_to_root(self) -> None:
        """Test finding path from event to root."""
        root_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    parent_id=root_id,
                    data={"iteration": 1},
                ),
                TraceEvent(
                    id=grandchild_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=child_id,
                ),
            ]
        )

        path = TrajectoryReconstructor.find_path_to_root(trace, grandchild_id)
        assert len(path) == 3
        assert path[0].id == grandchild_id
        assert path[1].id == child_id
        assert path[2].id == root_id

    def test_find_path_to_root_not_found(self) -> None:
        """Test finding path for non-existent event."""
        trace = make_trace(
            events=[
                TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START),
            ]
        )
        path = TrajectoryReconstructor.find_path_to_root(trace, uuid4())
        assert path == []

    def test_get_max_depth(self) -> None:
        """Test getting maximum depth."""
        root_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    parent_id=root_id,
                    data={"iteration": 1},
                ),
                TraceEvent(
                    id=grandchild_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=child_id,
                ),
            ]
        )
        assert TrajectoryReconstructor.get_max_depth(trace) == 2

    def test_get_max_depth_empty(self) -> None:
        """Test getting max depth for empty trace."""
        trace = make_trace()
        assert TrajectoryReconstructor.get_max_depth(trace) == -1

    def test_get_max_depth_single(self) -> None:
        """Test getting max depth for single event."""
        trace = make_trace(
            events=[TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START)]
        )
        assert TrajectoryReconstructor.get_max_depth(trace) == 0

    def test_get_siblings(self) -> None:
        """Test getting sibling events."""
        root_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child1_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=root_id,
                ),
                TraceEvent(
                    id=child2_id,
                    event_type=TraceEventType.LLM_RESPONSE,
                    parent_id=root_id,
                ),
            ]
        )

        siblings = TrajectoryReconstructor.get_siblings(trace, child1_id)
        assert len(siblings) == 1
        assert siblings[0].id == child2_id

    def test_get_siblings_no_parent(self) -> None:
        """Test getting siblings for root event (no parent)."""
        root_id = uuid4()
        trace = make_trace(
            events=[TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START)]
        )
        siblings = TrajectoryReconstructor.get_siblings(trace, root_id)
        assert siblings == []

    def test_get_siblings_not_found(self) -> None:
        """Test getting siblings for non-existent event."""
        trace = make_trace(
            events=[TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START)]
        )
        siblings = TrajectoryReconstructor.get_siblings(trace, uuid4())
        assert siblings == []

    def test_get_descendants(self) -> None:
        """Test getting all descendants."""
        root_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    parent_id=root_id,
                    data={"iteration": 1},
                ),
                TraceEvent(
                    id=grandchild_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=child_id,
                ),
            ]
        )

        descendants = TrajectoryReconstructor.get_descendants(trace, root_id)
        assert len(descendants) == 2
        ids = {d.id for d in descendants}
        assert ids == {child_id, grandchild_id}

    def test_get_descendants_leaf(self) -> None:
        """Test getting descendants of leaf node."""
        root_id = uuid4()
        child_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=root_id,
                ),
            ]
        )

        descendants = TrajectoryReconstructor.get_descendants(trace, child_id)
        assert descendants == []

    def test_pretty_print_tree(self) -> None:
        """Test pretty printing the tree."""
        root_id = uuid4()
        child_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=root_id,
                ),
            ]
        )

        output = TrajectoryReconstructor.pretty_print_tree(trace)
        assert "execution_start" in output
        assert "llm_call" in output
        assert str(root_id)[:8] in output
        assert str(child_id)[:8] in output

    def test_pretty_print_tree_max_depth(self) -> None:
        """Test pretty printing with max depth."""
        root_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    parent_id=root_id,
                    data={"iteration": 1},
                ),
                TraceEvent(
                    id=grandchild_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=child_id,
                ),
            ]
        )

        output = TrajectoryReconstructor.pretty_print_tree(trace, max_depth=1)
        assert "execution_start" in output
        assert "agent_iteration" in output
        assert "llm_call" not in output  # Depth 2 should be excluded


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_reconstruct_trajectory(self) -> None:
        """Test reconstruct_trajectory convenience function."""
        root_id = uuid4()
        child_id = uuid4()
        trace = make_trace(
            events=[
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=root_id,
                ),
            ]
        )

        roots = reconstruct_trajectory(trace)
        assert len(roots) == 1
        assert roots[0].event.id == root_id

    def test_validate_trajectory(self) -> None:
        """Test validate_trajectory convenience function."""
        trace = make_trace(
            events=[TraceEvent(id=uuid4(), event_type=TraceEventType.EXECUTION_START)]
        )

        result = validate_trajectory(trace)
        assert isinstance(result, TrajectoryValidationResult)
        assert result.is_valid is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_events_out_of_order(self) -> None:
        """Test tree building with events in random order."""
        root_id = uuid4()
        child_id = uuid4()
        grandchild_id = uuid4()

        # Add events in reverse order
        trace = make_trace(
            events=[
                TraceEvent(
                    id=grandchild_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=child_id,
                ),
                TraceEvent(
                    id=child_id,
                    event_type=TraceEventType.AGENT_ITERATION,
                    parent_id=root_id,
                    data={"iteration": 1},
                ),
                TraceEvent(id=root_id, event_type=TraceEventType.EXECUTION_START),
            ]
        )

        roots = TrajectoryReconstructor.build_tree(trace)
        assert len(roots) == 1
        assert len(roots[0].children) == 1
        assert len(roots[0].children[0].children) == 1

    def test_deep_nesting(self) -> None:
        """Test deeply nested events."""
        event_ids = [uuid4() for _ in range(10)]
        events = [
            TraceEvent(id=event_ids[0], event_type=TraceEventType.EXECUTION_START)
        ]
        for i in range(1, 10):
            events.append(
                TraceEvent(
                    id=event_ids[i],
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=event_ids[i - 1],
                )
            )

        trace = make_trace(events=events)
        roots = TrajectoryReconstructor.build_tree(trace)
        assert len(roots) == 1
        assert TrajectoryReconstructor.get_max_depth(trace) == 9

        # Check depth of leaf
        node = roots[0]
        for _ in range(9):
            assert len(node.children) == 1
            node = node.children[0]
        assert node.depth == 9

    def test_multiple_orphans_validated(self) -> None:
        """Test validation with multiple orphans."""
        trace = make_trace(
            events=[
                TraceEvent(
                    id=uuid4(),
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=uuid4(),
                ),
                TraceEvent(
                    id=uuid4(),
                    event_type=TraceEventType.LLM_RESPONSE,
                    parent_id=uuid4(),
                ),
            ]
        )

        result = TrajectoryReconstructor.validate_trajectory(trace)
        assert result.is_valid is False
        assert len(result.orphan_events) == 2
        assert len(result.root_events) == 2

    def test_mixed_valid_and_invalid(self) -> None:
        """Test validation with mix of valid and invalid events."""
        valid_root_id = uuid4()
        valid_child_id = uuid4()
        orphan_id = uuid4()
        cycle_a_id = uuid4()
        cycle_b_id = uuid4()

        trace = make_trace(
            events=[
                TraceEvent(
                    id=valid_root_id,
                    event_type=TraceEventType.EXECUTION_START,
                ),
                TraceEvent(
                    id=valid_child_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=valid_root_id,
                ),
                TraceEvent(
                    id=orphan_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=uuid4(),
                ),
                TraceEvent(
                    id=cycle_a_id,
                    event_type=TraceEventType.LLM_CALL,
                    parent_id=cycle_b_id,
                ),
                TraceEvent(
                    id=cycle_b_id,
                    event_type=TraceEventType.LLM_RESPONSE,
                    parent_id=cycle_a_id,
                ),
            ]
        )

        result = TrajectoryReconstructor.validate_trajectory(trace)
        assert result.is_valid is False
        assert len(result.orphan_events) == 1
        assert len(result.circular_references) == 1
