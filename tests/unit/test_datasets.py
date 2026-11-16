"""Tests for dataset models."""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from eval_harness.datasets.models import (
    AnswerabilityMetadata,
    EvaluationCase,
    EvaluationDataset,
    ExpectedToolCall,
    ExpectedTrajectory,
)


class TestExpectedToolCall:
    """Tests for ExpectedToolCall model."""

    def test_minimal_tool_call(self) -> None:
        tool = ExpectedToolCall(name="search")
        assert tool.name == "search"
        assert tool.arguments == {}
        assert tool.required is True
        assert tool.order is None

    def test_full_tool_call(self) -> None:
        tool = ExpectedToolCall(
            name="calculate",
            arguments={"x": 1, "y": 2},
            description="Add numbers",
            required=False,
            order=1,
        )
        assert tool.name == "calculate"
        assert tool.arguments == {"x": 1, "y": 2}
        assert tool.description == "Add numbers"
        assert tool.required is False
        assert tool.order == 1

    def test_invalid_name(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedToolCall(name="")

    def test_invalid_order(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedToolCall(name="test", order=-1)


class TestExpectedTrajectory:
    """Tests for ExpectedTrajectory model."""

    def test_empty_trajectory(self) -> None:
        traj = ExpectedTrajectory()
        assert traj.steps == []
        assert traj.max_deviation == 0.0

    def test_with_steps(self) -> None:
        traj = ExpectedTrajectory(
            steps=[{"action": "search"}, {"action": "read"}],
            description="Search then read",
            max_deviation=0.2,
        )
        assert len(traj.steps) == 2
        assert traj.description == "Search then read"
        assert traj.max_deviation == 0.2

    def test_invalid_deviation(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedTrajectory(max_deviation=-0.1)

        with pytest.raises(ValidationError):
            ExpectedTrajectory(max_deviation=1.1)


class TestAnswerabilityMetadata:
    """Tests for AnswerabilityMetadata model."""

    def test_defaults(self) -> None:
        meta = AnswerabilityMetadata()
        assert meta.is_answerable is True
        assert meta.confidence == 1.0
        assert meta.requires_tools is False
        assert meta.required_tools == []

    def test_custom_values(self) -> None:
        meta = AnswerabilityMetadata(
            is_answerable=False,
            confidence=0.5,
            requires_tools=True,
            required_tools=["search", "read"],
            context_requirements="Needs web access",
            ambiguity_notes="Query is vague",
        )
        assert meta.is_answerable is False
        assert meta.confidence == 0.5
        assert meta.requires_tools is True
        assert meta.required_tools == ["search", "read"]
        assert meta.context_requirements == "Needs web access"
        assert meta.ambiguity_notes == "Query is vague"

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValidationError):
            AnswerabilityMetadata(confidence=-0.1)

        with pytest.raises(ValidationError):
            AnswerabilityMetadata(confidence=1.1)


class TestEvaluationCase:
    """Tests for EvaluationCase model."""

    def test_minimal_case(self) -> None:
        case = EvaluationCase(input="What is 2+2?")
        assert case.input == "What is 2+2?"
        assert case.expected_output is None
        assert case.expected_tool_calls == []
        assert case.expected_trajectory is None
        assert isinstance(case.id, UUID)
        assert isinstance(case.created_at, datetime)
        assert isinstance(case.updated_at, datetime)
        assert case.tags == []
        assert case.metadata == {}

    def test_full_case(self) -> None:
        case = EvaluationCase(
            input="Search for Python docs",
            expected_output="Python.org",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "Python"})
            ],
            expected_trajectory=ExpectedTrajectory(steps=[{"action": "search"}]),
            answerability=AnswerabilityMetadata(requires_tools=True),
            metadata={"source": "test"},
            tags=["search", "web"],
        )
        assert case.input == "Search for Python docs"
        assert case.expected_output == "Python.org"
        assert len(case.expected_tool_calls) == 1
        assert case.expected_trajectory is not None
        assert case.answerability.requires_tools is True
        assert case.metadata == {"source": "test"}
        assert case.tags == ["search", "web"]

    def test_invalid_input(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationCase(input="")

    def test_updated_at_on_modification(self) -> None:
        case = EvaluationCase(input="test")
        original_updated = case.updated_at
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)
        # Create a new case with modified input (simulating update)
        case2 = EvaluationCase(input="modified", id=case.id)
        assert case2.updated_at >= original_updated


class TestEvaluationDataset:
    """Tests for EvaluationDataset model."""

    def test_empty_dataset(self) -> None:
        dataset = EvaluationDataset(name="empty")
        assert dataset.name == "empty"
        assert len(dataset) == 0
        assert dataset.cases == []

    def test_dataset_with_cases(self) -> None:
        cases = [
            EvaluationCase(input="Case 1"),
            EvaluationCase(input="Case 2"),
        ]
        dataset = EvaluationDataset(name="test", cases=cases)
        assert len(dataset) == 2
        assert dataset.cases[0].input == "Case 1"
        assert dataset.cases[1].input == "Case 2"

    def test_add_case(self) -> None:
        dataset = EvaluationDataset(name="test")
        case = EvaluationCase(input="New case")
        dataset.add_case(case)
        assert len(dataset) == 1
        assert dataset.cases[0] is case

    def test_remove_case(self) -> None:
        case1 = EvaluationCase(input="Case 1")
        case2 = EvaluationCase(input="Case 2")
        dataset = EvaluationDataset(name="test", cases=[case1, case2])

        removed = dataset.remove_case(case1.id)
        assert removed is True
        assert len(dataset) == 1
        assert dataset.cases[0].input == "Case 2"

        # Remove non-existent
        removed = dataset.remove_case(uuid4())
        assert removed is False
        assert len(dataset) == 1

    def test_get_case(self) -> None:
        case = EvaluationCase(input="Find me")
        dataset = EvaluationDataset(name="test", cases=[case])

        found = dataset.get_case(case.id)
        assert found is case

        not_found = dataset.get_case(uuid4())
        assert not_found is None

    def test_filter_by_tags(self) -> None:
        case1 = EvaluationCase(input="C1", tags=["a", "b"])
        case2 = EvaluationCase(input="C2", tags=["b", "c"])
        case3 = EvaluationCase(input="C3", tags=["a"])
        dataset = EvaluationDataset(name="test", cases=[case1, case2, case3])

        # Match any
        filtered = dataset.filter_by_tags(["a"])
        assert len(filtered) == 2
        assert all("a" in c.tags for c in filtered)

        # Match all
        filtered = dataset.filter_by_tags(["a", "b"], match_all=True)
        assert len(filtered) == 1
        assert filtered[0].input == "C1"

    def test_iteration(self) -> None:
        cases = [EvaluationCase(input=f"Case {i}") for i in range(3)]
        dataset = EvaluationDataset(name="test", cases=cases)
        assert dataset.cases == cases

    def test_version_and_metadata(self) -> None:
        dataset = EvaluationDataset(
            name="versioned",
            version="2.0.0",
            metadata={"author": "test"},
            tags=["regression"],
        )
        assert dataset.version == "2.0.0"
        assert dataset.metadata == {"author": "test"}
        assert dataset.tags == ["regression"]
