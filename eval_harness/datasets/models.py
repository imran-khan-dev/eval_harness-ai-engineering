"""Dataset models for evaluation cases and datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExpectedToolCall(BaseModel):
    """Represents an expected tool call in an evaluation case.

    Attributes:
        name: Name of the tool/function to be called.
        arguments: Expected arguments for the tool call.
        description: Optional description of the tool call purpose.
        required: Whether this tool call is required (vs optional).
        order: Expected order index for sequencing validation.
    """

    name: str = Field(..., min_length=1, description="Tool/function name")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Expected tool arguments"
    )
    description: str | None = Field(None, description="Purpose of this tool call")
    required: bool = Field(default=True, description="Whether call is required")
    order: int | None = Field(None, ge=0, description="Expected order index")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class ExpectedTrajectory(BaseModel):
    """Represents an expected agent trajectory for evaluation.

    Attributes:
        steps: Sequence of expected steps (tool calls, observations, etc.).
        description: Optional description of the expected trajectory.
        max_deviation: Maximum allowed deviation from expected path.
    """

    steps: list[dict[str, Any]] = Field(
        default_factory=list, description="Expected trajectory steps"
    )
    description: str | None = Field(None, description="Trajectory description")
    max_deviation: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Max allowed deviation"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class AnswerabilityMetadata(BaseModel):
    """Metadata about whether a case is answerable and under what conditions.

    Attributes:
        is_answerable: Whether the case has a definitive answer.
        confidence: Confidence level in answerability (0.0 to 1.0).
        requires_tools: Whether answering requires tool usage.
        required_tools: List of tool names required to answer.
        context_requirements: Description of required context.
        ambiguity_notes: Notes about potential ambiguities.
    """

    is_answerable: bool = Field(default=True, description="Case has definitive answer")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Answerability confidence"
    )
    requires_tools: bool = Field(default=False, description="Requires tool usage")
    required_tools: list[str] = Field(
        default_factory=list, description="Required tool names"
    )
    context_requirements: str | None = Field(
        None, description="Required context description"
    )
    ambiguity_notes: str | None = Field(None, description="Ambiguity notes")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class EvaluationCase(BaseModel):
    """A single evaluation test case.

    Attributes:
        id: Unique identifier for the case.
        input: The input/query to evaluate.
        expected_output: Optional expected output/answer.
        expected_tool_calls: Optional list of expected tool calls.
        expected_trajectory: Optional expected agent trajectory.
        answerability: Optional answerability metadata.
        metadata: Arbitrary metadata for the case.
        tags: Tags for filtering and grouping cases.
        created_at: Timestamp when the case was created.
        updated_at: Timestamp when the case was last updated.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique case identifier")
    input: str = Field(..., min_length=1, description="Input query or prompt")
    expected_output: str | None = Field(None, description="Expected answer/output")
    expected_tool_calls: list[ExpectedToolCall] = Field(
        default_factory=list, description="Expected tool calls"
    )
    expected_trajectory: ExpectedTrajectory | None = Field(
        None, description="Expected agent trajectory"
    )
    answerability: AnswerabilityMetadata = Field(
        default_factory=lambda: AnswerabilityMetadata(
            is_answerable=True,
            confidence=1.0,
            requires_tools=False,
            required_tools=[],
            context_requirements=None,
            ambiguity_notes=None,
        ),
        description="Answerability metadata",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Case metadata")
    tags: list[str] = Field(default_factory=list, description="Filtering tags")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }


class EvaluationDataset(BaseModel):
    """A collection of evaluation cases.

    Attributes:
        id: Unique identifier for the dataset.
        name: Human-readable dataset name.
        description: Optional dataset description.
        cases: List of evaluation cases.
        version: Dataset version string.
        metadata: Arbitrary dataset metadata.
        tags: Tags for dataset categorization.
        created_at: Timestamp when dataset was created.
        updated_at: Timestamp when dataset was last updated.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique dataset identifier")
    name: str = Field(..., min_length=1, description="Dataset name")
    description: str | None = Field(None, description="Dataset description")
    cases: list[EvaluationCase] = Field(default_factory=list, description="Cases")
    version: str = Field(default="1.0.0", description="Dataset version")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    tags: list[str] = Field(default_factory=list, description="Dataset tags")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp"
    )

    def add_case(self, case: EvaluationCase) -> None:
        """Add a case to the dataset."""
        self.cases.append(case)
        self.updated_at = datetime.now(timezone.utc)

    def remove_case(self, case_id: UUID) -> bool:
        """Remove a case by ID. Returns True if removed."""
        for i, case in enumerate(self.cases):
            if case.id == case_id:
                self.cases.pop(i)
                self.updated_at = datetime.now(timezone.utc)
                return True
        return False

    def get_case(self, case_id: UUID) -> EvaluationCase | None:
        """Get a case by ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    def filter_by_tags(self, tags: list[str], match_all: bool = False) -> list[EvaluationCase]:
        """Filter cases by tags."""
        if match_all:
            return [c for c in self.cases if all(t in c.tags for t in tags)]
        return [c for c in self.cases if any(t in c.tags for t in tags)]

    def __len__(self) -> int:
        return len(self.cases)

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "use_enum_values": True,
    }
