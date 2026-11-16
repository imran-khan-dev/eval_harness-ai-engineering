"""Configuration models for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run.

    Attributes:
        name: Human-readable name for the evaluation run.
        description: Optional description of the evaluation purpose.
        dataset_path: Path to the evaluation dataset file.
        output_dir: Directory for output artifacts (reports, traces, etc.).
        parallel_workers: Number of parallel workers for evaluation.
        timeout_seconds: Maximum time per evaluation case.
        retry_attempts: Number of retry attempts for failed cases.
        enable_tracing: Whether to record execution traces.
        trace_sampling_rate: Fraction of traces to record (0.0 to 1.0).
        cost_tracking_enabled: Whether to track token usage and estimated costs.
        metadata: Additional arbitrary metadata for the run.
    """

    name: str = Field(..., min_length=1, description="Name of the evaluation run")
    description: str | None = Field(None, description="Optional description")
    dataset_path: Path = Field(..., description="Path to evaluation dataset")
    output_dir: Path = Field(
        default_factory=lambda: Path("./eval_output"),
        description="Output directory for artifacts",
    )
    parallel_workers: int = Field(
        default=1, ge=1, le=64, description="Number of parallel workers"
    )
    timeout_seconds: float = Field(
        default=300.0, gt=0, description="Timeout per evaluation case in seconds"
    )
    retry_attempts: int = Field(
        default=0, ge=0, le=10, description="Number of retry attempts"
    )
    enable_tracing: bool = Field(
        default=True, description="Whether to record execution traces"
    )
    trace_sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate"
    )
    cost_tracking_enabled: bool = Field(
        default=True, description="Enable token and cost tracking"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional run metadata"
    )

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return v

    @field_validator("output_dir")
    @classmethod
    def ensure_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }
