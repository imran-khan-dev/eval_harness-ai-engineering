"""Base evaluator interface and protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from eval_harness.datasets.models import EvaluationCase
from eval_harness.metrics.models import MetricScore, MetricType


class Evaluator(Protocol):
    """Protocol for evaluators that produce MetricScore objects.

    Evaluators accept an evaluation case and an actual prediction/answer,
    and return a standardized MetricScore.
    """

    @property
    def metric_type(self) -> MetricType:
        """The type of metric this evaluator produces."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name of the metric."""
        ...

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate a prediction against an evaluation case.

        Args:
            case: The evaluation case containing expected output.
            prediction: The actual prediction/answer to evaluate.
            **kwargs: Additional evaluator-specific parameters.

        Returns:
            A MetricScore with the evaluation result.
        """
        ...


class BaseEvaluator(ABC):
    """Abstract base class for evaluators.

    Provides common functionality and enforces the evaluator interface.
    """

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """The type of metric this evaluator produces."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the metric."""

    @abstractmethod
    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate a prediction against an evaluation case.

        Args:
            case: The evaluation case containing expected output.
            prediction: The actual prediction/answer to evaluate.
            **kwargs: Additional evaluator-specific parameters.

        Returns:
            A MetricScore with the evaluation result.
        """

    def _create_score(
        self,
        value: float,
        threshold: float | None = None,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Create a standardized MetricScore.

        Args:
            value: The numeric score (0.0 to 1.0).
            threshold: Optional passing threshold.
            details: Additional metric-specific details.
            metadata: Arbitrary metadata.

        Returns:
            A MetricScore with the evaluator's metric_type and name.
        """
        return MetricScore(
            metric_type=self.metric_type,
            name=self.name,
            value=max(0.0, min(1.0, value)),
            threshold=threshold,
            passed=None,
            details=details or {},
            metadata=metadata or {},
        )
