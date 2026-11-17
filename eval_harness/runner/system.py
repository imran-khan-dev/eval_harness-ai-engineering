"""System Under Evaluation interface for provider-independent execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from eval_harness.datasets.models import EvaluationCase
from eval_harness.runner.models import SystemExecutionResult


class SystemUnderEvaluation(Protocol):
    """Protocol for systems that can be evaluated.

    This provider-agnostic interface allows the ExperimentRunner to execute
    evaluation cases against any AI system: a simple Python function, an LLM
    application, an agent system, or an orchestrator.

    Implementations must provide an `execute` method that accepts an
    EvaluationCase and returns a SystemExecutionResult containing the
    prediction, trace, and execution metadata.
    """

    def execute(
        self,
        case: EvaluationCase,
        **kwargs: Any,
    ) -> SystemExecutionResult:
        """Execute the system on a single evaluation case.

        Args:
            case: The evaluation case containing input, expected outputs,
                  and optional trajectory/tool call expectations.
            **kwargs: Optional execution context such as run_id, configuration,
                      or system-specific parameters.

        Returns:
            SystemExecutionResult with the system's prediction, execution trace,
            status, and any error information.
        """
        ...


class BaseSystem(ABC):
    """Abstract base class for SystemUnderEvaluation implementations.

    Provides a common foundation for concrete system implementations while
    enforcing the SystemUnderEvaluation protocol. Subclasses must implement
    the execute method.
    """

    @abstractmethod
    def execute(
        self,
        case: EvaluationCase,
        **kwargs: Any,
    ) -> SystemExecutionResult:
        """Execute the system on a single evaluation case.

        Args:
            case: The evaluation case containing input and expectations.
            **kwargs: Optional execution context and configuration.

        Returns:
            SystemExecutionResult with the execution outcome.
        """
        ...
