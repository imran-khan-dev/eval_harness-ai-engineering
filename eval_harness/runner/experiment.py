"""Experiment runner for executing systems against datasets."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast
from uuid import UUID, uuid4

from eval_harness.datasets.models import EvaluationCase, EvaluationDataset
from eval_harness.evaluators.base import Evaluator
from eval_harness.metrics.aggregation import MetricAggregator
from eval_harness.metrics.models import (
    EvaluationResult,
    EvaluationStatus,
    ExperimentResult,
    MetricScore,
)
from eval_harness.runner.models import (
    RunnerExperimentResult,
    SystemExecutionResult,
    SystemExecutionStatus,
)
from eval_harness.runner.system import SystemUnderEvaluation


class ExperimentRunner:
    """Executes a SystemUnderEvaluation against an EvaluationDataset.

    The runner handles sequential execution of cases, failure isolation,
    and result aggregation. Both synchronous and asynchronous execution
    are supported depending on the system implementation.

    Evaluators can be configured to run after execution, producing
    per-case metric scores and experiment-level aggregates.
    """

    def __init__(
        self,
        system: SystemUnderEvaluation,
        *,
        evaluators: list[Evaluator] | None = None,
        run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the experiment runner.

        Args:
            system: The system to evaluate.
            evaluators: Optional list of evaluators to run after execution.
            run_id: Optional run/session identifier. Generated if not provided.
            metadata: Additional metadata to include in the experiment result.
        """
        self.system = system
        self.evaluators = evaluators or []
        self.run_id = run_id or uuid4()
        self.metadata = metadata or {}

    def run(
        self,
        dataset: EvaluationDataset,
        **execution_kwargs: Any,
    ) -> RunnerExperimentResult:
        """Execute the system against all cases in the dataset synchronously.

        Args:
            dataset: The evaluation dataset containing cases to execute.
            **execution_kwargs: Additional arguments passed to system.execute().

        Returns:
            RunnerExperimentResult with all execution results and aggregates.
        """
        experiment = RunnerExperimentResult(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            run_id=self.run_id,
            metadata=self.metadata,
            completed_at=None,
            error=None,
        )

        for case in dataset.cases:
            exec_result = self._execute_case(case, **execution_kwargs)
            experiment.add_execution_result(exec_result)

        experiment.finalize()
        return experiment

    def run_and_evaluate(
        self,
        dataset: EvaluationDataset,
        config_name: str = "runner_experiment",
        **execution_kwargs: Any,
    ) -> ExperimentResult:
        """Execute the system and run evaluators on the results.

        This method runs the system against all cases, then applies
        the configured evaluators to each successful execution. The
        result is a complete ExperimentResult with per-case metric scores
        and experiment-level aggregates.

        Args:
            dataset: The evaluation dataset containing cases to execute.
            config_name: Name to use for the evaluation config.
            **execution_kwargs: Additional arguments passed to system.execute().

        Returns:
            ExperimentResult with execution results, metric scores, and aggregates.
        """
        # First, execute all cases
        runner_result = self.run(dataset, **execution_kwargs)

        # Convert to metrics ExperimentResult and run evaluators
        return self._evaluate_results(runner_result, dataset.cases, config_name)

    async def run_async(
        self,
        dataset: EvaluationDataset,
        **execution_kwargs: Any,
    ) -> RunnerExperimentResult:
        """Execute the system against all cases asynchronously.

        This method should be used when the system implements an async execute method.
        If the system only has a synchronous execute method, it will be run in a thread pool.

        Args:
            dataset: The evaluation dataset containing cases to execute.
            **execution_kwargs: Additional arguments passed to system.execute().

        Returns:
            RunnerExperimentResult with all execution results and aggregates.
        """
        experiment = RunnerExperimentResult(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            run_id=self.run_id,
            metadata=self.metadata,
            completed_at=None,
            error=None,
        )

        for case in dataset.cases:
            exec_result = await self._execute_case_async(case, **execution_kwargs)
            experiment.add_execution_result(exec_result)

        experiment.finalize()
        return experiment

    async def run_and_evaluate_async(
        self,
        dataset: EvaluationDataset,
        config_name: str = "runner_experiment",
        **execution_kwargs: Any,
    ) -> ExperimentResult:
        """Execute the system asynchronously and run evaluators on the results.

        Args:
            dataset: The evaluation dataset containing cases to execute.
            config_name: Name to use for the evaluation config.
            **execution_kwargs: Additional arguments passed to system.execute().

        Returns:
            ExperimentResult with execution results, metric scores, and aggregates.
        """
        # First, execute all cases
        runner_result = await self.run_async(dataset, **execution_kwargs)

        # Convert to metrics ExperimentResult and run evaluators
        return self._evaluate_results(runner_result, dataset.cases, config_name)

    def _execute_case(
        self,
        case: EvaluationCase,
        **execution_kwargs: Any,
    ) -> SystemExecutionResult:
        """Execute a single case with error handling.

        Args:
            case: The evaluation case to execute.
            **execution_kwargs: Additional arguments passed to system.execute().

        Returns:
            SystemExecutionResult with the outcome.
        """
        # Prepare execution kwargs with run_id
        kwargs = {**execution_kwargs, "run_id": self.run_id}

        try:
            result = self.system.execute(case, **kwargs)
            # Ensure the result has the correct run_id
            if result.run_id is None:
                result.run_id = self.run_id
            return result
        except Exception as e:
            # Convert exceptions to ERROR execution results
            return SystemExecutionResult(
                case_id=case.id,
                run_id=self.run_id,
                status=SystemExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {e}",
                prediction=None,
                trace=None,
                latency_ms=None,
                token_usage=None,
                cost_estimate=None,
                completed_at=None,
                metadata={},
            )

    async def _execute_case_async(
        self,
        case: EvaluationCase,
        **execution_kwargs: Any,
    ) -> SystemExecutionResult:
        """Execute a single case asynchronously with error handling.

        Args:
            case: The evaluation case to execute.
            **execution_kwargs: Additional arguments passed to system.execute().

        Returns:
            SystemExecutionResult with the outcome.
        """
        # Prepare execution kwargs with run_id
        kwargs = {**execution_kwargs, "run_id": self.run_id}

        try:
            # Check if system has an async execute method
            execute_method = getattr(self.system, "execute", None)
            if execute_method is None:
                raise AttributeError("System has no execute method")

            # Check if it's a coroutine function or returns a coroutine
            if inspect.iscoroutinefunction(execute_method):
                result = await execute_method(case, **kwargs)
            else:
                # Run synchronous execute in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: execute_method(case, **kwargs)
                )
            result = cast(SystemExecutionResult, result)

            # Ensure the result has the correct run_id
            if result.run_id is None:
                result.run_id = self.run_id
            return result
        except Exception as e:
            # Convert exceptions to ERROR execution results
            return SystemExecutionResult(
                case_id=case.id,
                run_id=self.run_id,
                status=SystemExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {e}",
                prediction=None,
                trace=None,
                latency_ms=None,
                token_usage=None,
                cost_estimate=None,
                completed_at=None,
                metadata={},
            )

    def _evaluate_results(
        self,
        runner_result: RunnerExperimentResult,
        cases: list[EvaluationCase],
        config_name: str,
    ) -> ExperimentResult:
        """Run evaluators on execution results and produce ExperimentResult.

        Args:
            runner_result: The RunnerExperimentResult from execution phase.
            cases: The evaluation cases that were executed.
            config_name: Name to use for the evaluation config.

        Returns:
            Complete ExperimentResult with metric scores and aggregates.
        """
        exp_result = ExperimentResult(
            config_name=config_name,
            dataset_name=runner_result.dataset_name,
            total_tokens=None,
            total_cost=None,
            completed_at=runner_result.completed_at,
            error=runner_result.error,
        )
        exp_result.id = runner_result.id
        exp_result.started_at = runner_result.started_at
        exp_result.status = runner_result.status
        exp_result.metadata = runner_result.metadata

        # Create a mapping from case_id to execution result for easy lookup
        exec_results_by_case = {
            r.case_id: r for r in runner_result.execution_results
        }

        # Initialize metric aggregator for experiment-level aggregation
        metric_aggregator = MetricAggregator()

        # Evaluate each case
        for case in cases:
            exec_result = exec_results_by_case.get(case.id)
            if exec_result is None:
                # No execution result for this case - should not happen normally
                eval_result = EvaluationResult(
                    case_id=case.id,
                    run_id=self.run_id,
                    status=EvaluationStatus.ERROR,
                    error="No execution result found",
                    started_at=runner_result.started_at,
                    actual_output=None,
                    actual_trajectory=None,
                    trace=None,
                    latency_ms=None,
                    token_usage=None,
                    cost_estimate=None,
                    completed_at=None,
                )
                exp_result.add_case_result(eval_result)
                continue

            # Map system execution status to evaluation status
            if exec_result.is_successful():
                eval_status = EvaluationStatus.PASSED
            elif exec_result.status == SystemExecutionStatus.ERROR:
                eval_status = EvaluationStatus.ERROR
            elif exec_result.status == SystemExecutionStatus.TIMEOUT:
                eval_status = EvaluationStatus.FAILED
            elif exec_result.status == SystemExecutionStatus.PARTIAL:
                eval_status = EvaluationStatus.PARTIAL
            else:
                eval_status = EvaluationStatus.ERROR

            # Create the evaluation result
            eval_result = EvaluationResult(
                id=exec_result.id,
                case_id=case.id,
                run_id=exec_result.run_id or self.run_id,
                status=eval_status,
                actual_output=exec_result.prediction,
                actual_trajectory=None,
                trace=exec_result.trace,
                latency_ms=exec_result.latency_ms,
                token_usage=exec_result.token_usage,
                cost_estimate=exec_result.cost_estimate,
                error=exec_result.error,
                started_at=exec_result.started_at,
                completed_at=exec_result.completed_at,
                metadata=exec_result.metadata,
            )

            # Run evaluators if execution was successful and we have evaluators
            if self.evaluators and exec_result.is_successful():
                for evaluator in self.evaluators:
                    try:
                        # Prepare evaluator kwargs based on evaluator type
                        eval_kwargs = self._get_evaluator_kwargs(evaluator, case, exec_result)
                        score = evaluator.evaluate(case, exec_result.prediction, **eval_kwargs)
                        eval_result.add_metric_score(score)
                        metric_aggregator.add_score(score)
                    except Exception as e:
                        # Evaluator failed - record error but continue with other evaluators
                        error_score = MetricScore(
                            metric_type=evaluator.metric_type,
                            name=evaluator.name,
                            value=0.0,
                            threshold=None,
                            passed=False,
                            details={
                                "reason": "evaluator_exception",
                                "error": f"{type(e).__name__}: {e}",
                            },
                            metadata={"case_id": str(case.id)},
                        )
                        eval_result.add_metric_score(error_score)
                        metric_aggregator.add_score(error_score)
            elif self.evaluators and not exec_result.is_successful():
                # Execution failed - create SKIPPED scores for each evaluator
                for evaluator in self.evaluators:
                    skipped_score = MetricScore(
                        metric_type=evaluator.metric_type,
                        name=evaluator.name,
                        value=0.0,
                        threshold=None,
                        passed=None,
                        details={
                            "reason": "execution_failed",
                            "execution_status": exec_result.status,
                            "execution_error": exec_result.error,
                        },
                        metadata={"case_id": str(case.id)},
                    )
                    eval_result.add_metric_score(skipped_score)
                    metric_aggregator.add_score(skipped_score)

            # Finalize the evaluation result
            eval_result.finalize(status=eval_status)
            exp_result.add_case_result(eval_result)

        # Finalize experiment result - this computes aggregates from case results
        exp_result.finalize(status=runner_result.status, error=runner_result.error)

        return exp_result

    def _get_evaluator_kwargs(
        self,
        evaluator: Evaluator,
        case: EvaluationCase,
        exec_result: SystemExecutionResult,
    ) -> dict[str, Any]:
        """Get evaluator-specific kwargs based on evaluator type and available data.

        Args:
            evaluator: The evaluator to get kwargs for.
            case: The evaluation case.
            exec_result: The system execution result.

        Returns:
            Dictionary of kwargs to pass to the evaluator.
        """
        kwargs: dict[str, Any] = {}

        # Evaluators that need trace
        if hasattr(evaluator, 'evaluate'):
            import inspect
            sig = inspect.signature(evaluator.evaluate)
            params = sig.parameters

            if 'trace' in params and exec_result.trace is not None:
                kwargs['trace'] = exec_result.trace

            if 'expected_trajectory' in params and case.expected_trajectory is not None:
                kwargs['expected_trajectory'] = case.expected_trajectory

        return kwargs
