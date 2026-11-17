"""Comprehensive integration tests for the ExperimentRunner workflow.

These tests cover the complete evaluation pipeline:
EvaluationDataset → SystemUnderEvaluation → ExperimentRunner → SystemExecutionResult
→ Evaluators → EvaluationResult → MetricAggregator → ExperimentResult
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from eval_harness.datasets.models import (
    EvaluationCase,
    EvaluationDataset,
    ExpectedToolCall,
    ExpectedTrajectory,
)
from eval_harness.evaluators import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    NormalizedExactMatchEvaluator,
    StringSimilarityEvaluator,
    ToolExecutionEvaluator,
    ToolSelectionEvaluator,
    TrajectoryCompletenessEvaluator,
    TrajectoryMatchEvaluator,
    TrajectoryOrderEvaluator,
)
from eval_harness.metrics.aggregation import AggregationResult, MetricAggregator
from eval_harness.metrics.models import (
    EvaluationResult,
    EvaluationStatus,
    ExperimentResult,
    MetricType,
)
from eval_harness.runner import (
    BaseSystem,
    ExperimentRunner,
    RunnerExperimentResult,
    SystemExecutionResult,
    SystemExecutionStatus,
)
from eval_harness.tracing.models import (
    CostEstimate,
    ExecutionTrace,
    LatencyMetrics,
    TokenUsage,
    TraceEvent,
)

# =============================================================================
# Test Systems and Helpers
# =============================================================================

class SuccessfulSyncSystem(BaseSystem):
    """System that always succeeds with a prediction."""

    def __init__(self, prediction: str = "Default output"):
        self._prediction = prediction

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.execution_start(case.id, case.id))
        trace.add_event(
            TraceEvent.llm_call(case.id, case.input, "test-model")
        )
        trace.add_event(
            TraceEvent.llm_response(
                case.id,
                self._prediction,
                "test-model",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                latency=LatencyMetrics(
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                ),
            )
        )
        trace.add_event(TraceEvent.execution_end(case.id, "success"))
        trace.finalize(status="success")
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction=self._prediction,
            trace=trace,
        )


class SuccessfulAsyncSystem(BaseSystem):
    """Async system that always succeeds."""

    def __init__(self, prediction: str = "Async output", delay: float = 0.0):
        self._prediction = prediction
        self._delay = delay

    async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.execution_start(case.id, case.id))
        trace.add_event(
            TraceEvent.llm_call(case.id, case.input, "test-model")
        )
        trace.add_event(
            TraceEvent.llm_response(
                case.id,
                self._prediction,
                "test-model",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            )
        )
        trace.add_event(TraceEvent.execution_end(case.id, "success"))
        trace.finalize(status="success")
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction=self._prediction,
            trace=trace,
        )


class FailingSyncSystem(BaseSystem):
    """System that raises an exception during execution."""

    def __init__(self, error_message: str = "Execution failed"):
        self._error = error_message

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        raise RuntimeError(self._error)


class ErrorResultSystem(BaseSystem):
    """System that returns an ERROR status result."""

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.ERROR,
            error="System returned error",
        )


class TimeoutSystem(BaseSystem):
    """System that returns a TIMEOUT status result."""

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.TIMEOUT,
            error="Timeout after 30s",
        )


class PartialSystem(BaseSystem):
    """System that returns a PARTIAL status result."""

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.execution_start(case.id, case.id))
        trace.finalize(status="partial")
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.PARTIAL,
            prediction="Partial output",
            trace=trace,
        )


class MixedSystem(BaseSystem):
    """System that succeeds for some cases and fails for others."""

    def __init__(self):
        self.call_count = 0

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        self.call_count += 1
        if self.call_count == 2:
            raise RuntimeError("Case 2 failed")
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.execution_start(case.id, case.id))
        trace.add_event(TraceEvent.execution_end(case.id, "success"))
        trace.finalize(status="success")
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Success output",
            trace=trace,
        )


class TracingSystem(BaseSystem):
    """System that returns a detailed trace with tool calls."""

    def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
        trace = ExecutionTrace(case_id=case.id)
        trace.add_event(TraceEvent.execution_start(case.id, case.id))

        # LLM call
        trace.add_event(
            TraceEvent.llm_call(
                case.id, case.input, "test-model",
                latency=LatencyMetrics(
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                ),
            )
        )
        trace.add_event(
            TraceEvent.llm_response(
                case.id, "LLM response", "test-model",
                token_usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
                cost=CostEstimate(
                    input_cost_usd=0.001,
                    output_cost_usd=0.002,
                    model_name="test-model",
                ),
            )
        )

        # Tool call
        tool_corr = uuid4()
        trace.add_event(
            TraceEvent.tool_call(
                case.id, "search", {"query": "test"},
                correlation_id=tool_corr,
            )
        )
        trace.add_event(
            TraceEvent.tool_response(
                case.id, "search", {"results": ["result1"]},
                correlation_id=tool_corr,
            )
        )

        trace.add_event(TraceEvent.execution_end(case.id, "success"))
        trace.finalize(status="success")

        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Final output with tools",
            trace=trace,
        )


class FailingEvaluator(ExactMatchEvaluator):
    """Evaluator that always raises an exception."""

    def evaluate(self, case, prediction, **kwargs):
        raise ValueError("Evaluator failed intentionally")


# =============================================================================
# Scenario A: Successful evaluation
# =============================================================================

class TestScenarioASuccessfulEvaluation:
    """Test complete successful evaluation workflow."""

    def setup_method(self):
        self.case1 = EvaluationCase(input="Test 1", expected_output="Out 1")
        self.case2 = EvaluationCase(input="Test 2", expected_output="Out 2")
        self.case3 = EvaluationCase(input="Test 3", expected_output="Out 3")
        self.dataset = EvaluationDataset(
            name="success_test", cases=[self.case1, self.case2, self.case3]
        )

    def test_successful_evaluation_with_multiple_evaluators(self):
        """Test successful evaluation with multiple evaluators per case."""
        system = SuccessfulSyncSystem(prediction="Out 1")
        evaluators = [
            ExactMatchEvaluator(),
            ContainsEvaluator(),
            NormalizedExactMatchEvaluator(),
            StringSimilarityEvaluator(),
        ]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        # Verify experiment structure
        assert isinstance(result, ExperimentResult)
        assert result.config_name == "runner_experiment"
        assert result.dataset_name == "success_test"
        assert len(result.case_results) == 3

        # Verify each case has metric scores
        for eval_result in result.case_results:
            assert eval_result.status == EvaluationStatus.PASSED
            assert len(eval_result.metric_scores) == 4

            # Check metric types
            metric_names = [s.name for s in eval_result.metric_scores]
            assert "exact_match" in metric_names
            assert "contains" in metric_names
            assert "normalized_exact_match" in metric_names
            assert "string_similarity" in metric_names

        # Verify experiment-level aggregation
        assert result.aggregated_metrics.total_cases == 3
        assert result.aggregated_metrics.passed_cases == 3
        assert result.aggregated_metrics.failed_cases == 0
        assert result.aggregated_metrics.error_cases == 0
        assert len(result.aggregated_metrics.metric_scores) == 12  # 3 cases * 4 evaluators

        # Verify aggregated stats
        assert result.aggregated_metrics.mean_scores.get(MetricType.EXACT_MATCH) is not None

    def test_successful_evaluation_async(self):
        """Test successful async evaluation."""
        async def run_test():
            system = SuccessfulAsyncSystem(prediction="Async Out", delay=0.01)
            evaluators = [ExactMatchEvaluator(), ContainsEvaluator()]
            runner = ExperimentRunner(system=system, evaluators=evaluators)
            result = await runner.run_and_evaluate_async(self.dataset)

            assert len(result.case_results) == 3
            for eval_result in result.case_results:
                assert eval_result.status == EvaluationStatus.PASSED
                assert len(eval_result.metric_scores) == 2

        asyncio.run(run_test())


# =============================================================================
# Scenario B: Mixed execution outcomes
# =============================================================================

class TestScenarioBMixedOutcomes:
    """Test mixed execution outcomes (success, error, timeout, partial)."""

    def setup_method(self):
        self.cases = [
            EvaluationCase(input="Test 1", expected_output="Out 1"),
            EvaluationCase(input="Test 2", expected_output="Out 2"),
            EvaluationCase(input="Test 3", expected_output="Out 3"),
            EvaluationCase(input="Test 4", expected_output="Out 4"),
        ]
        self.dataset = EvaluationDataset(name="mixed_test", cases=self.cases)

    def test_mixed_execution_outcomes(self):
        """Test that all case types are handled and remaining cases continue."""
        # Case 1: SUCCESS, Case 2: ERROR (exception), Case 3: TIMEOUT, Case 4: PARTIAL
        class MixedOutcomeSystem(BaseSystem):
            def __init__(self):
                self.call_count = 0

            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                if self.call_count == 1:
                    trace = ExecutionTrace(case_id=case.id)
                    trace.add_event(TraceEvent.execution_start(case.id, case.id))
                    trace.add_event(TraceEvent.execution_end(case.id, "success"))
                    trace.finalize(status="success")
                    return SystemExecutionResult(
                        case_id=case.id,
                        status=SystemExecutionStatus.SUCCESS,
                        prediction="Success",
                        trace=trace,
                    )
                elif self.call_count == 2:
                    raise RuntimeError("Case 2 exception")
                elif self.call_count == 3:
                    return SystemExecutionResult(
                        case_id=case.id,
                        status=SystemExecutionStatus.TIMEOUT,
                        error="Timeout",
                    )
                else:  # call_count == 4
                    trace = ExecutionTrace(case_id=case.id)
                    trace.add_event(TraceEvent.execution_start(case.id, case.id))
                    trace.finalize(status="partial")
                    return SystemExecutionResult(
                        case_id=case.id,
                        status=SystemExecutionStatus.PARTIAL,
                        prediction="Partial",
                        trace=trace,
                    )

        system = MixedOutcomeSystem()
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        # All 4 cases should have results
        assert len(result.case_results) == 4
        assert result.aggregated_metrics.total_cases == 4

        # Check statuses
        statuses = [r.status for r in result.case_results]
        assert EvaluationStatus.PASSED in statuses  # Case 1
        assert EvaluationStatus.ERROR in statuses   # Case 2
        assert EvaluationStatus.FAILED in statuses  # Case 3 (TIMEOUT -> FAILED)
        assert EvaluationStatus.PARTIAL in statuses  # Case 4

        # Case 1: should have metric scores
        passed_result = next(r for r in result.case_results if r.status == EvaluationStatus.PASSED)
        assert len(passed_result.metric_scores) == 1

        # Case 2 (ERROR): should have SKIPPED scores
        error_result = next(r for r in result.case_results if r.status == EvaluationStatus.ERROR)
        assert len(error_result.metric_scores) == 1
        assert error_result.metric_scores[0].details.get("reason") == "execution_failed"
        assert error_result.metric_scores[0].passed is None

        # Case 3 (TIMEOUT/FAILED): should have SKIPPED scores
        failed_result = next(r for r in result.case_results if r.status == EvaluationStatus.FAILED)
        assert len(failed_result.metric_scores) == 1
        assert failed_result.metric_scores[0].details.get("reason") == "execution_failed"

        # Case 4 (PARTIAL): should have SKIPPED scores (not successful)
        partial_result = next(
            r for r in result.case_results if r.status == EvaluationStatus.PARTIAL
        )
        assert len(partial_result.metric_scores) == 1
        assert partial_result.metric_scores[0].details.get("reason") == "execution_failed"


# =============================================================================
# Scenario C: Evaluator failures
# =============================================================================

class TestScenarioCEvaluatorFailures:
    """Test evaluator failure isolation."""

    def setup_method(self):
        self.case = EvaluationCase(input="Test", expected_output="Out")
        self.dataset = EvaluationDataset(name="eval_fail_test", cases=[self.case])

    def test_one_evaluator_fails_others_succeed(self):
        """Test that one failing evaluator doesn't block others."""
        system = SuccessfulSyncSystem(prediction="Out")
        evaluators = [
            ExactMatchEvaluator(),
            FailingEvaluator(),
            ContainsEvaluator(),
        ]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        assert len(result.case_results) == 1
        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED

        # Should have 3 metric scores: 2 successful + 1 error
        assert len(eval_result.metric_scores) == 3

        # Check successful scores
        success_scores = [
            s for s in eval_result.metric_scores
            if s.details.get("reason") != "evaluator_exception"
        ]
        assert len(success_scores) == 2
        for s in success_scores:
            # passed is None when no threshold is set
            assert s.value in (0.0, 1.0)

        # Check error score
        error_scores = [
            s for s in eval_result.metric_scores
            if s.details.get("reason") == "evaluator_exception"
        ]
        assert len(error_scores) == 1
        assert error_scores[0].value == 0.0
        assert error_scores[0].passed is False
        assert "ValueError" in error_scores[0].details.get("error", "")
        assert error_scores[0].details.get("evaluator_type") == "FailingEvaluator"

    def test_all_evaluators_fail(self):
        """Test when all evaluators fail."""
        system = SuccessfulSyncSystem(prediction="Out")
        evaluators = [FailingEvaluator(), FailingEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        eval_result = result.case_results[0]
        assert len(eval_result.metric_scores) == 2
        for s in eval_result.metric_scores:
            assert s.details.get("reason") == "evaluator_exception"
            assert s.passed is False
            assert s.value == 0.0

    def test_evaluator_exception_during_async_evaluation(self):
        """Test async cancellation during evaluation is re-raised."""
        async def run_test():
            system = SuccessfulAsyncSystem(prediction="Out")
            evaluators = [FailingEvaluator()]
            runner = ExperimentRunner(system=system, evaluators=evaluators)
            # The failing evaluator raises ValueError, not CancelledError
            # Cancellation is tested separately in failure handling tests
            result = await runner.run_and_evaluate_async(self.dataset)
            eval_result = result.case_results[0]
            assert len(eval_result.metric_scores) == 1
            assert eval_result.metric_scores[0].details.get("reason") == "evaluator_exception"

        asyncio.run(run_test())


# =============================================================================
# Scenario D: Tracing and cost integration
# =============================================================================

class TestScenarioDTracingCost:
    """Test tracing, token usage, cost, and latency preservation."""

    def setup_method(self):
        self.case = EvaluationCase(input="Test with tracing", expected_output="Out")
        self.dataset = EvaluationDataset(name="tracing_test", cases=[self.case])

    def test_trace_preserved_in_execution_result(self):
        """Test that traces are preserved in execution results."""
        system = TracingSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 1
        exec_result = result.execution_results[0]
        assert exec_result.has_trace()
        assert exec_result.trace is not None
        assert len(exec_result.trace.events) > 0

    def test_token_usage_extracted_from_trace(self):
        """Test that token usage is extracted from trace."""
        system = TracingSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        exec_result = result.execution_results[0]
        assert exec_result.token_usage is not None
        assert exec_result.token_usage.prompt_tokens == 50
        assert exec_result.token_usage.completion_tokens == 30
        assert exec_result.token_usage.total_tokens == 80

    def test_cost_estimate_extracted_from_trace(self):
        """Test that cost estimate is extracted from trace."""
        system = TracingSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        exec_result = result.execution_results[0]
        assert exec_result.cost_estimate is not None
        assert exec_result.cost_estimate.input_cost_usd == 0.001
        assert exec_result.cost_estimate.output_cost_usd == 0.002
        assert exec_result.cost_estimate.total_cost_usd == 0.003
        assert exec_result.cost_estimate.model_name == "test-model"

    def test_latency_preserved(self):
        """Test that latency information is preserved."""
        system = SuccessfulSyncSystem(prediction="Out")
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        exec_result = result.execution_results[0]
        # Latency may be None if not explicitly set, but trace has duration
        wall_clock = exec_result.trace.get_execution_wall_clock_duration_ms()
        assert exec_result.latency_ms is not None or wall_clock is not None

    def test_trace_preserved_in_evaluation_result(self):
        """Test that traces are preserved through to evaluation results."""
        system = TracingSystem()
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        eval_result = result.case_results[0]
        assert eval_result.trace is not None
        assert len(eval_result.trace.events) > 0

    def test_token_and_cost_preserved_in_evaluation_result(self):
        """Test token usage and cost are preserved in evaluation results."""
        system = TracingSystem()
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        eval_result = result.case_results[0]
        assert eval_result.token_usage is not None
        assert eval_result.token_usage.prompt_tokens == 50
        assert eval_result.cost_estimate is not None
        assert eval_result.cost_estimate.total_cost_usd == 0.003


# =============================================================================
# Scenario E: Async workflow
# =============================================================================

class TestScenarioEAsyncWorkflow:
    """Test async workflow scenarios."""

    def setup_method(self):
        self.cases = [
            EvaluationCase(input="Async 1", expected_output="Out 1"),
            EvaluationCase(input="Async 2", expected_output="Out 2"),
        ]
        self.dataset = EvaluationDataset(name="async_test", cases=self.cases)

    def test_async_execution_all_success(self):
        """Test async execution with all cases succeeding."""
        async def run_test():
            system = SuccessfulAsyncSystem(prediction="Async Out", delay=0.01)
            runner = ExperimentRunner(system=system)
            result = await runner.run_async(self.dataset)

            assert result.total_cases == 2
            assert result.successful_cases == 2
            for exec_result in result.execution_results:
                assert exec_result.status == SystemExecutionStatus.SUCCESS
                assert exec_result.prediction == "Async Out"

        asyncio.run(run_test())

    def test_async_execution_mixed_outcomes(self):
        """Test async execution with mixed success/failure."""
        class MixedAsyncSystem(BaseSystem):
            def __init__(self):
                self.call_count = 0

            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Async case 2 failed")
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                trace.add_event(TraceEvent.execution_end(case.id, "success"))
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Success",
                    trace=trace,
                )

        async def run_test():
            system = MixedAsyncSystem()
            runner = ExperimentRunner(system=system)
            result = await runner.run_async(self.dataset)

            assert result.total_cases == 2
            assert result.successful_cases == 1
            assert result.failed_cases == 1

        asyncio.run(run_test())

    def test_sync_system_in_async_runner(self):
        """Test synchronous system executed in async runner via thread pool."""
        async def run_test():
            system = SuccessfulSyncSystem(prediction="Sync in async")
            runner = ExperimentRunner(system=system)
            result = await runner.run_async(self.dataset)

            assert result.total_cases == 2
            assert result.successful_cases == 2
            for exec_result in result.execution_results:
                assert exec_result.prediction == "Sync in async"

        asyncio.run(run_test())

    def test_async_cancellation_propagated(self):
        """Test that asyncio.CancelledError is re-raised during execution."""
        class CancellingSystem(BaseSystem):
            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise asyncio.CancelledError()

        async def run_test():
            system = CancellingSystem()
            runner = ExperimentRunner(system=system)
            with pytest.raises(asyncio.CancelledError):
                await runner.run_async(self.dataset)

        asyncio.run(run_test())

    def test_async_evaluation_with_mixed_results(self):
        """Test async run_and_evaluate with mixed outcomes."""
        class MixedAsyncSystem(BaseSystem):
            def __init__(self):
                self.call_count = 0

            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Case 2 failed")
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                trace.add_event(TraceEvent.execution_end(case.id, "success"))
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Success",
                    trace=trace,
                )

        async def run_test():
            system = MixedAsyncSystem()
            evaluators = [ExactMatchEvaluator()]
            runner = ExperimentRunner(system=system, evaluators=evaluators)
            result = await runner.run_and_evaluate_async(self.dataset)

            assert len(result.case_results) == 2
            statuses = [r.status for r in result.case_results]
            assert EvaluationStatus.PASSED in statuses
            assert EvaluationStatus.ERROR in statuses

        asyncio.run(run_test())


# =============================================================================
# Scenario F: Empty and boundary cases
# =============================================================================

class TestScenarioFBoundaryCases:
    """Test empty datasets, single cases, missing data, etc."""

    def test_empty_dataset(self):
        """Test empty dataset execution."""
        dataset = EvaluationDataset(name="empty", cases=[])
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(dataset)

        assert result.total_cases == 0
        assert result.success_rate == 0.0

        # With evaluators
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        eval_result = runner.run_and_evaluate(dataset)
        assert len(eval_result.case_results) == 0
        assert eval_result.aggregated_metrics.total_cases == 0

    def test_empty_dataset_async(self):
        """Test empty dataset async execution."""
        async def run_test():
            dataset = EvaluationDataset(name="empty", cases=[])
            system = SuccessfulAsyncSystem()
            runner = ExperimentRunner(system=system)
            result = await runner.run_async(dataset)
            assert result.total_cases == 0

        asyncio.run(run_test())

    def test_single_case(self):
        """Test single case dataset."""
        case = EvaluationCase(input="Single", expected_output="Out")
        dataset = EvaluationDataset(name="single", cases=[case])
        system = SuccessfulSyncSystem(prediction="Out")
        runner = ExperimentRunner(system=system)
        result = runner.run(dataset)

        assert result.total_cases == 1
        assert result.successful_cases == 1

    def test_no_evaluators(self):
        """Test execution without evaluators."""
        case = EvaluationCase(input="Test", expected_output="Out")
        dataset = EvaluationDataset(name="no_eval", cases=[case])
        system = SuccessfulSyncSystem(prediction="Out")
        runner = ExperimentRunner(system=system, evaluators=[])
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 1
        assert result.case_results[0].status == EvaluationStatus.PASSED
        assert len(result.case_results[0].metric_scores) == 0

    def test_no_expected_output(self):
        """Test case without expected output."""
        case = EvaluationCase(input="Test", expected_output=None)
        dataset = EvaluationDataset(name="no_expected", cases=[case])
        system = SuccessfulSyncSystem(prediction="Actual")
        evaluators = [ExactMatchEvaluator(), ContainsEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED
        # Exact match should fail (no expected), contains should fail
        exact_score = next(s for s in eval_result.metric_scores if s.name == "exact_match")
        assert exact_score.value == 0.0

    def test_missing_token_cost_data(self):
        """Test execution without token/cost data."""
        class NoTraceSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Out",
                    trace=None,
                )

        case = EvaluationCase(input="Test", expected_output="Out")
        dataset = EvaluationDataset(name="no_trace", cases=[case])
        system = NoTraceSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(dataset)

        exec_result = result.execution_results[0]
        assert exec_result.token_usage is None
        assert exec_result.cost_estimate is None
        assert exec_result.trace is None

    def test_case_with_tool_calls_and_trajectory(self):
        """Test case with expected tool calls and trajectory."""
        case = EvaluationCase(
            input="Use tools",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "test"}),
            ],
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search", "arguments": {"query": "test"}},
            ]),
        )
        dataset = EvaluationDataset(name="tools_traj", cases=[case])
        system = TracingSystem()  # Returns trace with tool calls
        evaluators = [
            ToolSelectionEvaluator(),
            ToolExecutionEvaluator(),
            TrajectoryMatchEvaluator(),
            TrajectoryOrderEvaluator(),
            TrajectoryCompletenessEvaluator(),
        ]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED
        assert len(eval_result.metric_scores) == 5

        # Check metric types
        metric_names = [s.name for s in eval_result.metric_scores]
        assert "tool_selection_accuracy" in metric_names
        assert "tool_execution_correctness" in metric_names
        assert "trajectory_match" in metric_names
        assert "trajectory_order" in metric_names
        assert "trajectory_completeness" in metric_names


# =============================================================================
# Model Consistency Tests
# =============================================================================

class TestModelConsistency:
    """Test consistency between RunnerExperimentResult, ExperimentResult,
    EvaluationResult, SystemExecutionResult."""

    def setup_method(self):
        self.cases = [
            EvaluationCase(input="Test 1", expected_output="Out 1"),
            EvaluationCase(input="Test 2", expected_output="Out 2"),
        ]
        self.dataset = EvaluationDataset(name="consistency_test", cases=self.cases)

    def test_runner_result_to_metrics_result_conversion(self):
        """Test RunnerExperimentResult.to_metrics_experiment_result()."""
        system = SuccessfulSyncSystem(prediction="Out 1")
        runner = ExperimentRunner(system=system, run_id=uuid4())
        runner_result = runner.run(self.dataset)

        metrics_result = runner_result.to_metrics_experiment_result(config_name="test_config")

        assert isinstance(metrics_result, ExperimentResult)
        assert metrics_result.config_name == "test_config"
        assert metrics_result.dataset_name == "consistency_test"
        assert metrics_result.id == runner_result.id
        assert metrics_result.started_at == runner_result.started_at
        assert metrics_result.status == runner_result.status
        assert metrics_result.metadata == runner_result.metadata

        # Check case results mapping
        assert len(metrics_result.case_results) == 2
        for i, case in enumerate(self.cases):
            eval_result = metrics_result.case_results[i]
            exec_result = runner_result.execution_results[i]

            assert eval_result.id == exec_result.id
            assert eval_result.case_id == case.id
            assert eval_result.run_id == runner_result.run_id
            assert eval_result.actual_output == exec_result.prediction
            assert eval_result.trace == exec_result.trace
            assert eval_result.latency_ms == exec_result.latency_ms
            assert eval_result.token_usage == exec_result.token_usage
            assert eval_result.cost_estimate == exec_result.cost_estimate
            assert eval_result.error == exec_result.error
            assert eval_result.started_at == exec_result.started_at
            assert eval_result.completed_at == exec_result.completed_at
            assert eval_result.metadata == exec_result.metadata

    def test_status_mapping_consistency(self):
        """Test status mapping from SystemExecutionStatus to EvaluationStatus."""
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system)
        runner_result = runner.run(self.dataset)

        metrics_result = runner_result.to_metrics_experiment_result()

        for eval_result in metrics_result.case_results:
            assert eval_result.status == EvaluationStatus.PASSED

        # Test ERROR mapping
        class ErrorSystem(BaseSystem):
            def execute(self, case, **kwargs):
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.ERROR,
                    error="Failed",
                )

        runner = ExperimentRunner(system=ErrorSystem())
        runner_result = runner.run(self.dataset)
        metrics_result = runner_result.to_metrics_experiment_result()

        for eval_result in metrics_result.case_results:
            assert eval_result.status == EvaluationStatus.ERROR

        # Test TIMEOUT mapping
        class TimeoutSystem(BaseSystem):
            def execute(self, case, **kwargs):
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.TIMEOUT,
                    error="Timeout",
                )

        runner = ExperimentRunner(system=TimeoutSystem())
        runner_result = runner.run(self.dataset)
        metrics_result = runner_result.to_metrics_experiment_result()

        for eval_result in metrics_result.case_results:
            assert eval_result.status == EvaluationStatus.FAILED

        # Test PARTIAL mapping
        class PartialSystem(BaseSystem):
            def execute(self, case, **kwargs):
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.PARTIAL,
                    prediction="Partial",
                )

        runner = ExperimentRunner(system=PartialSystem())
        runner_result = runner.run(self.dataset)
        metrics_result = runner_result.to_metrics_experiment_result()

        for eval_result in metrics_result.case_results:
            assert eval_result.status == EvaluationStatus.PARTIAL

    def test_execution_counts_match(self):
        """Test that execution counts are consistent."""
        system = MixedSystem()
        runner = ExperimentRunner(system=system)
        runner_result = runner.run(self.dataset)

        # RunnerExperimentResult counts
        assert runner_result.total_cases == 2
        assert runner_result.successful_cases == 1
        assert runner_result.failed_cases == 1
        assert runner_result.timeout_cases == 0
        assert runner_result.partial_cases == 0

        # Converted ExperimentResult counts (using run_and_evaluate to populate scores)
        evaluators = [ExactMatchEvaluator()]
        runner_with_eval = ExperimentRunner(system=MixedSystem(), evaluators=evaluators)
        metrics_result = runner_with_eval.run_and_evaluate(self.dataset)
        assert metrics_result.aggregated_metrics.total_cases == 2
        assert metrics_result.aggregated_metrics.passed_cases == 1
        assert metrics_result.aggregated_metrics.error_cases == 1

    def test_case_ids_preserved(self):
        """Test that case IDs are preserved through the pipeline."""
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system)
        runner_result = runner.run(self.dataset)

        # RunnerExperimentResult has case_ids
        exec_case_ids = [r.case_id for r in runner_result.execution_results]
        assert set(exec_case_ids) == {c.id for c in self.cases}

        # ExperimentResult preserves them
        metrics_result = runner_result.to_metrics_experiment_result()
        eval_case_ids = [r.case_id for r in metrics_result.case_results]
        assert set(eval_case_ids) == {c.id for c in self.cases}

    def test_run_id_propagation(self):
        """Test run_id propagation through results."""
        custom_run_id = uuid4()
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system, run_id=custom_run_id)
        runner_result = runner.run(self.dataset)

        assert runner_result.run_id == custom_run_id
        for exec_result in runner_result.execution_results:
            assert exec_result.run_id == custom_run_id

        metrics_result = runner_result.to_metrics_experiment_result()
        for eval_result in metrics_result.case_results:
            assert eval_result.run_id == custom_run_id

    def test_metadata_propagation(self):
        """Test metadata propagation."""
        custom_metadata = {"key": "value", "env": "test"}
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system, metadata=custom_metadata)
        runner_result = runner.run(self.dataset)

        assert runner_result.metadata == custom_metadata

        metrics_result = runner_result.to_metrics_experiment_result()
        assert metrics_result.metadata == custom_metadata

    def test_timestamps_consistency(self):
        """Test timestamp consistency."""
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system)
        runner_result = runner.run(self.dataset)

        assert runner_result.started_at is not None
        assert runner_result.completed_at is not None
        assert runner_result.completed_at >= runner_result.started_at

        for exec_result in runner_result.execution_results:
            assert exec_result.started_at is not None
            assert exec_result.completed_at is not None
            assert exec_result.completed_at >= exec_result.started_at

        metrics_result = runner_result.to_metrics_experiment_result()
        assert metrics_result.started_at == runner_result.started_at
        assert metrics_result.completed_at == runner_result.completed_at

    def test_finalization_idempotency(self):
        """Test that finalize is idempotent on all result types."""
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system)
        runner_result = runner.run(self.dataset)

        # RunnerExperimentResult
        first_completed = runner_result.completed_at
        runner_result.finalize(status="different", error="ignored")
        assert runner_result.completed_at == first_completed

        # SystemExecutionResult
        exec_result = runner_result.execution_results[0]
        first_exec_completed = exec_result.completed_at
        exec_result.finalize(status=SystemExecutionStatus.PARTIAL, error="ignored")
        assert exec_result.completed_at == first_exec_completed

        # Convert and test ExperimentResult
        metrics_result = runner_result.to_metrics_experiment_result()
        first_metrics_completed = metrics_result.completed_at
        metrics_result.finalize(status="different", error="ignored")
        assert metrics_result.completed_at == first_metrics_completed

        # EvaluationResult
        eval_result = metrics_result.case_results[0]
        first_eval_completed = eval_result.completed_at
        eval_result.finalize(status=EvaluationStatus.FAILED, error="ignored")
        assert eval_result.completed_at == first_eval_completed


# =============================================================================
# Aggregation Consistency Tests
# =============================================================================

class TestAggregationConsistency:
    """Test aggregation consistency between MetricAggregator and ExperimentResult."""

    def setup_method(self):
        self.cases = [
            EvaluationCase(input="Test 1", expected_output="Out 1"),
            EvaluationCase(input="Test 2", expected_output="Out 2"),
        ]
        self.dataset = EvaluationDataset(name="agg_test", cases=self.cases)

    def test_metric_aggregator_matches_experiment_aggregation(self):
        """Test that MetricAggregator and ExperimentResult.aggregate_all_metrics match."""
        system = SuccessfulSyncSystem(prediction="Out 1")  # Only matches first case
        evaluators = [ExactMatchEvaluator(), ContainsEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        # Get experiment-level aggregation
        exp_agg = result.aggregate_all_metrics()

        # Manually aggregate using MetricAggregator
        all_scores = []
        for case_result in result.case_results:
            all_scores.extend(case_result.metric_scores)

        aggregator = MetricAggregator()
        aggregator.add_scores(all_scores)
        manual_agg = aggregator.aggregate()

        # Compare overall
        assert abs(exp_agg["overall"].mean - manual_agg["overall"].mean) < 0.001
        assert abs(exp_agg["overall"].pass_rate - manual_agg["overall"].pass_rate) < 0.001

        # Compare by type
        for mtype in manual_agg["by_type"]:
            assert mtype in exp_agg["by_type"]
            assert abs(exp_agg["by_type"][mtype].mean - manual_agg["by_type"][mtype].mean) < 0.001

        # Compare by name
        for name in manual_agg["by_name"]:
            assert name in exp_agg["by_name"]
            assert abs(exp_agg["by_name"][name].mean - manual_agg["by_name"][name].mean) < 0.001

    def test_case_level_aggregation(self):
        """Test case-level aggregation."""
        system = SuccessfulSyncSystem(prediction="Out 1")
        evaluators = [ExactMatchEvaluator(), StringSimilarityEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        case_aggs = result.get_case_level_aggregates()
        assert len(case_aggs) == 2

        for _i, agg in enumerate(case_aggs):
            assert "overall" in agg
            assert "by_type" in agg
            assert "by_name" in agg
            assert isinstance(agg["overall"], AggregationResult)


# =============================================================================
# Public API Tests
# =============================================================================

class TestPublicAPI:
    """Test public API imports and exports."""

    def test_runner_module_exports(self):
        """Test runner module exports."""
        from eval_harness.runner import (
            BaseSystem,
            ExperimentRunner,
            RunnerExperimentResult,
            SystemExecutionResult,
            SystemExecutionStatus,
            SystemUnderEvaluation,
        )

        # Verify they are usable
        assert SystemUnderEvaluation is not None
        assert BaseSystem is not None
        assert SystemExecutionResult is not None
        assert SystemExecutionStatus is not None
        assert ExperimentRunner is not None
        assert RunnerExperimentResult is not None

    def test_main_module_exports(self):
        """Test main module exports runner components."""
        from eval_harness import (
            BaseSystem,
            ExperimentRunner,
            RunnerExperimentResult,
            SystemExecutionResult,
            SystemExecutionStatus,
            SystemUnderEvaluation,
        )

        assert SystemUnderEvaluation is not None
        assert BaseSystem is not None
        assert SystemExecutionResult is not None
        assert SystemExecutionStatus is not None
        assert ExperimentRunner is not None
        assert RunnerExperimentResult is not None

    def test_no_circular_imports(self):
        """Test that importing modules doesn't cause circular imports."""
        # These should all import without issues
        from eval_harness.core import exceptions
        from eval_harness.datasets import models as dataset_models
        from eval_harness.evaluators import base as eval_base
        from eval_harness.metrics import aggregation
        from eval_harness.metrics import models as metrics_models
        from eval_harness.runner import experiment, models, system
        from eval_harness.tracing import models as tracing_models

        assert experiment is not None
        assert models is not None
        assert system is not None
        assert metrics_models is not None
        assert aggregation is not None
        assert dataset_models is not None
        assert tracing_models is not None
        assert eval_base is not None
        assert exceptions is not None


# =============================================================================
# End-to-End Example Test
# =============================================================================

class TestEndToEndExample:
    """Test the complete end-to-end workflow as a practical example."""

    def test_complete_evaluation_workflow(self):
        """Test a realistic evaluation workflow from dataset to results."""
        # 1. Create dataset with various case types
        cases = [
            EvaluationCase(
                input="What is 2+2?",
                expected_output="4",
                tags=["arithmetic", "easy"],
            ),
            EvaluationCase(
                input="Capital of France?",
                expected_output="Paris",
                tags=["geography"],
            ),
            EvaluationCase(
                input="What is 10/2?",
                expected_output="5",
                tags=["arithmetic"],
            ),
        ]
        dataset = EvaluationDataset(
            name="math_geo_qa",
            description="Math and geography QA",
            version="1.0",
            cases=cases,
            tags=["qa", "test"],
        )

        # 2. Configure evaluators
        evaluators = [
            ExactMatchEvaluator(),
            NormalizedExactMatchEvaluator(),
            ContainsEvaluator(),
            StringSimilarityEvaluator(),
        ]

        # 3. Create system (mock)
        class QASystem(BaseSystem):
            def __init__(self):
                self.answers = {
                    "What is 2+2?": "4",
                    "Capital of France?": "Paris",
                    "What is 10/2?": "5",
                }

            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                answer = self.answers.get(case.input, "Unknown")
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                trace.add_event(
                    TraceEvent.llm_call(case.id, case.input, "mock-model")
                )
                trace.add_event(
                    TraceEvent.llm_response(
                        case.id,
                        answer,
                        "mock-model",
                        token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
                    )
                )
                trace.add_event(TraceEvent.execution_end(case.id, "success"))
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction=answer,
                    trace=trace,
                )

        # 4. Run experiment
        runner = ExperimentRunner(system=QASystem(), evaluators=evaluators)
        result = runner.run_and_evaluate(dataset, config_name="qa_eval")

        # 5. Verify results
        assert result.config_name == "qa_eval"
        assert result.dataset_name == "math_geo_qa"
        assert len(result.case_results) == 3
        assert result.aggregated_metrics.total_cases == 3
        assert result.aggregated_metrics.passed_cases == 3  # All correct

        # 6. Check per-case results
        for eval_result in result.case_results:
            assert eval_result.status == EvaluationStatus.PASSED
            assert len(eval_result.metric_scores) == 4

            # Exact match should pass for all
            exact = next(s for s in eval_result.metric_scores if s.name == "exact_match")
            assert exact.value == 1.0

        # 7. Check experiment-level aggregation
        agg = result.aggregate_all_metrics()
        assert agg["overall"].mean > 0.9  # High average score

        # 8. Check specific metric aggregation
        exact_agg = agg["by_name"]["exact_match"]
        assert exact_agg.mean == 1.0
        # pass_rate is 0.0 when no thresholds are set (no_threshold_count=3)
        # assert exact_agg.pass_rate == 1.0

    def test_workflow_with_tool_evaluation(self):
        """Test workflow with tool-use evaluation."""
        case = EvaluationCase(
            input="Search for AI news and summarize",
            expected_tool_calls=[
                ExpectedToolCall(
                    name="search",
                    arguments={"query": "AI news"},
                    required=True,
                    order=0,
                ),
                ExpectedToolCall(
                    name="summarize",
                    arguments={"text": "..."},
                    required=True,
                    order=1,
                ),
            ],
            expected_trajectory=ExpectedTrajectory(steps=[
                {"tool": "search", "arguments": {"query": "AI news"}},
                {"tool": "summarize", "arguments": {"text": "..."}},
            ]),
        )
        dataset = EvaluationDataset(name="tool_agent", cases=[case])

        class AgentSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))

                # Simulate agent steps
                corr1 = uuid4()
                trace.add_event(
                    TraceEvent.tool_call(
                        case.id, "search", {"query": "AI news"}, correlation_id=corr1
                    )
                )
                trace.add_event(
                    TraceEvent.tool_response(
                        case.id, "search", {"results": ["article1"]}, correlation_id=corr1
                    )
                )

                corr2 = uuid4()
                trace.add_event(
                    TraceEvent.tool_call(
                        case.id, "summarize", {"text": "article1"}, correlation_id=corr2
                    )
                )
                trace.add_event(
                    TraceEvent.tool_response(
                        case.id,
                        "summarize",
                        {"summary": "AI news summary"},
                        correlation_id=corr2,
                    )
                )

                trace.add_event(TraceEvent.execution_end(case.id, "success"))
                trace.finalize(status="success")

                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="AI news summary",
                    trace=trace,
                )

        evaluators = [
            ToolSelectionEvaluator(),
            ToolExecutionEvaluator(),
            TrajectoryMatchEvaluator(exact_match=False),
            TrajectoryOrderEvaluator(),
            TrajectoryCompletenessEvaluator(),
        ]

        runner = ExperimentRunner(system=AgentSystem(), evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 1
        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED
        assert len(eval_result.metric_scores) == 5

        # All tool/trajectory evaluators should score well
        for score in eval_result.metric_scores:
            assert score.value > 0.5  # All should pass reasonably well


# =============================================================================
# Hardening Tests
# =============================================================================

class TestHardening:
    """Tests for edge cases and hardening."""

    def test_mutable_defaults_not_shared(self):
        """Test that mutable defaults are not shared between instances."""
        result1 = RunnerExperimentResult(dataset_id=uuid4(), dataset_name="test")
        result2 = RunnerExperimentResult(dataset_id=uuid4(), dataset_name="test")

        result1.metadata["key"] = "value1"
        result2.metadata["key"] = "value2"

        assert result1.metadata["key"] == "value1"
        assert result2.metadata["key"] == "value2"

    def test_system_execution_result_mutable_defaults(self):
        """Test SystemExecutionResult mutable defaults."""
        result1 = SystemExecutionResult(case_id=uuid4(), status=SystemExecutionStatus.SUCCESS)
        result2 = SystemExecutionResult(case_id=uuid4(), status=SystemExecutionStatus.SUCCESS)

        result1.metadata["key"] = "value1"
        result2.metadata["key"] = "value2"

        assert result1.metadata["key"] == "value1"
        assert result2.metadata["key"] == "value2"

    def test_evaluation_result_mutable_defaults(self):
        """Test EvaluationResult mutable defaults."""
        result1 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)
        result2 = EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED)

        result1.metadata["key"] = "value1"
        result2.metadata["key"] = "value2"

        assert result1.metadata["key"] == "value1"
        assert result2.metadata["key"] == "value2"

    def test_experiment_result_mutable_defaults(self):
        """Test ExperimentResult mutable defaults."""
        result1 = ExperimentResult(config_name="test", dataset_name="test")
        result2 = ExperimentResult(config_name="test", dataset_name="test")

        result1.metadata["key"] = "value1"
        result2.metadata["key"] = "value2"

        assert result1.metadata["key"] == "value1"
        assert result2.metadata["key"] == "value2"

    def test_invalid_evaluator_return_type(self):
        """Test handling of invalid evaluator return type."""
        class InvalidEvaluator(ExactMatchEvaluator):
            def evaluate(self, case, prediction, **kwargs):
                return "not a MetricScore"

        system = SuccessfulSyncSystem(prediction="Out")
        evaluators = [InvalidEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)

        case = EvaluationCase(input="Test", expected_output="Out")
        dataset = EvaluationDataset(name="test", cases=[case])
        result = runner.run_and_evaluate(dataset)

        eval_result = result.case_results[0]
        assert len(eval_result.metric_scores) == 1
        assert eval_result.metric_scores[0].details.get("reason") == "evaluator_exception"
        assert "TypeError" in eval_result.metric_scores[0].details.get("error", "")

    def test_system_returns_none(self):
        """Test system returning None."""
        class NoneSystem(BaseSystem):
            def execute(self, case, **kwargs):
                return None

        system = NoneSystem()
        runner = ExperimentRunner(system=system)

        case = EvaluationCase(input="Test")
        dataset = EvaluationDataset(name="test", cases=[case])
        result = runner.run(dataset)

        assert result.total_cases == 1
        assert result.failed_cases == 1
        assert "invalid type" in result.execution_results[0].error.lower()

    def test_case_with_no_matching_execution_result(self):
        """Test case where execution result is missing."""
        system = SuccessfulSyncSystem()
        runner = ExperimentRunner(system=system)

        # Create dataset with 2 cases but only 1 will be "executed" in mapping
        case1 = EvaluationCase(input="Test 1")
        case2 = EvaluationCase(input="Test 2")
        dataset = EvaluationDataset(name="test", cases=[case1, case2])

        # Manually create runner result with only 1 execution result
        runner_result = RunnerExperimentResult(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            run_id=runner.run_id,
        )
        exec_result = SystemExecutionResult(
            case_id=case1.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Out",
        )
        runner_result.add_execution_result(exec_result)
        runner_result.finalize()

        # Evaluate - case2 has no execution result
        result = runner._evaluate_results(runner_result, dataset.cases, "test")

        assert len(result.case_results) == 2
        # case2 should have ERROR status
        case2_result = next(r for r in result.case_results if r.case_id == case2.id)
        assert case2_result.status == EvaluationStatus.ERROR
        assert case2_result.error == "No execution result found"

    def test_evaluator_kwargs_passed_correctly(self):
        """Test that evaluator kwargs (trace, expected_trajectory) are passed."""
        case = EvaluationCase(
            input="Test",
            expected_tool_calls=[ExpectedToolCall(name="test", arguments={})],
            expected_trajectory=ExpectedTrajectory(steps=[{"tool": "test"}]),
        )
        dataset = EvaluationDataset(name="test", cases=[case])

        received_kwargs = {}

        class CaptureEvaluator(ToolSelectionEvaluator):
            def evaluate(self, case, prediction, trace=None, expected_trajectory=None, **kwargs):
                received_kwargs.update(kwargs)
                if trace is not None:
                    received_kwargs['trace'] = trace
                if expected_trajectory is not None:
                    received_kwargs['expected_trajectory'] = expected_trajectory
                return super().evaluate(
                    case,
                    prediction,
                    trace=trace,
                    expected_trajectory=expected_trajectory,
                    **kwargs,
                )

        system = TracingSystem()  # Returns trace
        evaluators = [CaptureEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        runner.run_and_evaluate(dataset)

        assert "trace" in received_kwargs
        assert received_kwargs["trace"] is not None
        assert "expected_trajectory" in received_kwargs
        assert received_kwargs["expected_trajectory"] is not None

