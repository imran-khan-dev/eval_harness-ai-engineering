"""Tests for the runner module: system interface and execution models."""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from eval_harness.datasets.models import EvaluationCase, EvaluationDataset
from eval_harness.metrics.models import (
    EvaluationResult,
    EvaluationStatus,
    ExperimentResult,
)
from eval_harness.runner import ExperimentRunner
from eval_harness.runner.models import (
    RunnerExperimentResult,
    SystemExecutionResult,
    SystemExecutionStatus,
)
from eval_harness.runner.system import (
    BaseSystem,
    SystemUnderEvaluation,
)
from eval_harness.tracing.models import (
    CostEstimate,
    ExecutionTrace,
    TokenUsage,
    TraceEvent,
)


class MockSystem(BaseSystem):
    """Mock system implementation for testing."""

    def __init__(self, prediction: str | None = None, error: str | None = None):
        self._prediction = prediction
        self._error = error

    def execute(
        self,
        case: EvaluationCase,
        **kwargs: Any,
    ) -> SystemExecutionResult:
        if self._error:
            return SystemExecutionResult(
                case_id=case.id,
                status=SystemExecutionStatus.ERROR,
                error=self._error,
            )
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction=self._prediction,
        )


class TestSystemExecutionStatus:
    """Tests for SystemExecutionStatus enum."""

    def test_status_values(self) -> None:
        assert SystemExecutionStatus.SUCCESS == "success"
        assert SystemExecutionStatus.ERROR == "error"
        assert SystemExecutionStatus.TIMEOUT == "timeout"
        assert SystemExecutionStatus.PARTIAL == "partial"


class TestSystemExecutionResult:
    """Tests for SystemExecutionResult model."""

    def setup_method(self) -> None:
        self.case_id = uuid4()

    def test_minimal_successful_result(self) -> None:
        """Test creating a minimal successful execution result."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test prediction",
        )

        assert result.case_id == self.case_id
        assert result.status == SystemExecutionStatus.SUCCESS
        assert result.prediction == "Test prediction"
        assert result.error is None
        assert isinstance(result.started_at, datetime)
        assert result.completed_at is None

    def test_minimal_error_result(self) -> None:
        """Test creating a minimal error execution result."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.ERROR,
            error="Something went wrong",
        )

        assert result.status == SystemExecutionStatus.ERROR
        assert result.error == "Something went wrong"
        assert result.prediction is None

    def test_result_with_trace(self) -> None:
        """Test result with execution trace."""
        trace = ExecutionTrace(case_id=self.case_id)
        trace.add_event(TraceEvent.execution_start(self.case_id, self.case_id))
        trace.finalize(status="success")

        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test prediction",
            trace=trace,
        )

        assert result.trace is trace
        assert result.has_trace() is True

    def test_result_with_run_id(self) -> None:
        """Test result with run ID."""
        run_id = uuid4()
        result = SystemExecutionResult(
            case_id=self.case_id,
            run_id=run_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )

        assert result.run_id == run_id

    def test_result_with_latency(self) -> None:
        """Test result with latency."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
            latency_ms=1500.5,
        )

        assert result.latency_ms == 1500.5

    def test_result_with_metadata(self) -> None:
        """Test result with metadata."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
            metadata={"model": "test-model", "temperature": 0.7},
        )

        assert result.metadata == {"model": "test-model", "temperature": 0.7}

    def test_result_with_token_usage_and_cost(self) -> None:
        """Test result with explicit token usage and cost."""
        token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        cost = CostEstimate(input_cost_usd=0.001, output_cost_usd=0.002)

        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
            token_usage=token_usage,
            cost_estimate=cost,
        )

        assert result.token_usage == token_usage
        assert result.cost_estimate == cost

    def test_extract_aggregates_from_trace(self) -> None:
        """Test that aggregates are extracted from trace when not explicitly set."""
        trace = ExecutionTrace(case_id=self.case_id)
        trace.add_event(TraceEvent.execution_start(self.case_id, self.case_id))
        trace.add_event(
            TraceEvent.llm_response(
                self.case_id,
                "response",
                "test-model",
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
                cost=CostEstimate(input_cost_usd=0.001, output_cost_usd=0.002),
            )
        )
        trace.finalize(status="success")

        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
            trace=trace,
        )

        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == 100
        assert result.cost_estimate is not None
        assert result.cost_estimate.input_cost_usd == 0.001

    def test_explicit_token_usage_not_overridden(self) -> None:
        """Test that explicitly set token usage is not overridden by trace."""
        trace = ExecutionTrace(case_id=self.case_id)
        trace.add_event(
            TraceEvent.llm_response(
                self.case_id,
                "response",
                "test-model",
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            )
        )
        trace.finalize(status="success")

        explicit_usage = TokenUsage(prompt_tokens=200, completion_tokens=100)
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
            trace=trace,
            token_usage=explicit_usage,
        )

        assert result.token_usage == explicit_usage
        assert result.token_usage.prompt_tokens == 200

    def test_finalize_success(self) -> None:
        """Test finalizing a successful execution."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        result.finalize()

        assert result.completed_at is not None
        assert result.status == SystemExecutionStatus.SUCCESS

    def test_finalize_with_error(self) -> None:
        """Test finalizing with an error overrides status."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        result.finalize(error="Something failed")

        assert result.completed_at is not None
        assert result.status == SystemExecutionStatus.ERROR
        assert result.error == "Something failed"

    def test_finalize_with_explicit_status(self) -> None:
        """Test finalizing with explicit status preserves it even with error."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.PARTIAL,
            prediction="Test",
        )
        result.finalize(status=SystemExecutionStatus.PARTIAL, error="Warning")

        assert result.status == SystemExecutionStatus.PARTIAL
        assert result.error == "Warning"

    def test_is_successful(self) -> None:
        """Test is_successful method."""
        # Successful with prediction
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        assert result.is_successful() is True

        # Successful without prediction
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction=None,
        )
        assert result.is_successful() is False

        # Error status
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.ERROR,
            error="Failed",
        )
        assert result.is_successful() is False

    def test_has_trace(self) -> None:
        """Test has_trace method."""
        # No trace
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        assert result.has_trace() is False

        # Empty trace
        trace = ExecutionTrace(case_id=self.case_id)
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
            trace=trace,
        )
        assert result.has_trace() is False

        # Trace with events
        trace.add_event(TraceEvent.execution_start(self.case_id, self.case_id))
        assert result.has_trace() is True

    def test_error_consistency_validation(self) -> None:
        """Test that ERROR status without error gets default message."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.ERROR,
        )
        assert result.error == "Execution failed without error message"

    def test_timeout_status(self) -> None:
        """Test TIMEOUT status."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.TIMEOUT,
            error="Execution timed out after 30s",
        )
        assert result.status == SystemExecutionStatus.TIMEOUT
        assert result.error == "Execution timed out after 30s"

    def test_partial_status(self) -> None:
        """Test PARTIAL status."""
        result = SystemExecutionResult(
            case_id=self.case_id,
            status=SystemExecutionStatus.PARTIAL,
            prediction="Partial output",
        )
        assert result.status == SystemExecutionStatus.PARTIAL
        assert result.prediction == "Partial output"


class TestSystemUnderEvaluationProtocol:
    """Tests for the SystemUnderEvaluation protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that BaseSystem implements the protocol."""
        # Just verify the protocol exists and can be used for type checking
        system: SystemUnderEvaluation = MockSystem(prediction="Test")
        case = EvaluationCase(input="Test input")
        result = system.execute(case)
        assert isinstance(result, SystemExecutionResult)

    def test_mock_system_success(self) -> None:
        """Test MockSystem successful execution."""
        system = MockSystem(prediction="Expected output")
        case = EvaluationCase(input="Test input", expected_output="Expected output")

        result = system.execute(case)

        assert result.status == SystemExecutionStatus.SUCCESS
        assert result.prediction == "Expected output"
        assert result.case_id == case.id

    def test_mock_system_error(self) -> None:
        """Test MockSystem error execution."""
        system = MockSystem(error="Connection failed")
        case = EvaluationCase(input="Test input")

        result = system.execute(case)

        assert result.status == SystemExecutionStatus.ERROR
        assert result.error == "Connection failed"
        assert result.prediction is None

    def test_base_system_is_abstract(self) -> None:
        """Test that BaseSystem cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSystem()  # type: ignore[abstract]

    def test_base_system_subclass_requires_execute(self) -> None:
        """Test that subclass must implement execute."""

        class IncompleteSystem(BaseSystem):
            pass

        with pytest.raises(TypeError):
            IncompleteSystem()  # type: ignore[abstract]


class TestSystemExecutionResultEdgeCases:
    """Tests for edge cases and validation."""

    def test_negative_latency_rejected(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError):
            SystemExecutionResult(
                case_id=uuid4(),
                status=SystemExecutionStatus.SUCCESS,
                prediction="Test",
                latency_ms=-1.0,
            )

    def test_uuid_generation(self) -> None:
        """Test that UUID is auto-generated."""
        result = SystemExecutionResult(
            case_id=uuid4(),
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        assert result.id is not None

    def test_started_at_auto_generated(self) -> None:
        """Test that started_at is auto-generated as UTC."""
        before = datetime.now(timezone.utc)
        result = SystemExecutionResult(
            case_id=uuid4(),
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        after = datetime.now(timezone.utc)

        assert before <= result.started_at <= after
        assert result.started_at.tzinfo == timezone.utc

    def test_completed_at_after_finalize(self) -> None:
        """Test that completed_at is set on finalize."""
        result = SystemExecutionResult(
            case_id=uuid4(),
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        assert result.completed_at is None

        before = datetime.now(timezone.utc)
        result.finalize()
        after = datetime.now(timezone.utc)

        assert result.completed_at is not None
        assert before <= result.completed_at <= after
        assert result.completed_at.tzinfo == timezone.utc


class AsyncMockSystem(BaseSystem):
    """Async mock system for testing async execution."""

    def __init__(
        self,
        prediction: str | None = None,
        error: str | None = None,
        delay: float = 0.0,
    ):
        self._prediction = prediction
        self._error = error
        self._delay = delay

    async def execute(
        self,
        case: EvaluationCase,
        **kwargs: Any,
    ) -> SystemExecutionResult:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            return SystemExecutionResult(
                case_id=case.id,
                status=SystemExecutionStatus.ERROR,
                error=self._error,
            )
        return SystemExecutionResult(
            case_id=case.id,
            status=SystemExecutionStatus.SUCCESS,
            prediction=self._prediction,
        )


class TestRunnerExperimentResult:
    """Tests for RunnerExperimentResult model."""

    def setup_method(self) -> None:
        self.dataset_id = uuid4()
        self.case_ids = [uuid4() for _ in range(3)]

    def test_minimal_result(self) -> None:
        """Test creating a minimal experiment result."""
        result = RunnerExperimentResult(
            dataset_id=self.dataset_id,
            dataset_name="test_dataset",
        )

        assert result.dataset_id == self.dataset_id
        assert result.dataset_name == "test_dataset"
        assert result.execution_results == []
        assert result.total_cases == 0
        assert result.successful_cases == 0
        assert result.success_rate == 0.0

    def test_add_execution_result(self) -> None:
        """Test adding execution results."""
        result = RunnerExperimentResult(
            dataset_id=self.dataset_id,
            dataset_name="test_dataset",
        )

        exec_result = SystemExecutionResult(
            case_id=self.case_ids[0],
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        result.add_execution_result(exec_result)

        assert result.total_cases == 1
        assert result.successful_cases == 1

    def test_finalize(self) -> None:
        """Test finalizing an experiment result."""
        result = RunnerExperimentResult(
            dataset_id=self.dataset_id,
            dataset_name="test_dataset",
        )
        result.finalize(status="completed")

        assert result.completed_at is not None
        assert result.status == "completed"

    def test_execution_counts(self) -> None:
        """Test execution count properties."""
        result = RunnerExperimentResult(
            dataset_id=self.dataset_id,
            dataset_name="test_dataset",
        )

        # Add various statuses
        result.add_execution_result(
            SystemExecutionResult(
                case_id=self.case_ids[0],
                status=SystemExecutionStatus.SUCCESS,
                prediction="1",
            )
        )
        result.add_execution_result(
            SystemExecutionResult(
                case_id=self.case_ids[1],
                status=SystemExecutionStatus.SUCCESS,
                prediction="2",
            )
        )
        result.add_execution_result(
            SystemExecutionResult(
                case_id=self.case_ids[2],
                status=SystemExecutionStatus.ERROR,
                error="Failed",
            )
        )

        assert result.total_cases == 3
        assert result.successful_cases == 2
        assert result.failed_cases == 1
        assert result.timeout_cases == 0
        assert result.partial_cases == 0
        assert result.success_rate == pytest.approx(2 / 3)

    def test_get_execution_result(self) -> None:
        """Test getting execution result by case ID."""
        result = RunnerExperimentResult(
            dataset_id=self.dataset_id,
            dataset_name="test_dataset",
        )

        exec_result = SystemExecutionResult(
            case_id=self.case_ids[0],
            status=SystemExecutionStatus.SUCCESS,
            prediction="Test",
        )
        result.add_execution_result(exec_result)

        found = result.get_execution_result(self.case_ids[0])
        assert found is exec_result

        not_found = result.get_execution_result(uuid4())
        assert not_found is None

    def test_to_metrics_experiment_result(self) -> None:
        """Test conversion to metrics ExperimentResult."""
        result = RunnerExperimentResult(
            dataset_id=self.dataset_id,
            dataset_name="test_dataset",
        )

        result.add_execution_result(
            SystemExecutionResult(
                case_id=self.case_ids[0],
                status=SystemExecutionStatus.SUCCESS,
                prediction="Output 1",
            )
        )
        result.add_execution_result(
            SystemExecutionResult(
                case_id=self.case_ids[1],
                status=SystemExecutionStatus.ERROR,
                error="Failed",
            )
        )
        result.finalize()

        metrics_result = result.to_metrics_experiment_result(config_name="test_config")

        assert metrics_result.config_name == "test_config"
        assert metrics_result.dataset_name == "test_dataset"
        assert metrics_result.id == result.id
        assert len(metrics_result.case_results) == 2
        assert metrics_result.case_results[0].status == EvaluationStatus.PASSED
        assert metrics_result.case_results[1].status == EvaluationStatus.ERROR


class TestExperimentRunnerSync:
    """Tests for synchronous ExperimentRunner."""

    def setup_method(self) -> None:
        self.case1 = EvaluationCase(input="Test 1", expected_output="Out 1")
        self.case2 = EvaluationCase(input="Test 2", expected_output="Out 2")
        self.dataset = EvaluationDataset(name="test_dataset", cases=[self.case1, self.case2])

    def test_run_two_cases(self) -> None:
        """Test running a dataset with two cases."""
        system = MockSystem(prediction="Output")
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 2
        assert result.success_rate == 1.0
        assert result.dataset_name == "test_dataset"
        assert result.dataset_id == self.dataset.id

    def test_run_empty_dataset(self) -> None:
        """Test running an empty dataset."""
        empty_dataset = EvaluationDataset(name="empty", cases=[])
        system = MockSystem(prediction="Output")
        runner = ExperimentRunner(system=system)
        result = runner.run(empty_dataset)

        assert result.total_cases == 0
        assert result.successful_cases == 0
        assert result.success_rate == 0.0

    def test_run_with_run_id(self) -> None:
        """Test that run_id is propagated to execution results."""
        run_id = uuid4()
        system = MockSystem(prediction="Output")
        runner = ExperimentRunner(system=system, run_id=run_id)
        result = runner.run(self.dataset)

        assert result.run_id == run_id
        for exec_result in result.execution_results:
            assert exec_result.run_id == run_id

    def test_run_with_metadata(self) -> None:
        """Test that metadata is included in result."""
        system = MockSystem(prediction="Output")
        runner = ExperimentRunner(system=system, metadata={"model": "test-model"})
        result = runner.run(self.dataset)

        assert result.metadata == {"model": "test-model"}

    def test_case_to_result_association(self) -> None:
        """Test that results are correctly associated with cases."""
        system = MockSystem(prediction="Output")
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert len(result.execution_results) == 2
        case_ids = {r.case_id for r in result.execution_results}
        expected_ids = {self.case1.id, self.case2.id}
        assert case_ids == expected_ids

    def test_system_error_result(self) -> None:
        """Test handling of system returning ERROR result."""
        system = MockSystem(error="Connection failed")
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 0
        assert result.failed_cases == 2
        assert result.success_rate == 0.0
        for exec_result in result.execution_results:
            assert exec_result.status == SystemExecutionStatus.ERROR
            assert exec_result.error == "Connection failed"

    def test_system_exception(self) -> None:
        """Test handling of system raising an exception."""

        class FailingSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise ValueError("Something went wrong")

        system = FailingSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 0
        assert result.failed_cases == 2
        for exec_result in result.execution_results:
            assert exec_result.status == SystemExecutionStatus.ERROR
            assert "ValueError" in exec_result.error

    def test_partial_result(self) -> None:
        """Test handling of PARTIAL system result."""

        class PartialSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.PARTIAL,
                    prediction="Partial output",
                )

        system = PartialSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 0
        assert result.partial_cases == 2

    def test_timeout_result(self) -> None:
        """Test handling of TIMEOUT system result."""

        class TimeoutSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.TIMEOUT,
                    error="Timeout after 30s",
                )

        system = TimeoutSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 0
        assert result.timeout_cases == 2

    def test_continue_after_error(self) -> None:
        """Test that execution continues after individual case fails."""

        class MixedSystem(BaseSystem):
            def __init__(self):
                self.call_count = 0

            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Case 2 failed")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Output",
                )

        system = MixedSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        # Should have results for both cases
        assert result.total_cases == 2
        # First succeeds, second fails
        assert result.successful_cases == 1
        assert result.failed_cases == 1


class TestExperimentRunnerAsync:
    """Tests for asynchronous ExperimentRunner."""

    def setup_method(self) -> None:
        self.case1 = EvaluationCase(input="Test 1")
        self.case2 = EvaluationCase(input="Test 2")
        self.dataset = EvaluationDataset(name="async_test", cases=[self.case1, self.case2])

    @pytest.mark.asyncio
    async def test_run_async_two_cases(self) -> None:
        """Test running a dataset asynchronously with two cases."""
        system = AsyncMockSystem(prediction="Async output")
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 2
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_async_empty_dataset(self) -> None:
        """Test running an empty dataset asynchronously."""
        empty_dataset = EvaluationDataset(name="empty", cases=[])
        system = AsyncMockSystem(prediction="Output")
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(empty_dataset)

        assert result.total_cases == 0

    @pytest.mark.asyncio
    async def test_run_async_with_error(self) -> None:
        """Test async execution with system returning error."""
        system = AsyncMockSystem(error="Async error")
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(self.dataset)

        assert result.total_cases == 2
        assert result.failed_cases == 2
        assert result.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_run_async_with_exception(self) -> None:
        """Test async execution with system raising exception."""

        class FailingAsyncSystem(BaseSystem):
            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise RuntimeError("Async failure")

        system = FailingAsyncSystem()
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(self.dataset)

        assert result.total_cases == 2
        assert result.failed_cases == 2
        for exec_result in result.execution_results:
            assert "RuntimeError" in exec_result.error

    @pytest.mark.asyncio
    async def test_run_async_continues_after_error(self) -> None:
        """Test async execution continues after error."""

        class MixedAsyncSystem(BaseSystem):
            def __init__(self):
                self.call_count = 0

            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Case 2 failed")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Output",
                )

        system = MixedAsyncSystem()
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 1
        assert result.failed_cases == 1

    @pytest.mark.asyncio
    async def test_run_async_with_delay(self) -> None:
        """Test async execution with delays."""
        system = AsyncMockSystem(prediction="Delayed", delay=0.01)
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(self.dataset)

        assert result.total_cases == 2
        assert result.successful_cases == 2


class TestExperimentRunnerIntegration:
    """Integration tests for experiment runner."""

    def test_run_with_trace(self) -> None:
        """Test that traces are preserved in results."""
        from eval_harness.tracing.models import ExecutionTrace, TraceEvent

        class TracingSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                trace.add_event(
                    TraceEvent.llm_call(case.id, "prompt", "test-model")
                )
                trace.add_event(
                    TraceEvent.llm_response(
                        case.id, "response", "test-model",
                        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5)
                    )
                )
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Traced output",
                    trace=trace,
                )

        case = EvaluationCase(input="Test", expected_output="Out")
        dataset = EvaluationDataset(name="trace_test", cases=[case])
        system = TracingSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(dataset)

        assert result.total_cases == 1
        exec_result = result.execution_results[0]
        assert exec_result.has_trace()
        assert exec_result.trace is not None
        assert exec_result.token_usage is not None
        assert exec_result.token_usage.prompt_tokens == 10
        assert exec_result.cost_estimate is not None

    def test_run_preserves_case_metadata(self) -> None:
        """Test that case metadata is accessible."""
        case = EvaluationCase(
            input="Test",
            expected_output="Out",
            metadata={"source": "test", "difficulty": "easy"},
            tags=["unit", "fast"],
        )
        dataset = EvaluationDataset(name="meta_test", cases=[case])
        system = MockSystem(prediction="Output")
        runner = ExperimentRunner(system=system)
        result = runner.run(dataset)

        assert result.total_cases == 1
        exec_result = result.execution_results[0]
        assert exec_result.case_id == case.id

    def test_unique_run_ids(self) -> None:
        """Test that each run gets a unique run_id."""
        system = MockSystem(prediction="Output")
        dataset = EvaluationDataset(name="test", cases=[EvaluationCase(input="Test")])

        runner1 = ExperimentRunner(system=system)
        result1 = runner1.run(dataset)

        runner2 = ExperimentRunner(system=system)
        result2 = runner2.run(dataset)

        assert result1.run_id != result2.run_id
        assert result1.id != result2.id


class TestExperimentRunnerEvaluation:
    """Tests for ExperimentRunner evaluator integration."""

    def setup_method(self) -> None:
        self.case1 = EvaluationCase(input="Test 1", expected_output="Out 1")
        self.case2 = EvaluationCase(input="Test 2", expected_output="Out 2")
        self.dataset = EvaluationDataset(name="test_dataset", cases=[self.case1, self.case2])

    def test_run_and_evaluate_with_answer_evaluators(self) -> None:
        """Test run_and_evaluate with answer evaluators."""
        from eval_harness.evaluators import ContainsEvaluator, ExactMatchEvaluator

        system = MockSystem(prediction="Out 1")  # Only matches first case
        evaluators = [ExactMatchEvaluator(), ContainsEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        assert result.config_name == "runner_experiment"
        assert result.dataset_name == "test_dataset"
        assert len(result.case_results) == 2

        # Case 1: exact match and contains should both pass
        case1_result = next(r for r in result.case_results if r.case_id == self.case1.id)
        assert case1_result.status == EvaluationStatus.PASSED
        assert len(case1_result.metric_scores) == 2
        scores = {s.name: s.value for s in case1_result.metric_scores}
        assert scores["exact_match"] == 1.0
        assert scores["contains"] == 1.0

        # Case 2: prediction is "Out 1" but expected is "Out 2"
        case2_result = next(r for r in result.case_results if r.case_id == self.case2.id)
        assert case2_result.status == EvaluationStatus.PASSED
        assert len(case2_result.metric_scores) == 2
        scores = {s.name: s.value for s in case2_result.metric_scores}
        assert scores["exact_match"] == 0.0
        assert scores["contains"] == 0.0

        # Check aggregation
        assert result.aggregated_metrics.total_cases == 2
        assert result.aggregated_metrics.passed_cases == 2  # Both PASSED (execution succeeded)
        assert "exact_match" in result.aggregated_metrics.mean_scores

    def test_run_and_evaluate_with_structured_evaluators(self) -> None:
        """Test run_and_evaluate with JSON evaluators."""
        import json

        from eval_harness.evaluators import JSONFieldMatchEvaluator, JSONValidityEvaluator

        expected_json = {"name": "John", "age": 30}
        prediction_json = json.dumps({"name": "John", "age": 30})

        case = EvaluationCase(
            input="Test",
            expected_output=json.dumps(expected_json),
        )
        dataset = EvaluationDataset(name="json_test", cases=[case])

        system = MockSystem(prediction=prediction_json)
        evaluators = [JSONValidityEvaluator(), JSONFieldMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 1
        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED
        scores = {s.name: s.value for s in eval_result.metric_scores}
        assert scores["json_validity"] == 1.0
        assert scores["json_field_match"] == 1.0

    def test_run_and_evaluate_with_tool_evaluators(self) -> None:
        """Test run_and_evaluate with tool evaluators (requires trace)."""
        from uuid import uuid4

        from eval_harness.datasets.models import ExpectedToolCall
        from eval_harness.evaluators import ToolSelectionEvaluator
        from eval_harness.tracing.models import ExecutionTrace, TraceEvent

        expected_calls = [
            {"tool": "search", "arguments": {"query": "test"}},
            {"tool": "fetch", "arguments": {"url": "http://example.com"}},
        ]

        case = EvaluationCase(
            input="Test",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"query": "test"}),
                ExpectedToolCall(name="fetch", arguments={"url": "http://example.com"}),
            ],
        )
        dataset = EvaluationDataset(name="tool_test", cases=[case])

        class ToolTracingSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                for call in expected_calls:
                    corr_id = uuid4()
                    trace.add_event(
                        TraceEvent.tool_call(
                            case.id,
                            call["tool"],
                            call["arguments"],
                            correlation_id=corr_id,
                        )
                    )
                    trace.add_event(
                        TraceEvent.tool_response(
                            case.id,
                            call["tool"],
                            {"result": "ok"},
                            correlation_id=corr_id,
                        )
                    )
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Done",
                    trace=trace,
                )
        system = ToolTracingSystem()
        evaluators = [ToolSelectionEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 1
        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED
        scores = {s.name: s.value for s in eval_result.metric_scores}
        assert scores["tool_selection_accuracy"] == 1.0

    def test_run_and_evaluate_with_trajectory_evaluators(self) -> None:
        """Test run_and_evaluate with trajectory evaluators (requires trace)."""
        from uuid import uuid4

        from eval_harness.datasets.models import ExpectedTrajectory
        from eval_harness.evaluators import TrajectoryMatchEvaluator
        from eval_harness.tracing.models import ExecutionTrace, TraceEvent

        case = EvaluationCase(
            input="Test",
            expected_trajectory=ExpectedTrajectory(
                steps=[
                    {"tool": "search"},
                    {"tool": "fetch"},
                    {"tool": "summarize"},
                ]
            ),
        )
        dataset = EvaluationDataset(name="traj_test", cases=[case])

        class TrajTracingSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                for tool in ["search", "fetch", "summarize"]:
                    trace.add_event(
                        TraceEvent.tool_call(
                            case.id,
                            tool,
                            {},
                            correlation_id=uuid4(),
                        )
                    )
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Done",
                    trace=trace,
                )

        system = TrajTracingSystem()
        evaluators = [TrajectoryMatchEvaluator(exact_match=True)]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 1
        eval_result = result.case_results[0]
        assert eval_result.status == EvaluationStatus.PASSED
        scores = {s.name: s.value for s in eval_result.metric_scores}
        assert scores["trajectory_match"] == 1.0

    def test_run_and_evaluate_execution_failure_skips_evaluation(self) -> None:
        """Test that execution failures result in skipped evaluation scores."""
        from eval_harness.evaluators import ExactMatchEvaluator

        system = MockSystem(error="Connection failed")
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        assert len(result.case_results) == 2
        for eval_result in result.case_results:
            assert eval_result.status == EvaluationStatus.ERROR
            # Should have SKIPPED scores (passed=None) for each evaluator
            for score in eval_result.metric_scores:
                assert score.details.get("reason") == "execution_failed"
                assert score.passed is None
                assert score.value == 0.0

    def test_run_and_evaluate_evaluator_exception_handled(self) -> None:
        """Test that evaluator exceptions are caught and recorded."""
        from eval_harness.evaluators import ExactMatchEvaluator

        class FailingEvaluator(ExactMatchEvaluator):
            def evaluate(self, case, prediction, **kwargs):
                raise ValueError("Evaluator failed")

        system = MockSystem(prediction="Out 1")
        evaluators = [FailingEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        assert len(result.case_results) == 2
        for eval_result in result.case_results:
            assert eval_result.status == EvaluationStatus.PASSED
            for score in eval_result.metric_scores:
                assert score.details.get("reason") == "evaluator_exception"
                assert "ValueError" in score.details.get("error", "")
                assert score.passed is False
                assert score.value == 0.0

    def test_run_and_evaluate_empty_evaluators(self) -> None:
        """Test run_and_evaluate with no evaluators configured."""
        system = MockSystem(prediction="Out 1")
        runner = ExperimentRunner(system=system, evaluators=[])
        result = runner.run_and_evaluate(self.dataset)

        assert len(result.case_results) == 2
        for eval_result in result.case_results:
            assert eval_result.status == EvaluationStatus.PASSED
            assert len(eval_result.metric_scores) == 0

    def test_run_and_evaluate_empty_dataset(self) -> None:
        """Test run_and_evaluate with empty dataset."""
        from eval_harness.evaluators import ExactMatchEvaluator

        empty_dataset = EvaluationDataset(name="empty", cases=[])
        system = MockSystem(prediction="Out")
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(empty_dataset)

        assert result.dataset_name == "empty"
        assert len(result.case_results) == 0
        assert result.aggregated_metrics.total_cases == 0

    def test_run_and_evaluate_async(self) -> None:
        """Test async run_and_evaluate."""
        import asyncio

        from eval_harness.evaluators import ExactMatchEvaluator

        async def test_async() -> None:
            system = MockSystem(prediction="Out 1")
            evaluators = [ExactMatchEvaluator()]
            runner = ExperimentRunner(system=system, evaluators=evaluators)
            result = await runner.run_and_evaluate_async(self.dataset)

            assert len(result.case_results) == 2
            case1_result = next(r for r in result.case_results if r.case_id == self.case1.id)
            assert case1_result.status == EvaluationStatus.PASSED
            scores = {s.name: s.value for s in case1_result.metric_scores}
            assert scores["exact_match"] == 1.0

        asyncio.run(test_async())

    def test_run_and_evaluate_multiple_cases_different_predictions(self) -> None:
        """Test multiple cases with different predictions per case."""
        from eval_harness.evaluators import ExactMatchEvaluator

        class MultiPredictionSystem(BaseSystem):
            def __init__(self):
                self.predictions = {
                    "case1": "Correct",
                    "case2": "Wrong",
                }
                self.call_count = 0

            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                pred = self.predictions.get(f"case{self.call_count}", "Default")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction=pred,
                )

        case1 = EvaluationCase(input="Test 1", expected_output="Correct")
        case2 = EvaluationCase(input="Test 2", expected_output="Wrong")
        dataset = EvaluationDataset(name="multi_test", cases=[case1, case2])

        system = MultiPredictionSystem()
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 2
        case1_result = next(r for r in result.case_results if r.case_id == case1.id)
        case2_result = next(r for r in result.case_results if r.case_id == case2.id)

        assert case1_result.metric_scores[0].value == 1.0
        assert case2_result.metric_scores[0].value == 1.0

    def test_run_and_evaluate_aggregation_across_cases(self) -> None:
        """Test that aggregation works correctly across multiple cases."""
        from eval_harness.evaluators import ExactMatchEvaluator, StringSimilarityEvaluator

        case1 = EvaluationCase(input="Test 1", expected_output="Exact")
        case2 = EvaluationCase(input="Test 2", expected_output="Close")
        case3 = EvaluationCase(input="Test 3", expected_output="Different")
        dataset = EvaluationDataset(name="agg_test", cases=[case1, case2, case3])

        class VariedSystem(BaseSystem):
            def __init__(self):
                self.predictions = ["Exact", "Clsoe", "Totally Different"]
                self.call_count = 0

            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                pred = self.predictions[self.call_count]
                self.call_count += 1
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction=pred,
                )

        system = VariedSystem()
        evaluators = [ExactMatchEvaluator(), StringSimilarityEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        # Check per-case scores
        assert len(result.case_results) == 3

        # Check experiment-level aggregation
        agg = result.aggregated_metrics
        assert agg.total_cases == 3
        assert "exact_match" in agg.mean_scores
        assert "semantic_similarity" in agg.mean_scores

        # exact_match: 1.0, 0.0, 0.0 -> mean = 0.333...
        assert abs(agg.mean_scores["exact_match"] - 1/3) < 0.01

        # semantic_similarity: 1.0, ~0.8, ~0.69 -> mean ~0.83
        assert 0.8 < agg.mean_scores["semantic_similarity"] < 0.9

    def test_run_and_evaluate_preserves_execution_info(self) -> None:
        """Test that execution info (trace, latency, tokens) is preserved."""
        from eval_harness.evaluators import ExactMatchEvaluator
        from eval_harness.tracing.models import (
            ExecutionTrace,
            TokenUsage,
            TraceEvent,
        )

        case = EvaluationCase(input="Test", expected_output="Out")
        dataset = EvaluationDataset(name="info_test", cases=[case])

        class InfoSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                trace = ExecutionTrace(case_id=case.id)
                trace.add_event(TraceEvent.execution_start(case.id, case.id))
                trace.add_event(
                    TraceEvent.llm_call(case.id, "prompt", "test-model")
                )
                trace.add_event(
                    TraceEvent.llm_response(
                        case.id, "response", "test-model",
                        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5)
                    )
                )
                trace.finalize(status="success")
                return SystemExecutionResult(
                    case_id=case.id,
                    status=SystemExecutionStatus.SUCCESS,
                    prediction="Out",
                    trace=trace,
                    latency_ms=150.0,
                )

        system = InfoSystem()
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(dataset)

        assert len(result.case_results) == 1
        eval_result = result.case_results[0]

        # Check execution info is preserved
        assert eval_result.trace is not None
        assert eval_result.latency_ms == 150.0
        assert eval_result.token_usage is not None
        assert eval_result.token_usage.prompt_tokens == 10
        assert eval_result.token_usage.completion_tokens == 5
        assert eval_result.cost_estimate is not None

    def test_run_and_evaluate_with_custom_config_name(self) -> None:
        """Test that custom config_name is used."""
        from eval_harness.evaluators import ExactMatchEvaluator

        system = MockSystem(prediction="Out 1")
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset, config_name="my_custom_config")

        assert result.config_name == "my_custom_config"

    def test_run_and_evaluate_mixed_success_failure(self) -> None:
        """Test mixed successful and failed executions."""
        from eval_harness.evaluators import ExactMatchEvaluator

        class MixedSystem(BaseSystem):
            def __init__(self):
                self.call_count = 0

            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                self.call_count += 1
                if self.call_count == 1:
                    return SystemExecutionResult(
                        case_id=case.id,
                        status=SystemExecutionStatus.SUCCESS,
                        prediction="Out 1",
                    )
                else:
                    return SystemExecutionResult(
                        case_id=case.id,
                        status=SystemExecutionStatus.ERROR,
                        error="Failed",
                    )

        system = MixedSystem()
        evaluators = [ExactMatchEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        assert len(result.case_results) == 2
        passed_count = sum(1 for r in result.case_results if r.status == EvaluationStatus.PASSED)
        error_count = sum(1 for r in result.case_results if r.status == EvaluationStatus.ERROR)
        assert passed_count == 1
        assert error_count == 1

        # Successful case should have metric scores
        success_result = next(r for r in result.case_results if r.status == EvaluationStatus.PASSED)
        assert len(success_result.metric_scores) == 1
        assert success_result.metric_scores[0].value == 1.0

        # Failed case should have SKIPPED scores
        error_result = next(r for r in result.case_results if r.status == EvaluationStatus.ERROR)
        assert len(error_result.metric_scores) == 1
        assert error_result.metric_scores[0].details.get("reason") == "execution_failed"
        assert error_result.metric_scores[0].passed is None


class TestExperimentRunnerFailureHandling:
    """Tests for failure handling and execution isolation (M6.4)."""

    def setup_method(self) -> None:
        self.case1 = EvaluationCase(input="Test 1")
        self.case2 = EvaluationCase(input="Test 2")
        self.dataset = EvaluationDataset(name="failure_test", cases=[self.case1, self.case2])

    def test_sync_keyboard_interrupt_reraised(self) -> None:
        """Test that KeyboardInterrupt is re-raised in sync execution."""

        class InterruptingSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise KeyboardInterrupt("Interrupted")

        system = InterruptingSystem()
        runner = ExperimentRunner(system=system)

        with pytest.raises(KeyboardInterrupt):
            runner.run(self.dataset)

    def test_sync_system_exit_reraised(self) -> None:
        """Test that SystemExit is re-raised in sync execution."""

        class ExitingSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise SystemExit("Exiting")

        system = ExitingSystem()
        runner = ExperimentRunner(system=system)

        with pytest.raises(SystemExit):
            runner.run(self.dataset)

    @pytest.mark.asyncio
    async def test_async_cancelled_error_reraised(self) -> None:
        """Test that asyncio.CancelledError is re-raised in async execution."""

        class CancellingSystem(BaseSystem):
            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise asyncio.CancelledError()

        system = CancellingSystem()
        runner = ExperimentRunner(system=system)

        with pytest.raises(asyncio.CancelledError):
            await runner.run_async(self.dataset)

    @pytest.mark.asyncio
    async def test_async_keyboard_interrupt_reraised(self) -> None:
        """Test that KeyboardInterrupt is re-raised in async execution."""

        class InterruptingAsyncSystem(BaseSystem):
            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise KeyboardInterrupt("Interrupted")

        system = InterruptingAsyncSystem()
        runner = ExperimentRunner(system=system)

        with pytest.raises(KeyboardInterrupt):
            await runner.run_async(self.dataset)

    @pytest.mark.asyncio
    async def test_async_system_exit_reraised(self) -> None:
        """Test that SystemExit is re-raised in async execution."""

        class ExitingAsyncSystem(BaseSystem):
            async def execute(self, case: EvaluationCase, **kwargs: Any) -> SystemExecutionResult:
                raise SystemExit("Exiting")

        system = ExitingAsyncSystem()
        runner = ExperimentRunner(system=system)

        with pytest.raises(SystemExit):
            await runner.run_async(self.dataset)

    def test_invalid_system_return_value_sync(self) -> None:
        """Test that invalid sync system return value is handled gracefully."""

        class InvalidReturnSystem(BaseSystem):
            def execute(self, case: EvaluationCase, **kwargs: Any) -> Any:
                return "not a SystemExecutionResult"

        system = InvalidReturnSystem()
        runner = ExperimentRunner(system=system)
        result = runner.run(self.dataset)

        assert result.total_cases == 2
        assert result.failed_cases == 2
        for exec_result in result.execution_results:
            assert exec_result.status == SystemExecutionStatus.ERROR
            assert "invalid type" in exec_result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_system_return_value_async(self) -> None:
        """Test that invalid async system return value is handled gracefully."""

        class InvalidReturnAsyncSystem(BaseSystem):
            async def execute(self, case: EvaluationCase, **kwargs: Any) -> Any:
                return "not a SystemExecutionResult"

        system = InvalidReturnAsyncSystem()
        runner = ExperimentRunner(system=system)
        result = await runner.run_async(self.dataset)

        assert result.total_cases == 2
        assert result.failed_cases == 2
        for exec_result in result.execution_results:
            assert exec_result.status == SystemExecutionStatus.ERROR
            assert "invalid type" in exec_result.error.lower()

    def test_invalid_evaluator_return_value(self) -> None:
        """Test that invalid evaluator return value is handled gracefully."""
        from eval_harness.evaluators import ExactMatchEvaluator

        class InvalidEvaluator(ExactMatchEvaluator):
            def evaluate(self, case, prediction, **kwargs):
                return "not a MetricScore"

        system = MockSystem(prediction="Out 1")
        evaluators = [InvalidEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)
        result = runner.run_and_evaluate(self.dataset)

        assert len(result.case_results) == 2
        for eval_result in result.case_results:
            assert eval_result.status == EvaluationStatus.PASSED
            for score in eval_result.metric_scores:
                assert score.details.get("reason") == "evaluator_exception"
                assert "TypeError" in score.details.get("error", "")
                assert score.passed is False
                assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_async_cancellation_during_evaluation(self) -> None:
        """Test that async cancellation during evaluation is re-raised."""
        from eval_harness.evaluators import ExactMatchEvaluator

        class CancellingEvaluator(ExactMatchEvaluator):
            def evaluate(self, case, prediction, **kwargs):
                raise asyncio.CancelledError()

        system = MockSystem(prediction="Out 1")
        evaluators = [CancellingEvaluator()]
        runner = ExperimentRunner(system=system, evaluators=evaluators)

        with pytest.raises(asyncio.CancelledError):
            await runner.run_and_evaluate_async(self.dataset)

    def test_idempotent_finalize_runner_result(self) -> None:
        """Test that RunnerExperimentResult.finalize is idempotent."""
        runner_result = RunnerExperimentResult(
            dataset_id=uuid4(),
            dataset_name="test",
        )
        runner_result.add_execution_result(
            SystemExecutionResult(
                case_id=self.case1.id,
                status=SystemExecutionStatus.SUCCESS,
                prediction="Out",
            )
        )

        # First finalize
        runner_result.finalize(status="completed")
        first_completed_at = runner_result.completed_at
        first_status = runner_result.status

        # Second finalize should not change anything
        runner_result.finalize(status="different_status", error="some error")
        second_completed_at = runner_result.completed_at
        second_status = runner_result.status

        assert first_completed_at == second_completed_at
        assert first_status == second_status
        assert runner_result.error is None  # error not updated on second call

    def test_idempotent_finalize_experiment_result(self) -> None:
        """Test that ExperimentResult.finalize is idempotent."""
        exp_result = ExperimentResult(
            config_name="test",
            dataset_name="test",
        )
        exp_result.add_case_result(
            EvaluationResult(
                case_id=self.case1.id,
                status=EvaluationStatus.PASSED,
            )
        )

        # First finalize
        exp_result.finalize(status="completed")
        first_completed_at = exp_result.completed_at
        first_status = exp_result.status
        first_total_cases = exp_result.aggregated_metrics.total_cases

        # Second finalize should not change anything
        exp_result.finalize(status="different_status", error="some error")
        second_completed_at = exp_result.completed_at
        second_status = exp_result.status
        second_total_cases = exp_result.aggregated_metrics.total_cases

        assert first_completed_at == second_completed_at
        assert first_status == second_status
        assert first_total_cases == second_total_cases
        assert exp_result.error is None  # error not updated on second call

    def test_idempotent_finalize_evaluation_result(self) -> None:
        """Test that EvaluationResult.finalize is idempotent."""
        eval_result = EvaluationResult(
            case_id=self.case1.id,
            status=EvaluationStatus.PASSED,
        )

        # First finalize
        eval_result.finalize(status=EvaluationStatus.PASSED)
        first_completed_at = eval_result.completed_at
        first_status = eval_result.status

        # Second finalize should not change anything
        eval_result.finalize(status=EvaluationStatus.ERROR, error="some error")
        second_completed_at = eval_result.completed_at
        second_status = eval_result.status

        assert first_completed_at == second_completed_at
        assert first_status == second_status
        assert eval_result.error is None  # error not updated on second call
