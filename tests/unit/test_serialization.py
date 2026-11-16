"""Tests for model serialization and round-tripping."""

from datetime import UTC, datetime
from uuid import uuid4

from eval_harness.core.config import EvaluationConfig
from eval_harness.datasets.models import (
    AnswerabilityMetadata,
    EvaluationCase,
    EvaluationDataset,
    ExpectedToolCall,
    ExpectedTrajectory,
)
from eval_harness.metrics.models import (
    EvaluationResult,
    EvaluationStatus,
    ExperimentResult,
    MetricScore,
    MetricType,
)
from eval_harness.tracing.models import (
    CostEstimate,
    ExecutionTrace,
    LatencyMetrics,
    TokenUsage,
    TraceEvent,
    TraceEventType,
)


class TestDatasetSerialization:
    """Tests for dataset model serialization."""

    def test_case_json_roundtrip(self) -> None:
        case = EvaluationCase(
            input="Test query",
            expected_output="Expected answer",
            expected_tool_calls=[
                ExpectedToolCall(name="search", arguments={"q": "test"})
            ],
            expected_trajectory=ExpectedTrajectory(steps=[{"action": "search"}]),
            answerability=AnswerabilityMetadata(requires_tools=True),
            tags=["test"],
            metadata={"source": "unit_test"},
        )

        json_str = case.model_dump_json()
        restored = EvaluationCase.model_validate_json(json_str)

        assert restored.input == case.input
        assert restored.expected_output == case.expected_output
        assert len(restored.expected_tool_calls) == 1
        assert restored.expected_tool_calls[0].name == "search"
        assert restored.expected_trajectory is not None
        assert restored.answerability.requires_tools is True
        assert restored.tags == ["test"]
        assert restored.metadata == {"source": "unit_test"}
        assert restored.id == case.id

    def test_dataset_json_roundtrip(self) -> None:
        cases = [
            EvaluationCase(input="Case 1"),
            EvaluationCase(input="Case 2", tags=["tag1"]),
        ]
        dataset = EvaluationDataset(
            name="test_dataset",
            description="Test description",
            cases=cases,
            version="1.2.3",
            tags=["regression"],
            metadata={"author": "test"},
        )

        json_str = dataset.model_dump_json()
        restored = EvaluationDataset.model_validate_json(json_str)

        assert restored.name == dataset.name
        assert restored.description == dataset.description
        assert len(restored.cases) == 2
        assert restored.version == "1.2.3"
        assert restored.tags == ["regression"]
        assert restored.metadata == {"author": "test"}
        assert restored.id == dataset.id

    def test_case_dict_roundtrip(self) -> None:
        case = EvaluationCase(input="Dict test")
        data = case.model_dump()
        restored = EvaluationCase.model_validate(data)
        assert restored.input == case.input
        assert restored.id == case.id


class TestTracingSerialization:
    """Tests for tracing model serialization."""

    def test_token_usage_roundtrip(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        json_str = usage.model_dump_json()
        restored = TokenUsage.model_validate_json(json_str)
        assert restored.prompt_tokens == 100
        assert restored.completion_tokens == 50
        assert restored.total_tokens == 150

    def test_latency_metrics_roundtrip(self) -> None:
        start = datetime.now(UTC)
        end = start
        latency = LatencyMetrics(
            start_time=start,
            end_time=end,
            queue_time_ms=10,
            processing_time_ms=90,
        )
        json_str = latency.model_dump_json()
        restored = LatencyMetrics.model_validate_json(json_str)
        assert restored.queue_time_ms == 10
        assert restored.processing_time_ms == 90

    def test_cost_estimate_roundtrip(self) -> None:
        cost = CostEstimate(
            input_cost_usd=0.01,
            output_cost_usd=0.02,
            model_name="gpt-4",
            pricing_version="2024-01",
        )
        json_str = cost.model_dump_json()
        restored = CostEstimate.model_validate_json(json_str)
        assert restored.input_cost_usd == 0.01
        assert restored.output_cost_usd == 0.02
        assert restored.model_name == "gpt-4"
        assert restored.pricing_version == "2024-01"

    def test_trace_event_roundtrip(self) -> None:
        event = TraceEvent(
            event_type=TraceEventType.LLM_CALL,
            data={"model": "gpt-4", "prompt": "Hello"},
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            latency=LatencyMetrics(start_time=datetime.now(UTC)),
            cost=CostEstimate(input_cost_usd=0.001),
            tags=["llm"],
        )
        json_str = event.model_dump_json()
        restored = TraceEvent.model_validate_json(json_str)
        assert restored.event_type == TraceEventType.LLM_CALL
        assert restored.data["model"] == "gpt-4"
        assert restored.token_usage is not None
        assert restored.token_usage.prompt_tokens == 10
        assert restored.latency is not None
        assert restored.cost is not None

    def test_execution_trace_roundtrip(self) -> None:
        trace = ExecutionTrace(case_id=uuid4(), run_id=uuid4())
        trace.add_event(
            TraceEvent(
                event_type=TraceEventType.LLM_CALL,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            )
        )
        trace.finalize(status="success")

        json_str = trace.model_dump_json()
        restored = ExecutionTrace.model_validate_json(json_str)

        assert restored.case_id == trace.case_id
        assert restored.run_id == trace.run_id
        assert len(restored.events) == 1
        assert restored.status == "success"
        assert restored.total_tokens.prompt_tokens == 100
        assert restored.end_time is not None


class TestMetricsSerialization:
    """Tests for metrics model serialization."""

    def test_metric_score_roundtrip(self) -> None:
        score = MetricScore(
            metric_type=MetricType.ACCURACY,
            name="accuracy",
            value=0.95,
            threshold=0.9,
            details={"method": "exact_match"},
        )
        json_str = score.model_dump_json()
        restored = MetricScore.model_validate_json(json_str)
        assert restored.metric_type == MetricType.ACCURACY
        assert restored.value == 0.95
        assert restored.threshold == 0.9
        assert restored.passed is True
        assert restored.details == {"method": "exact_match"}

    def test_evaluation_result_roundtrip(self) -> None:
        result = EvaluationResult(
            case_id=uuid4(),
            run_id=uuid4(),
            status=EvaluationStatus.PASSED,
            actual_output="Correct answer",
            latency_ms=1200.0,
        )
        result.add_metric_score(
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=0.9, threshold=0.8)
        )
        result.finalize()

        json_str = result.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)

        assert restored.case_id == result.case_id
        assert restored.run_id == result.run_id
        assert restored.status == EvaluationStatus.PASSED
        assert restored.actual_output == "Correct answer"
        assert restored.latency_ms == 1200.0
        assert len(restored.metric_scores) == 1
        assert restored.completed_at is not None

    def test_experiment_result_roundtrip(self) -> None:
        exp = ExperimentResult(config_name="config1", dataset_name="dataset1")
        exp.add_case_result(
            EvaluationResult(case_id=uuid4(), status=EvaluationStatus.PASSED, latency_ms=100)
        )
        exp.add_case_result(
            EvaluationResult(case_id=uuid4(), status=EvaluationStatus.FAILED, latency_ms=200)
        )
        exp.finalize()

        json_str = exp.model_dump_json()
        restored = ExperimentResult.model_validate_json(json_str)

        assert restored.config_name == "config1"
        assert restored.dataset_name == "dataset1"
        assert len(restored.case_results) == 2
        assert restored.total_latency_ms == 300
        assert restored.aggregated_metrics.total_cases == 2
        assert restored.aggregated_metrics.passed_cases == 1
        assert restored.aggregated_metrics.failed_cases == 1


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_config_dict_roundtrip(self, tmp_path) -> None:
        dataset_file = tmp_path / "dataset.json"
        dataset_file.write_text('{"cases": []}')

        config = EvaluationConfig(
            name="serialization_test",
            dataset_path=dataset_file,
            parallel_workers=4,
            timeout_seconds=60.0,
            metadata={"key": "value"},
        )

        data = config.model_dump()
        # Path objects become strings in dict
        assert str(data["dataset_path"]) == str(dataset_file)
        assert data["parallel_workers"] == 4
        assert data["metadata"] == {"key": "value"}

        # Can't fully roundtrip due to path validation, but dict dump works
        restored = EvaluationConfig.model_validate(data)
        assert restored.name == config.name
        assert restored.parallel_workers == config.parallel_workers


class TestComplexSerialization:
    """Tests for complex nested serialization scenarios."""

    def test_full_evaluation_pipeline_serialization(self, tmp_path) -> None:
        """Test serializing a complete evaluation scenario."""
        dataset_file = tmp_path / "dataset.json"
        dataset_file.write_text('{"cases": []}')

        # Create config
        config = EvaluationConfig(name="pipeline_test", dataset_path=dataset_file)

        # Create dataset
        dataset = EvaluationDataset(
            name="pipeline_dataset",
            cases=[
                EvaluationCase(
                    input="Question 1",
                    expected_output="Answer 1",
                    expected_tool_calls=[ExpectedToolCall(name="tool1")],
                ),
                EvaluationCase(input="Question 2"),
            ],
        )

        # Create trace
        trace = ExecutionTrace(case_id=dataset.cases[0].id, run_id=uuid4())
        trace.add_event(
            TraceEvent(
                event_type=TraceEventType.LLM_CALL,
                token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25),
                cost=CostEstimate(input_cost_usd=0.005, output_cost_usd=0.01),
            )
        )
        trace.finalize()

        # Create result
        result = EvaluationResult(
            case_id=dataset.cases[0].id,
            run_id=uuid4(),
            status=EvaluationStatus.PASSED,
            trace=trace,
            actual_output="Answer 1",
        )
        result.add_metric_score(
            MetricScore(metric_type=MetricType.ACCURACY, name="acc", value=1.0, threshold=0.8)
        )
        result.finalize()

        # Create experiment
        exp = ExperimentResult(config_name=config.name, dataset_name=dataset.name)
        exp.add_case_result(result)
        exp.finalize()

        # Serialize everything
        config_json = config.model_dump_json()
        dataset_json = dataset.model_dump_json()
        trace_json = trace.model_dump_json()
        result_json = result.model_dump_json()
        exp_json = exp.model_dump_json()

        # Verify all can be parsed back
        assert EvaluationConfig.model_validate_json(config_json)
        assert EvaluationDataset.model_validate_json(dataset_json)
        assert ExecutionTrace.model_validate_json(trace_json)
        assert EvaluationResult.model_validate_json(result_json)
        assert ExperimentResult.model_validate_json(exp_json)
