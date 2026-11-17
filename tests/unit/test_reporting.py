"""Tests for the reporting module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from eval_harness.datasets.models import EvaluationCase
from eval_harness.metrics.models import (
    EvaluationResult,
    EvaluationStatus,
    ExperimentResult,
    MetricScore,
    MetricType,
)
from eval_harness.reporting import JSONReportGenerator, MarkdownReportGenerator
from eval_harness.runner.models import (
    RunnerExperimentResult,
    SystemExecutionResult,
    SystemExecutionStatus,
)
from eval_harness.tracing.models import CostEstimate, TokenUsage


class TestJSONReportGenerator:
    """Tests for JSONReportGenerator."""

    def setup_method(self):
        self.generator = JSONReportGenerator()

    def _create_experiment_result(self) -> ExperimentResult:
        """Create a sample ExperimentResult for testing."""
        case1 = EvaluationCase(input="Test 1", expected_output="Answer 1")
        case2 = EvaluationCase(input="Test 2", expected_output="Answer 2")

        result = ExperimentResult(
            config_name="test_config",
            dataset_name="test_dataset",
        )

        # Case 1: Passed with multiple metrics
        eval1 = EvaluationResult(
            id=uuid4(),
            case_id=case1.id,
            status=EvaluationStatus.PASSED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.EXACT_MATCH,
                    name="exact_match",
                    value=1.0,
                    threshold=0.8,
                    passed=True,
                ),
                MetricScore(
                    metric_type=MetricType.SEMANTIC_SIMILARITY,
                    name="string_similarity",
                    value=0.95,
                    threshold=0.7,
                    passed=True,
                ),
            ],
            actual_output="Answer 1",
            latency_ms=150.0,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, cached_tokens=10),
            cost_estimate=CostEstimate(
                input_cost_usd=0.001,
                output_cost_usd=0.002,
                total_cost_usd=0.003,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        # Case 2: Failed
        eval2 = EvaluationResult(
            id=uuid4(),
            case_id=case2.id,
            status=EvaluationStatus.FAILED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.EXACT_MATCH,
                    name="exact_match",
                    value=0.0,
                    threshold=0.8,
                    passed=False,
                ),
            ],
            actual_output="Wrong Answer",
            error="Output mismatch",
            latency_ms=120.0,
            token_usage=TokenUsage(prompt_tokens=80, completion_tokens=30),
            cost_estimate=CostEstimate(
                input_cost_usd=0.0008,
                output_cost_usd=0.0015,
                total_cost_usd=0.0023,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval1)
        result.add_case_result(eval2)
        result.finalize()

        return result

    def _create_runner_result(self) -> RunnerExperimentResult:
        """Create a sample RunnerExperimentResult for testing."""
        dataset_id = uuid4()
        run_id = uuid4()

        result = RunnerExperimentResult(
            dataset_id=dataset_id,
            dataset_name="runner_dataset",
            run_id=run_id,
        )

        # Successful execution
        exec1 = SystemExecutionResult(
            id=uuid4(),
            case_id=uuid4(),
            run_id=run_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Correct answer",
            latency_ms=200.0,
            token_usage=TokenUsage(prompt_tokens=150, completion_tokens=75, cached_tokens=5),
            cost_estimate=CostEstimate(
                input_cost_usd=0.0015,
                output_cost_usd=0.003,
                total_cost_usd=0.0045,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        # Failed execution
        exec2 = SystemExecutionResult(
            id=uuid4(),
            case_id=uuid4(),
            run_id=run_id,
            status=SystemExecutionStatus.ERROR,
            prediction=None,
            error="System error",
            latency_ms=50.0,
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=0),
            cost_estimate=CostEstimate(
                input_cost_usd=0.0005,
                output_cost_usd=0.0,
                total_cost_usd=0.0005,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_execution_result(exec1)
        result.add_execution_result(exec2)
        result.finalize()

        return result

    def test_generate_from_experiment_result(self):
        """Test generating JSON from ExperimentResult."""
        result = self._create_experiment_result()
        report = self.generator.generate(result)

        assert report["report_metadata"]["report_format"] == "json"
        assert report["experiment"]["config_name"] == "test_config"
        assert report["experiment"]["dataset_name"] == "test_dataset"
        assert report["execution_summary"]["total_cases"] == 2
        assert report["execution_summary"]["passed_cases"] == 1
        assert report["execution_summary"]["failed_cases"] == 1
        assert report["execution_summary"]["pass_rate"] == 0.5

    def test_generate_from_runner_result(self):
        """Test generating JSON from RunnerExperimentResult."""
        result = self._create_runner_result()
        report = self.generator.generate(result)

        assert report["report_metadata"]["report_format"] == "json"
        assert report["experiment"]["dataset_name"] == "runner_dataset"
        assert report["execution_summary"]["total_cases"] == 2
        assert report["execution_summary"]["successful_cases"] == 1
        assert report["execution_summary"]["failed_cases"] == 1
        assert report["execution_summary"]["success_rate"] == 0.5

    def test_json_serialization(self):
        """Test that generated report is JSON serializable."""
        result = self._create_experiment_result()
        report = self.generator.generate(result)

        # Should not raise
        json_str = json.dumps(report)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_generate_json_string(self):
        """Test generate_json method returns string."""
        result = self._create_experiment_result()
        json_str = self.generator.generate_json(result)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["experiment"]["config_name"] == "test_config"

    def test_metric_summary_in_report(self):
        """Test metric summary is included correctly."""
        result = self._create_experiment_result()
        report = self.generator.generate(result)

        metric_summary = report["metric_summary"]
        assert "exact_match" in metric_summary
        assert "semantic_similarity" in metric_summary
        assert metric_summary["exact_match"]["count"] == 2
        assert metric_summary["exact_match"]["mean"] == 0.5  # (1.0 + 0.0) / 2

    def test_cost_summary_in_report(self):
        """Test cost summary is included correctly."""
        result = self._create_experiment_result()
        report = self.generator.generate(result)

        cost_summary = report["cost_summary"]
        assert cost_summary["total_cost_usd"] == pytest.approx(0.0053)
        assert cost_summary["total_input_tokens"] == 180
        assert cost_summary["total_output_tokens"] == 80
        assert cost_summary["total_cached_tokens"] == 10

    def test_latency_summary_in_report(self):
        """Test latency summary is included correctly."""
        result = self._create_experiment_result()
        report = self.generator.generate(result)

        latency_summary = report["latency_summary"]
        assert latency_summary["total_cases_with_latency"] == 2
        assert latency_summary["avg_latency_ms"] == 135.0
        assert latency_summary["min_latency_ms"] == 120.0
        assert latency_summary["max_latency_ms"] == 150.0

    def test_case_results_in_report(self):
        """Test individual case results are included."""
        result = self._create_experiment_result()
        report = self.generator.generate(result)

        case_results = report["case_results"]
        assert len(case_results) == 2
        assert case_results[0]["status"] == "passed"
        assert case_results[1]["status"] == "failed"
        assert len(case_results[0]["metric_scores"]) == 2
        assert len(case_results[1]["metric_scores"]) == 1

    def test_empty_experiment(self):
        """Test handling of empty experiment."""
        result = ExperimentResult(
            config_name="empty",
            dataset_name="empty_dataset",
        )
        result.finalize()

        report = self.generator.generate(result)

        assert report["execution_summary"]["total_cases"] == 0
        assert report["execution_summary"]["pass_rate"] == 0.0
        assert report["metric_summary"] == {}
        assert report["case_results"] == []

    def test_runner_empty_experiment(self):
        """Test handling of empty runner experiment."""
        result = RunnerExperimentResult(
            dataset_id=uuid4(),
            dataset_name="empty_runner",
        )
        result.finalize()

        report = self.generator.generate(result)

        assert report["execution_summary"]["total_cases"] == 0
        assert report["execution_summary"]["success_rate"] == 0.0
        assert report["execution_results"] == []

    def test_missing_optional_data(self):
        """Test handling of missing optional data (traces, costs, etc.)."""
        case = EvaluationCase(input="Test")
        result = ExperimentResult(
            config_name="test",
            dataset_name="test",
        )

        eval_result = EvaluationResult(
            id=uuid4(),
            case_id=case.id,
            status=EvaluationStatus.PASSED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.EXACT_MATCH,
                    name="exact_match",
                    value=1.0,
                ),
            ],
            # No latency, tokens, cost, trace
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval_result)
        result.finalize()

        report = self.generator.generate(result)

        assert report["cost_summary"]["total_cost_usd"] == 0.0
        assert report["cost_summary"]["total_tokens"] == 0
        assert report["latency_summary"]["total_cases_with_latency"] == 0
        assert report["case_results"][0]["latency_ms"] is None
        assert report["case_results"][0]["token_usage"] is None
        assert report["case_results"][0]["cost_estimate"] is None

    def test_deterministic_output(self):
        """Test that identical inputs produce identical output."""
        # Use fixed UUIDs for deterministic comparison
        fixed_case1_id = uuid4()
        fixed_case2_id = uuid4()
        fixed_eval1_id = uuid4()
        fixed_eval2_id = uuid4()
        fixed_exp_id = uuid4()
        fixed_start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        fixed_end = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

        def create_fixed_result():
            case1 = EvaluationCase(input="Test 1", expected_output="Answer 1")
            case1.id = fixed_case1_id
            case2 = EvaluationCase(input="Test 2", expected_output="Answer 2")
            case2.id = fixed_case2_id

            result = ExperimentResult(
                config_name="test_config",
                dataset_name="test_dataset",
                id=fixed_exp_id,
                started_at=fixed_start,
            )

            eval1 = EvaluationResult(
                id=fixed_eval1_id,
                case_id=fixed_case1_id,
                status=EvaluationStatus.PASSED,
                metric_scores=[
                    MetricScore(
                        metric_type=MetricType.EXACT_MATCH,
                        name="exact_match",
                        value=1.0,
                        threshold=0.8,
                        passed=True,
                    ),
                    MetricScore(
                        metric_type=MetricType.SEMANTIC_SIMILARITY,
                        name="string_similarity",
                        value=0.95,
                        threshold=0.7,
                        passed=True,
                    ),
                ],
                actual_output="Answer 1",
                latency_ms=150.0,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, cached_tokens=10),
                cost_estimate=CostEstimate(
                    input_cost_usd=0.001,
                    output_cost_usd=0.002,
                    total_cost_usd=0.003,
                    model_name="test-model",
                ),
                started_at=fixed_start,
                completed_at=fixed_end,
            )

            eval2 = EvaluationResult(
                id=fixed_eval2_id,
                case_id=fixed_case2_id,
                status=EvaluationStatus.FAILED,
                metric_scores=[
                    MetricScore(
                        metric_type=MetricType.EXACT_MATCH,
                        name="exact_match",
                        value=0.0,
                        threshold=0.8,
                        passed=False,
                    ),
                ],
                actual_output="Wrong Answer",
                error="Output mismatch",
                latency_ms=120.0,
                token_usage=TokenUsage(prompt_tokens=80, completion_tokens=30),
                cost_estimate=CostEstimate(
                    input_cost_usd=0.0008,
                    output_cost_usd=0.0015,
                    total_cost_usd=0.0023,
                    model_name="test-model",
                ),
                started_at=fixed_start,
                completed_at=fixed_end,
            )

            result.add_case_result(eval1)
            result.add_case_result(eval2)
            result.finalize()
            return result

        result1 = create_fixed_result()
        result2 = create_fixed_result()

        report1 = self.generator.generate(result1)
        report2 = self.generator.generate(result2)

        # Compare without timestamps
        for r in (report1, report2):
            r["report_metadata"]["generated_at"] = "FIXED_TIMESTAMP"
            r["experiment"]["started_at"] = "FIXED_TIMESTAMP"
            r["experiment"]["completed_at"] = "FIXED_TIMESTAMP"
            for cr in r["case_results"]:
                cr["started_at"] = "FIXED_TIMESTAMP"
                cr["completed_at"] = "FIXED_TIMESTAMP"

        assert report1 == report2

    def test_unsupported_result_type(self):
        """Test error on unsupported result type."""
        with pytest.raises(TypeError, match="Unsupported result type"):
            self.generator.generate("not a result")


class TestMarkdownReportGenerator:
    """Tests for MarkdownReportGenerator."""

    def setup_method(self):
        self.generator = MarkdownReportGenerator()

    def _create_experiment_result(self) -> ExperimentResult:
        """Create a sample ExperimentResult for testing."""
        case1 = EvaluationCase(input="Test 1", expected_output="Answer 1")
        case2 = EvaluationCase(input="Test 2", expected_output="Answer 2")

        result = ExperimentResult(
            config_name="test_config",
            dataset_name="test_dataset",
        )

        eval1 = EvaluationResult(
            id=uuid4(),
            case_id=case1.id,
            status=EvaluationStatus.PASSED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.EXACT_MATCH,
                    name="exact_match",
                    value=1.0,
                    threshold=0.8,
                    passed=True,
                ),
                MetricScore(
                    metric_type=MetricType.SEMANTIC_SIMILARITY,
                    name="string_similarity",
                    value=0.95,
                    threshold=0.7,
                    passed=True,
                ),
            ],
            actual_output="Answer 1",
            latency_ms=150.0,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            cost_estimate=CostEstimate(
                input_cost_usd=0.001,
                output_cost_usd=0.002,
                total_cost_usd=0.003,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        eval2 = EvaluationResult(
            id=uuid4(),
            case_id=case2.id,
            status=EvaluationStatus.FAILED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.EXACT_MATCH,
                    name="exact_match",
                    value=0.0,
                    threshold=0.8,
                    passed=False,
                ),
            ],
            actual_output="Wrong Answer",
            error="Output mismatch",
            latency_ms=120.0,
            token_usage=TokenUsage(prompt_tokens=80, completion_tokens=30),
            cost_estimate=CostEstimate(
                input_cost_usd=0.0008,
                output_cost_usd=0.0015,
                total_cost_usd=0.0023,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval1)
        result.add_case_result(eval2)
        result.finalize()

        return result

    def _create_runner_result(self) -> RunnerExperimentResult:
        """Create a sample RunnerExperimentResult for testing."""
        dataset_id = uuid4()
        run_id = uuid4()

        result = RunnerExperimentResult(
            dataset_id=dataset_id,
            dataset_name="runner_dataset",
            run_id=run_id,
        )

        exec1 = SystemExecutionResult(
            id=uuid4(),
            case_id=uuid4(),
            run_id=run_id,
            status=SystemExecutionStatus.SUCCESS,
            prediction="Correct answer",
            latency_ms=200.0,
            token_usage=TokenUsage(prompt_tokens=150, completion_tokens=75),
            cost_estimate=CostEstimate(
                input_cost_usd=0.0015,
                output_cost_usd=0.003,
                total_cost_usd=0.0045,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        exec2 = SystemExecutionResult(
            id=uuid4(),
            case_id=uuid4(),
            run_id=run_id,
            status=SystemExecutionStatus.ERROR,
            prediction=None,
            error="System error",
            latency_ms=50.0,
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=0),
            cost_estimate=CostEstimate(
                input_cost_usd=0.0005,
                output_cost_usd=0.0,
                total_cost_usd=0.0005,
                model_name="test-model",
            ),
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_execution_result(exec1)
        result.add_execution_result(exec2)
        result.finalize()

        return result

    def test_generate_from_experiment_result(self):
        """Test generating Markdown from ExperimentResult."""
        result = self._create_experiment_result()
        markdown = self.generator.generate(result)

        assert "# Experiment Report: test_dataset" in markdown
        assert "**Config:** `test_config`" in markdown
        assert "## Summary" in markdown
        assert "## Metric Summary" in markdown
        assert "## Cost & Performance Summary" in markdown
        assert "## Per-Case Results" in markdown
        assert "exact_match" in markdown
        assert "string_similarity" in markdown

    def test_generate_from_runner_result(self):
        """Test generating Markdown from RunnerExperimentResult."""
        result = self._create_runner_result()
        markdown = self.generator.generate(result)

        assert "# Execution Report: runner_dataset" in markdown
        assert "**Dataset ID:**" in markdown
        assert "## Summary" in markdown
        assert "## Cost & Performance Summary" in markdown
        assert "## Execution Results" in markdown
        assert "Correct answer" in markdown
        assert "System error" in markdown

    def test_empty_experiment(self):
        """Test handling of empty experiment."""
        result = ExperimentResult(
            config_name="empty",
            dataset_name="empty_dataset",
        )
        result.finalize()

        markdown = self.generator.generate(result)

        assert "# Experiment Report: empty_dataset" in markdown
        assert "Total Cases:** 0" in markdown
        assert "Passed:** 0 (0.0%)" in markdown
        assert "*No metrics available*" in markdown

    def test_runner_empty_experiment(self):
        """Test handling of empty runner experiment."""
        result = RunnerExperimentResult(
            dataset_id=uuid4(),
            dataset_name="empty_runner",
        )
        result.finalize()

        markdown = self.generator.generate(result)

        assert "# Execution Report: empty_runner" in markdown
        assert "Total Cases:** 0" in markdown
        assert "*No cases executed*" in markdown

    def test_failed_cases_section(self):
        """Test failed cases detail section."""
        result = self._create_experiment_result()
        markdown = self.generator.generate(result)

        assert "## Failed Cases Details" in markdown
        assert "Output mismatch" in markdown
        assert "exact_match" in markdown

    def test_skipped_cases_section(self):
        """Test skipped cases section."""
        case = EvaluationCase(input="Test")
        result = ExperimentResult(
            config_name="test",
            dataset_name="test",
        )

        eval_result = EvaluationResult(
            id=uuid4(),
            case_id=case.id,
            status=EvaluationStatus.SKIPPED,
            metric_scores=[],
            error="No execution result",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval_result)
        result.finalize()

        markdown = self.generator.generate(result)

        assert "## Skipped Cases" in markdown
        assert "No execution result" in markdown

    def test_missing_optional_data(self):
        """Test handling of missing optional data."""
        case = EvaluationCase(input="Test")
        result = ExperimentResult(
            config_name="test",
            dataset_name="test",
        )

        eval_result = EvaluationResult(
            id=uuid4(),
            case_id=case.id,
            status=EvaluationStatus.PASSED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.EXACT_MATCH,
                    name="exact_match",
                    value=1.0,
                ),
            ],
            # No latency, tokens, cost
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval_result)
        result.finalize()

        markdown = self.generator.generate(result)

        assert "N/A" in markdown  # Missing data shows as N/A
        assert "exact_match" in markdown

    def test_large_experiment_summary_only(self):
        """Test that large experiments show summary only."""
        generator = MarkdownReportGenerator(max_cases_in_table=2)
        result = self._create_experiment_result()

        # Add more cases to exceed limit
        for i in range(5):
            case = EvaluationCase(input=f"Test {i+3}")
            eval_result = EvaluationResult(
                id=uuid4(),
                case_id=case.id,
                status=EvaluationStatus.PASSED,
                metric_scores=[
                    MetricScore(
                        metric_type=MetricType.EXACT_MATCH,
                        name="exact_match",
                        value=1.0,
                    ),
                ],
                actual_output=f"Answer {i+3}",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            result.add_case_result(eval_result)

        result.finalize()
        markdown = generator.generate(result)

        assert "showing summary only" in markdown

    def test_write_markdown(self, tmp_path):
        """Test writing markdown to file."""
        result = self._create_experiment_result()
        filepath = tmp_path / "report.md"

        self.generator.write_markdown(result, str(filepath))

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "# Experiment Report: test_dataset" in content

    def test_unsupported_result_type(self):
        """Test error on unsupported result type."""
        with pytest.raises(TypeError, match="Unsupported result type"):
            self.generator.generate("not a result")


class TestReportingPublicAPI:
    """Tests for public API imports."""

    def test_json_generator_import(self):
        """Test JSONReportGenerator can be imported."""
        from eval_harness.reporting import JSONReportGenerator

        assert JSONReportGenerator is not None

    def test_markdown_generator_import(self):
        """Test MarkdownReportGenerator can be imported."""
        from eval_harness.reporting import MarkdownReportGenerator

        assert MarkdownReportGenerator is not None

    def test_main_module_import(self):
        """Test generators can be imported from main module."""
        from eval_harness import JSONReportGenerator, MarkdownReportGenerator

        assert JSONReportGenerator is not None
        assert MarkdownReportGenerator is not None


class TestMultipleMetricTypes:
    """Tests with various metric types."""

    def setup_method(self):
        self.generator = JSONReportGenerator()

    def test_tool_and_trajectory_metrics(self):
        """Test report includes tool and trajectory metric types."""
        case = EvaluationCase(input="Test", expected_tool_calls=[])
        result = ExperimentResult(
            config_name="test",
            dataset_name="test",
        )

        eval_result = EvaluationResult(
            id=uuid4(),
            case_id=case.id,
            status=EvaluationStatus.PASSED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.TOOL_SELECTION_ACCURACY,
                    name="tool_selection_accuracy",
                    value=0.8,
                    threshold=0.7,
                    passed=True,
                ),
                MetricScore(
                    metric_type=MetricType.TRAJECTORY_SIMILARITY,
                    name="trajectory_match",
                    value=0.9,
                    threshold=0.8,
                    passed=True,
                ),
                MetricScore(
                    metric_type=MetricType.LATENCY,
                    name="latency",
                    value=150.0,
                    threshold=500.0,
                    passed=True,
                ),
            ],
            actual_output="Result",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval_result)
        result.finalize()

        report = self.generator.generate(result)

        assert "tool_selection_accuracy" in report["metric_summary"]
        assert "trajectory_similarity" in report["metric_summary"]
        assert "latency" in report["metric_summary"]

    def test_custom_metric_type(self):
        """Test custom metric type is handled."""
        case = EvaluationCase(input="Test")
        result = ExperimentResult(
            config_name="test",
            dataset_name="test",
        )

        eval_result = EvaluationResult(
            id=uuid4(),
            case_id=case.id,
            status=EvaluationStatus.PASSED,
            metric_scores=[
                MetricScore(
                    metric_type=MetricType.CUSTOM,
                    name="custom_metric",
                    value=0.75,
                    details={"custom_field": "value"},
                ),
            ],
            actual_output="Result",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        result.add_case_result(eval_result)
        result.finalize()

        report = self.generator.generate(result)

        assert "custom" in report["metric_summary"]
        assert report["metric_summary"]["custom"]["mean"] == 0.75
