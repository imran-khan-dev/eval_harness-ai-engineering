"""Eval Harness - A reusable AI evaluation framework."""

from eval_harness.core.config import EvaluationConfig
from eval_harness.core.exceptions import (
    ConfigurationError,
    DatasetError,
    EvaluationError,
    MetricError,
    ReportingError,
    TraceError,
    ValidationError,
)
from eval_harness.datasets.models import (
    AnswerabilityMetadata,
    EvaluationCase,
    EvaluationDataset,
    ExpectedToolCall,
    ExpectedTrajectory,
)
from eval_harness.evaluators import (
    BaseEvaluator,
    ContainsEvaluator,
    Evaluator,
    ExactMatchEvaluator,
    JSONFieldMatchEvaluator,
    JSONSchemaEvaluator,
    JSONValidityEvaluator,
    NormalizedExactMatchEvaluator,
    StringSimilarityEvaluator,
    ToolArgumentEvaluator,
    ToolExecutionEvaluator,
    ToolSelectionEvaluator,
    TrajectoryCompletenessEvaluator,
    TrajectoryMatchEvaluator,
    TrajectoryOrderEvaluator,
)
from eval_harness.metrics.aggregation import (
    AggregationResult,
    MetricAggregator,
    aggregate_scores,
)
from eval_harness.metrics.models import (
    AggregatedMetrics,
    EvaluationResult,
    ExperimentResult,
    MetricScore,
)
from eval_harness.runner import (
    BaseSystem,
    ExperimentRunner,
    RunnerExperimentResult,
    SystemExecutionResult,
    SystemExecutionStatus,
    SystemUnderEvaluation,
)
from eval_harness.tracing.models import (
    CostEstimate,
    ExecutionTrace,
    LatencyMetrics,
    TokenUsage,
    TraceEvent,
)

__version__ = "0.1.0"

__all__ = [
    "EvaluationConfig",
    "EvaluationError",
    "ConfigurationError",
    "ValidationError",
    "TraceError",
    "DatasetError",
    "MetricError",
    "ReportingError",
    "EvaluationCase",
    "EvaluationDataset",
    "ExpectedToolCall",
    "ExpectedTrajectory",
    "AnswerabilityMetadata",
    "TraceEvent",
    "ExecutionTrace",
    "TokenUsage",
    "LatencyMetrics",
    "CostEstimate",
    "EvaluationResult",
    "ExperimentResult",
    "MetricScore",
    "AggregatedMetrics",
    "MetricAggregator",
    "AggregationResult",
    "aggregate_scores",
    "Evaluator",
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "NormalizedExactMatchEvaluator",
    "ContainsEvaluator",
    "StringSimilarityEvaluator",
    "JSONValidityEvaluator",
    "JSONSchemaEvaluator",
    "JSONFieldMatchEvaluator",
    "ToolSelectionEvaluator",
    "ToolArgumentEvaluator",
    "ToolExecutionEvaluator",
    "TrajectoryMatchEvaluator",
    "TrajectoryOrderEvaluator",
    "TrajectoryCompletenessEvaluator",
    "SystemUnderEvaluation",
    "BaseSystem",
    "SystemExecutionResult",
    "SystemExecutionStatus",
    "ExperimentRunner",
    "RunnerExperimentResult",
]
