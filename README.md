# Eval Harness

A reusable AI evaluation framework for LLM and agent evaluation.

## Purpose

Eval Harness provides a foundational architecture for building AI evaluation systems. It focuses on the core data models and structures needed to:

- Define evaluation datasets and test cases with expected outputs, tool calls, and trajectories
- Record execution traces with token usage, latency, and cost tracking
- Compute and aggregate evaluation metrics
- Run experiments and compare results
- Enable baseline comparisons and regression detection

## Architecture

```
eval_harness/
├── core/           # Configuration and exceptions
├── datasets/       # Evaluation cases and datasets
├── tracing/        # Execution traces and events
├── metrics/        # Evaluation results and metrics
├── evaluators/     # (Future) Evaluator implementations
└── reporting/      # (Future) Report generation
```

### Key Design Principles

1. **Provider-agnostic**: No coupling to specific LLM providers, agent frameworks, or external services
2. **Extensible**: Clean separation between datasets, tracing, metrics, evaluators, and reporting
3. **Type-safe**: Full Pydantic validation with clear type annotations
4. **Minimal dependencies**: Only Pydantic and standard library
5. **Serializable**: All models support JSON serialization/deserialization

## Current Implementation Scope (M1)

### Core Models

| Module | Models |
|--------|--------|
| `datasets` | `EvaluationCase`, `EvaluationDataset`, `ExpectedToolCall`, `ExpectedTrajectory`, `AnswerabilityMetadata` |
| `tracing` | `TraceEvent`, `ExecutionTrace`, `TokenUsage`, `LatencyMetrics`, `CostEstimate`, `TraceEventType` |
| `metrics` | `EvaluationResult`, `ExperimentResult`, `MetricScore`, `AggregatedMetrics`, `MetricType`, `EvaluationStatus` |
| `core` | `EvaluationConfig`, exception hierarchy |

### Features

- **Evaluation Cases**: Support for expected outputs, tool calls, trajectories, and answerability metadata
- **Datasets**: Case collections with filtering, tagging, and versioning
- **Tracing**: Hierarchical event recording with token/cost aggregation
- **Metrics**: Per-case and aggregated metrics with thresholds and pass rates
- **Experiments**: Multi-case evaluation runs with aggregated results
- **Configuration**: Validated evaluation run configuration

### Not Yet Implemented (Future Milestones)

- Actual evaluator implementations (M2)
- LLM-as-a-judge evaluators (M2)
- Database persistence (M3)
- Report generation (M3)
- CI/CD integration (M4)
- Web UI (M5)

## Installation

```bash
pip install eval-harness
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from eval_harness import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationConfig,
    EvaluationResult,
    ExperimentResult,
    MetricScore,
    MetricType,
    EvaluationStatus,
)

# Create a dataset
dataset = EvaluationDataset(
    name="qa_test",
    cases=[
        EvaluationCase(
            input="What is 2+2?",
            expected_output="4",
            tags=["arithmetic"],
        ),
        EvaluationCase(
            input="Capital of France?",
            expected_output="Paris",
            tags=["geography"],
        ),
    ],
)

# Configure evaluation
config = EvaluationConfig(
    name="test_run",
    dataset_path=Path("dataset.json"),
    parallel_workers=4,
)

# Run evaluation (pseudo-code - evaluators not yet implemented)
results = []
for case in dataset:
    # ... your evaluation logic here ...
    result = EvaluationResult(
        case_id=case.id,
        status=EvaluationStatus.PASSED,
        actual_output="4",
    )
    result.add_metric_score(MetricScore(
        metric_type=MetricType.ACCURACY,
        name="exact_match",
        value=1.0,
        threshold=0.8,
    ))
    result.finalize()
    results.append(result)

# Aggregate results
experiment = ExperimentResult(config_name=config.name, dataset_name=dataset.name)
for result in results:
    experiment.add_case_result(result)
experiment.finalize()

print(f"Pass rate: {experiment.get_pass_rate():.1%}")
```

## Running Tests

```bash
pytest tests/unit -v
```

## Linting and Type Checking

```bash
ruff check .
mypy eval_harness
```

## License

MIT