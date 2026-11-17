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

## Current Implementation Scope (M1 + M2)

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
- **Dataset Loading**: JSON and JSONL file loading with validation and content hashing
- **Tracing**: Hierarchical event recording with token/cost aggregation
- **Metrics**: Per-case and aggregated metrics with thresholds and pass rates
- **Experiments**: Multi-case evaluation runs with aggregated results
- **Configuration**: Validated evaluation run configuration

### Not Yet Implemented (Future Milestones)

- Actual evaluator implementations (M3)
- LLM-as-a-judge evaluators (M3)
- Database persistence (M4)
- Report generation (M4)
- CI/CD integration (M5)
- Web UI (M6)

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

## Dataset Management (M2)

Eval Harness provides robust dataset loading and validation capabilities for JSON and JSONL formats.

### Overview

The `eval_harness.datasets` module supports:
- **Loading** datasets from JSON and JSONL files
- **Validation** of structure and content with detailed error reporting
- **Content Hashing** using deterministic SHA-256 for version identification
- **Load Policies** for configurable edge case handling

### Supported JSON Format

JSON datasets are single-object files with the following structure:

```json
{
  "id": "optional-uuid",
  "name": "dataset_name",
  "description": "Optional description",
  "version": "1.0.0",
  "metadata": {},
  "tags": ["tag1", "tag2"],
  "cases": [
    {
      "id": "optional-case-uuid",
      "input": "Required input text",
      "expected_output": "Optional expected answer",
      "metadata": {},
      "tags": ["case-tag"],
      "expected_tool_calls": [
        {
          "name": "tool_name",
          "arguments": {"arg": "value"},
          "description": "Tool call purpose",
          "required": true,
          "order": 0
        }
      ],
      "expected_trajectory": {
        "steps": [],
        "description": "Expected agent path",
        "max_deviation": 0.0
      },
      "answerability": {
        "is_answerable": true,
        "confidence": 1.0,
        "requires_tools": false,
        "required_tools": [],
        "context_requirements": null,
        "ambiguity_notes": null
      }
    }
  ]
}
```

**Required fields**: `name` (string), `cases` (array)

**Optional fields**: `id`, `description`, `version`, `metadata`, `tags`

### Supported JSONL Format

JSONL (JSON Lines) datasets store one case per line. Dataset metadata is provided via an optional first-line `_meta` record:

```jsonl
{"_meta": {"name": "jsonl_dataset", "description": "Optional description", "version": "1.0.0", "metadata": {}, "tags": ["jsonl"]}}
{"input": "Case 1 input", "expected_output": "Case 1 answer", "tags": ["example"]}
{"input": "Case 2 input", "metadata": {"source": "manual"}}
{"input": "Case 3 input"}
```

**Format options**:
1. **Pure cases** (one per line) - uses filename as name, defaults for rest
2. **First line with `_meta`** - explicit dataset metadata, followed by cases

### Example Dataset Files

**JSON** (`dataset.json`):
```json
{
  "name": "qa_basic",
  "description": "Basic QA test cases",
  "version": "1.0.0",
  "cases": [
    {
      "input": "What is 2+2?",
      "expected_output": "4",
      "tags": ["arithmetic", "easy"]
    },
    {
      "input": "Capital of France?",
      "expected_output": "Paris",
      "tags": ["geography"]
    }
  ]
}
```

**JSONL** (`dataset.jsonl`):
```jsonl
{"_meta": {"name": "qa_jsonl", "description": "QA cases in JSONL", "tags": ["qa"]}}
{"input": "What is 2+2?", "expected_output": "4", "tags": ["arithmetic"]}
{"input": "Capital of France?", "expected_output": "Paris", "tags": ["geography"]}
{"input": "What is 3*7?"}
```

### Loading Datasets Programmatically

```python
from pathlib import Path
from eval_harness.datasets import (
    load_json_dataset,
    load_jsonl_dataset,
    DatasetLoadPolicy,
)

# Load JSON dataset
dataset = load_json_dataset(Path("datasets/qa.json"))

# Load JSONL dataset
dataset = load_jsonl_dataset(Path("datasets/qa.jsonl"))

# Custom load policy
policy = DatasetLoadPolicy(
    allow_empty=False,       # Reject empty datasets
    require_hash=True,       # Compute content hash (default)
    strip_whitespace=True,   # Strip string whitespace (not yet implemented)
)
dataset = load_json_dataset(Path("datasets/qa.json"), policy=policy)

# Access dataset content
print(f"Dataset: {dataset.name} ({len(dataset.cases)} cases)")
for case in dataset.cases:
    print(f"  - {case.input[:50]}...")
```

### Dataset Validation Behavior

Loading performs multi-stage validation:

1. **Structure validation** - Required fields (`name`, `cases`), correct types
2. **Case validation** - Each case requires non-empty `input`; all optional fields validated
3. **Duplicate detection** - Case IDs must be unique within dataset
4. **Empty dataset policy** - Rejected when `allow_empty=False`

```python
from eval_harness.datasets import load_json_dataset
from eval_harness.core.exceptions import ValidationError

try:
    dataset = load_json_dataset("dataset.json")
except ValidationError as e:
    # Detailed error with file path and case indices
    print(f"Validation failed: {e.message}")
    for error in e.details["errors"]:
        print(f"  Case {error['index']} ({error['case_id']}): {error['error']}")
```

### Dataset Version/Hash Identification

Every loaded dataset receives a deterministic SHA-256 content hash stored in metadata:

```python
from eval_harness.datasets import load_json_dataset

dataset = load_json_dataset("dataset.json")
content_hash = dataset.metadata.get("content_hash")
print(f"Content hash: {content_hash}")  # 64-char hex string

# Hash properties:
# - Deterministic: Same content = same hash
# - Order-independent: Cases sorted by ID before hashing
# - Comprehensive: Includes all case fields + dataset metadata
# - 64 characters (SHA-256 hex digest)
```

**Use cases**:
- Verify dataset integrity across environments
- Detect content changes without full comparison
- Reproducible evaluation runs