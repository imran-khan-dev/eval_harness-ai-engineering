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

## Current Implementation Scope (M1–M4)

### Core Models

| Module | Models |
|--------|--------|
| `datasets` | `EvaluationCase`, `EvaluationDataset`, `ExpectedToolCall`, `ExpectedTrajectory`, `AnswerabilityMetadata` |
| `tracing` | `TraceEvent`, `ExecutionTrace`, `TokenUsage`, `LatencyMetrics`, `LatencyStats`, `CostEstimate`, `ModelPricing`, `ModelPricingRegistry`, `TraceEventType` |
| `metrics` | `EvaluationResult`, `ExperimentResult`, `MetricScore`, `AggregatedMetrics`, `MetricType`, `EvaluationStatus` |
| `core` | `EvaluationConfig`, exception hierarchy |

### Features

- **Evaluation Cases**: Support for expected outputs, tool calls, trajectories, and answerability metadata
- **Datasets**: Case collections with filtering, tagging, and versioning
- **Dataset Loading**: JSON and JSONL file loading with validation and content hashing
- **Tracing**: Hierarchical event recording with token/cost/latency aggregation
- **Token Usage Tracking**: Prompt, completion, and cached token counting with automatic total computation
- **Model Pricing Registry**: Per-model pricing with input/output/cached token rates, fallback support, and version tracking
- **Cost Estimation**: Automatic cost calculation from token usage with cached-token awareness
- **Latency Statistics**: Min, max, avg, median, p95 per operation type (LLM vs tool), wall-clock execution duration
- **Metrics**: Per-case and aggregated metrics with thresholds and pass rates
- **Experiments**: Multi-case evaluation runs with aggregated results
- **Configuration**: Validated evaluation run configuration

### Not Yet Implemented (Future Milestones)

- Actual evaluator implementations (M3)
- LLM-as-a-judge evaluators (M3)
- Database persistence (M5)
- Report generation (M5)
- CI/CD integration (M6)
- Web UI (M7)

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

## Tool Evaluation (M5.3)

Eval Harness provides three evaluators for assessing tool usage in agent executions:

### ToolSelectionEvaluator

Evaluates whether the correct tools were selected by comparing expected tool calls against actual tool calls from trace events. Computes precision, recall, and F1 score.

```python
from eval_harness import ToolSelectionEvaluator, EvaluationCase, ExpectedToolCall
from eval_harness.tracing import ExecutionTrace, TraceEvent, TraceEventType

evaluator = ToolSelectionEvaluator()

case = EvaluationCase(
    input="Search for news and summarize",
    expected_tool_calls=[
        ExpectedToolCall(name="search", arguments={"query": "AI news"}),
        ExpectedToolCall(name="summarize", arguments={"text": "..."}),
    ],
)

trace = ExecutionTrace(case_id=case.id)
trace.add_event(TraceEvent.tool_call(exec_id, "search", {"query": "AI news"}))
trace.add_event(TraceEvent.tool_call(exec_id, "summarize", {"text": "..."}))
trace.finalize()

score = evaluator.evaluate(case, None, trace=trace)
# score.value = 1.0 (F1)
# score.details = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched_tools": ["search", "summarize"], ...}
```

**Options**:
- `count_duplicates=False`: Treat tool calls as sets (default). Use `True` for multiset counting.
- `required_only=False`: Evaluate only required tool calls (those with `required=True`).

### ToolArgumentEvaluator

Evaluates whether tool call arguments match expected arguments. Supports exact matching and nested dictionary comparison.

```python
from eval_harness import ToolArgumentEvaluator

evaluator = ToolArgumentEvaluator()  # exact_match=True by default

case = EvaluationCase(
    input="Test",
    expected_tool_calls=[
        ExpectedToolCall(name="search", arguments={"query": "test", "limit": 10}),
    ],
)

trace = ExecutionTrace(case_id=case.id)
trace.add_event(TraceEvent.tool_call(exec_id, "search", {"query": "test", "limit": 10}))
trace.finalize()

score = evaluator.evaluate(case, None, trace=trace)
# score.value = 1.0 (all arguments match)
# score.details["per_tool_results"]["search"][0]["match"] = True
```

**Options**:
- `exact_match=True`: Require exact argument match. Set `False` for subset matching.
- `required_only=False`: Evaluate only required tool calls.

### ToolExecutionEvaluator

Evaluates whether expected tool calls were executed and whether they succeeded or failed based on trace events (TOOL_RESPONSE or ERROR events).

```python
from eval_harness import ToolExecutionEvaluator

evaluator = ToolExecutionEvaluator()

case = EvaluationCase(
    input="Test",
    expected_tool_calls=[
        ExpectedToolCall(name="search", arguments={"query": "test"}),
        ExpectedToolCall(name="fetch", arguments={"url": "http://example.com"}),
    ],
)

trace = ExecutionTrace(case_id=case.id)
# Tool call + successful response
trace.add_event(TraceEvent.tool_call(exec_id, "search", {"query": "test"}, correlation_id=corr_0))
trace.add_event(TraceEvent.tool_response(exec_id, "search", {"results": [...]}, correlation_id=corr_0))
# Tool call + failed response
trace.add_event(TraceEvent.tool_call(exec_id, "fetch", {"url": "http://example.com"}, correlation_id=corr_1))
trace.add_event(TraceEvent.tool_response(exec_id, "fetch", {"error": "timeout"}, correlation_id=corr_1))
trace.finalize()

score = evaluator.evaluate(case, None, trace=trace)
# score.value = 0.5 (1 of 2 succeeded)
# score.details["per_tool_results"][0]["succeeded"] = True
# score.details["per_tool_results"][1]["succeeded"] = False
```

**Options**:
- `required_only=False`: Evaluate only required tool calls.
- `check_arguments=False`: When `True`, also verify arguments match (subset check) and count mismatches as failures.

### Metric Types

| Evaluator | Metric Type | Metric Name |
|-----------|-------------|-------------|
| ToolSelectionEvaluator | `tool_selection_accuracy` | `tool_selection_accuracy` |
| ToolArgumentEvaluator | `tool_call_correctness` | `tool_argument_correctness` |
| ToolExecutionEvaluator | `tool_call_correctness` | `tool_execution_correctness` |

All evaluators follow the standard `Evaluator` protocol and return `MetricScore` objects with values in [0, 1], optional thresholds, and detailed breakdowns in `details`.


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

## Cost and Performance Tracking (M4)

Eval Harness provides comprehensive cost and performance tracking for LLM and agent evaluations through the `eval_harness.tracing` module.

### Token Usage Tracking

The `TokenUsage` model tracks token consumption for LLM calls:

```python
from eval_harness.tracing import TokenUsage

usage = TokenUsage(
    prompt_tokens=1000,      # Input tokens sent to the model
    completion_tokens=500,   # Output tokens generated by the model
    cached_tokens=200,       # Cached input tokens (e.g., prompt caching)
)
# total_tokens = prompt + completion + cached = 1700
print(usage.total_tokens)  # 1700
```

**Key points**:
- `cached_tokens` are a subset of `prompt_tokens` (not additional tokens)
- For cost calculation, regular input tokens = `prompt_tokens - cached_tokens`
- `total_tokens` includes cached tokens as a separate count for budget tracking
- Token usage aggregates automatically across multiple LLM events in an `ExecutionTrace`

### Model Pricing Registry and Cost Estimation

The `ModelPricingRegistry` manages per-model pricing configurations:

```python
from eval_harness.tracing import ModelPricing, ModelPricingRegistry, TokenUsage

registry = ModelPricingRegistry()

# Register model pricing (prices per 1M tokens in USD)
gpt4o = ModelPricing(
    model_name="gpt-4o",
    input_price_per_million=5.0,        # $5/1M input tokens
    output_price_per_million=15.0,      # $15/1M output tokens
    cached_input_price_per_million=1.0, # $1/1M cached tokens (optional)
    version="2024-11",                  # Pricing data version
)
registry.register(gpt4o)

# Optional fallback for unknown models
fallback = ModelPricing(
    model_name="default",
    input_price_per_million=10.0,
    output_price_per_million=30.0,
)
registry.set_fallback(fallback)

# Estimate cost from token usage
usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, cached_tokens=200)
cost = registry.estimate_cost("gpt-4o", usage)

print(f"Input cost: ${cost.input_cost_usd:.6f}")   # (800/1M)*5 + (200/1M)*1 = $0.0042
print(f"Output cost: ${cost.output_cost_usd:.6f}") # (500/1M)*15 = $0.0075
print(f"Total cost: ${cost.total_cost_usd:.6f}")   # $0.0117
print(f"Model: {cost.model_name}")                 # "gpt-4o"
print(f"Version: {cost.pricing_version}")          # "2024-11"
print(f"All models: {cost.model_names}")           # ["gpt-4o"]
```

**Cached-token accounting convention**:
- Cached tokens are a **subset** of prompt tokens, not additional
- Regular input tokens = `max(0, prompt_tokens - cached_tokens)`
- Cached tokens priced at `cached_input_price_per_million` (falls back to input price)
- No double-counting: cached tokens are not billed as both regular and cached input

**Unknown model handling**:
- `estimate_cost()` returns `None` for unregistered models (no fallback)
- With fallback set, returns fallback pricing estimate
- Zero token usage always yields zero cost

**Multiple models in one execution**:
- Costs aggregate automatically in `ExecutionTrace.total_cost`
- `model_names` list tracks all contributing models
- `pricing_versions` list tracks all pricing versions (deduplicated)
- `model_name` and `pricing_version` retain first model's info for backward compatibility

### Latency Statistics

The `LatencyStats` model provides aggregated latency statistics:

```python
from eval_harness.tracing import LatencyStats

# From a list of durations (milliseconds)
stats = LatencyStats.from_durations([10.0, 20.0, 30.0, 40.0, 50.0])

print(f"Count: {stats.count}")        # 5
print(f"Min: {stats.min_ms}ms")       # 10.0ms
print(f"Max: {stats.max_ms}ms")       # 50.0ms
print(f"Avg: {stats.avg_ms}ms")       # 30.0ms
print(f"Median: {stats.median_ms}ms") # 30.0ms
print(f"P95: {stats.p95_ms}ms")       # 48.0ms (linear interpolation)
```

**Percentile calculation**: Uses linear interpolation on sorted values for deterministic results.

### ExecutionTrace Latency Aggregation

`ExecutionTrace` provides automatic latency aggregation:

```python
from eval_harness.tracing import (
    ExecutionTrace, TraceEvent, TraceEventType,
    LatencyMetrics, TokenUsage, CostEstimate
)
from uuid import uuid4
from datetime import UTC, datetime, timedelta

trace = ExecutionTrace(case_id=uuid4())

# Record LLM call with latency
call_latency = LatencyMetrics(
    start_time=datetime.now(UTC),
    end_time=datetime.now(UTC) + timedelta(milliseconds=50),
)
trace.add_event(TraceEvent.llm_call(exec_id, "prompt", "gpt-4o", latency=call_latency))

# Record LLM response with tokens, cost, and latency
resp_latency = LatencyMetrics(
    start_time=datetime.now(UTC),
    end_time=datetime.now(UTC) + timedelta(milliseconds=200),
)
usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
cost = CostEstimate(input_cost_usd=0.005, output_cost_usd=0.0075)
trace.add_event(TraceEvent.llm_response(
    exec_id, "response", "gpt-4o",
    token_usage=usage, cost=cost, latency=resp_latency
))

# Record tool call with latency
tool_latency = LatencyMetrics(
    start_time=datetime.now(UTC),
    end_time=datetime.now(UTC) + timedelta(milliseconds=100),
)
trace.add_event(TraceEvent.tool_call(exec_id, "search", {"q": "test"}, latency=tool_latency))

# Get latency summaries
llm_stats = trace.get_llm_latency_stats()           # All LLM events (call + response)
resp_stats = trace.get_llm_response_latency_stats() # Only LLM_RESPONSE events
tool_stats = trace.get_tool_latency_stats()         # Tool events only

# Wall-clock execution duration (separate from summed operations)
wall_clock = trace.get_execution_wall_clock_duration_ms()

# Complete summary
summary = trace.get_latency_summary()
# {
#   "llm_stats": LatencyStats(...),
#   "tool_stats": LatencyStats(...),
#   "execution_wall_clock_ms": 123.45
# }
```

**Key behaviors**:
- `get_llm_latency_stats()` includes both `LLM_CALL` and `LLM_RESPONSE` events
- `get_llm_response_latency_stats()` only includes `LLM_RESPONSE` (avoids double-counting)
- `get_tool_latency_stats()` only includes tool events (`TOOL_CALL`, `TOOL_RESPONSE`)
- `get_execution_wall_clock_duration_ms()` measures wall-clock time from trace start to end (or now)
- Wall-clock duration is **independent** from summed operation durations
- Missing durations are gracefully ignored (not counted, not zeroed)

### Complete Example: Recording Events and Retrieving Summaries

```python
from eval_harness.tracing import (
    InMemoryTraceRecorder, TraceEvent, TraceEventType,
    LatencyMetrics, TokenUsage, ModelPricing, ModelPricingRegistry,
    ExecutionTrace
)
from uuid import uuid4
from datetime import UTC, datetime, timedelta

# Setup
recorder = InMemoryTraceRecorder()
exec_id = uuid4()
case_id = uuid4()

# Configure pricing
registry = ModelPricingRegistry()
registry.register(ModelPricing(
    model_name="gpt-4o",
    input_price_per_million=5.0,
    output_price_per_million=15.0,
    cached_input_price_per_million=1.0,
    version="2024-11",
))

# Start execution
recorder.start_execution(exec_id, case_id)

# Record LLM call
recorder.record_event(TraceEvent.llm_call(
    exec_id, "What is 2+2?", "gpt-4o",
    latency=LatencyMetrics(
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(milliseconds=50)
    )
))

# Record LLM response with tokens and cost
usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, cached_tokens=200)
cost = registry.estimate_cost("gpt-4o", usage)
recorder.record_event(TraceEvent.llm_response(
    exec_id, "4", "gpt-4o",
    token_usage=usage,
    cost=cost,
    latency=LatencyMetrics(
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(milliseconds=200)
    )
))

# Record tool call
recorder.record_event(TraceEvent.tool_call(
    exec_id, "calculator", {"expr": "2+2"},
    latency=LatencyMetrics(
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(milliseconds=100)
    )
))

# Complete execution
recorder.end_execution(exec_id, status="success")

# Retrieve and analyze
trace = recorder.get_execution(exec_id)

print(f"Tokens: {trace.total_tokens.prompt_tokens} prompt, "
      f"{trace.total_tokens.completion_tokens} completion, "
      f"{trace.total_tokens.cached_tokens} cached")
print(f"Cost: ${trace.total_cost.total_cost_usd:.6f} "
      f"(model: {trace.total_cost.model_name}, "
      f"version: {trace.total_cost.pricing_version})")
print(f"Models used: {trace.total_cost.model_names}")

llm = trace.get_llm_response_latency_stats()
print(f"LLM latency: avg={llm.avg_ms:.1f}ms, p95={llm.p95_ms:.1f}ms, count={llm.count}")

tool = trace.get_tool_latency_stats()
print(f"Tool latency: avg={tool.avg_ms:.1f}ms, count={tool.count}")

print(f"Wall clock: {trace.get_execution_wall_clock_duration_ms():.1f}ms")

# Trajectory reconstruction (preserves all aggregates)
from eval_harness.tracing import reconstruct_trajectory, validate_trajectory
traj = reconstruct_trajectory(trace)
validation = validate_trajectory(trace)
print(f"Trajectory valid: {validation.is_valid}")
```

### Known Limitations

1. **In-memory pricing configuration**: `ModelPricingRegistry` is in-memory only. No persistence to disk or external configuration sources. Pricing must be registered programmatically at startup.

2. **Recorder thread-safety**: `InMemoryTraceRecorder` is not thread-safe. Concurrent access from multiple threads requires external synchronization.

3. **Latency double-counting**: `get_llm_latency_stats()` includes both `LLM_CALL` and `LLM_RESPONSE` events. If both have durations set, they are both counted. Use `get_llm_response_latency_stats()` to count only response events.

4. **No external pricing APIs**: Pricing must be manually configured. No integration with provider pricing APIs or automatic price updates.

5. **CostEstimate model attribution**: When aggregating costs from multiple models, `model_name` and `pricing_version` retain only the first model's values (for backward compatibility). Use `model_names` and `pricing_versions` lists for complete attribution.

6. **No automatic cost estimation**: The recorder does not automatically estimate costs from token usage. Costs must be computed via `ModelPricingRegistry` and attached to events manually.