"""Dataset models and loading utilities for evaluation cases and datasets."""

from eval_harness.datasets.loader import (
    DatasetLoadPolicy,
    load_json_dataset,
    load_jsonl_dataset,
)
from eval_harness.datasets.models import (
    AnswerabilityMetadata,
    EvaluationCase,
    EvaluationDataset,
    ExpectedToolCall,
    ExpectedTrajectory,
)
from eval_harness.datasets.validation import (
    check_duplicate_case_ids,
    compute_dataset_hash,
    validate_cases,
    validate_dataset_structure,
)

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "ExpectedToolCall",
    "ExpectedTrajectory",
    "AnswerabilityMetadata",
    "load_json_dataset",
    "load_jsonl_dataset",
    "DatasetLoadPolicy",
    "compute_dataset_hash",
    "validate_dataset_structure",
    "validate_cases",
    "check_duplicate_case_ids",
]
