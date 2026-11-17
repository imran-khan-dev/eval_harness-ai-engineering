"""Deterministic answer evaluators for exact match,
normalized match, contains, and string similarity."""

from __future__ import annotations

import difflib
import re
from typing import Any

from eval_harness.datasets.models import EvaluationCase
from eval_harness.evaluators.base import BaseEvaluator
from eval_harness.metrics.models import MetricScore, MetricType


class ExactMatchEvaluator(BaseEvaluator):
    """Exact string match evaluator.

    Compares prediction and expected answer character-by-character.
    Returns 1.0 for exact match, 0.0 otherwise.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.EXACT_MATCH

    @property
    def name(self) -> str:
        return "exact_match"

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate exact match between prediction and expected output.

        Args:
            case: Evaluation case with expected_output.
            prediction: The actual prediction to evaluate.
            **kwargs: Ignored.

        Returns:
            MetricScore with value 1.0 if exact match, 0.0 otherwise.
        """
        expected = case.expected_output

        if expected is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_expected_output"},
                metadata={"case_id": str(case.id)},
            )

        if prediction is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_prediction"},
                metadata={"case_id": str(case.id), "expected": expected},
            )

        match = prediction == expected
        return self._create_score(
            value=1.0 if match else 0.0,
            details={
                "match": match,
                "prediction_length": len(prediction),
                "expected_length": len(expected),
            },
            metadata={"case_id": str(case.id)},
        )


class NormalizedExactMatchEvaluator(BaseEvaluator):
    """Normalized exact match evaluator.

    Normalizes both prediction and expected answer by:
    - Stripping leading/trailing whitespace
    - Converting to lowercase
    - Collapsing multiple whitespace characters to single space
    Then performs exact match comparison.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.EXACT_MATCH

    @property
    def name(self) -> str:
        return "normalized_exact_match"

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Strip leading/trailing whitespace
        text = text.strip()
        # Convert to lowercase
        text = text.lower()
        # Collapse multiple whitespace to single space
        text = re.sub(r"\s+", " ", text)
        return text

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate normalized exact match.

        Args:
            case: Evaluation case with expected_output.
            prediction: The actual prediction to evaluate.
            **kwargs: Ignored.

        Returns:
            MetricScore with value 1.0 if normalized match, 0.0 otherwise.
        """
        expected = case.expected_output

        if expected is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_expected_output"},
                metadata={"case_id": str(case.id)},
            )

        if prediction is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_prediction"},
                metadata={"case_id": str(case.id), "expected": expected},
            )

        normalized_prediction = self._normalize(prediction)
        normalized_expected = self._normalize(expected)
        match = normalized_prediction == normalized_expected

        return self._create_score(
            value=1.0 if match else 0.0,
            details={
                "match": match,
                "normalized_prediction": normalized_prediction,
                "normalized_expected": normalized_expected,
                "original_prediction_length": len(prediction),
                "original_expected_length": len(expected),
            },
            metadata={"case_id": str(case.id)},
        )


class ContainsEvaluator(BaseEvaluator):
    """Substring contains evaluator.

    Checks whether the expected answer appears as a substring
    in the prediction. Case-sensitive by default.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.EXACT_MATCH

    @property
    def name(self) -> str:
        return "contains"

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        case_sensitive: bool = True,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate whether expected output is contained in prediction.

        Args:
            case: Evaluation case with expected_output.
            prediction: The actual prediction to evaluate.
            case_sensitive: Whether to do case-sensitive comparison (default True).
            **kwargs: Ignored.

        Returns:
            MetricScore with value 1.0 if expected is substring of prediction, 0.0 otherwise.
        """
        expected = case.expected_output

        if expected is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_expected_output"},
                metadata={"case_id": str(case.id)},
            )

        if prediction is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_prediction"},
                metadata={"case_id": str(case.id), "expected": expected},
            )

        if not case_sensitive:
            expected = expected.lower()
            prediction = prediction.lower()

        match = expected in prediction
        return self._create_score(
            value=1.0 if match else 0.0,
            details={
                "match": match,
                "case_sensitive": case_sensitive,
                "expected_length": len(expected),
                "prediction_length": len(prediction),
            },
            metadata={"case_id": str(case.id)},
        )


class StringSimilarityEvaluator(BaseEvaluator):
    """String similarity evaluator using difflib.SequenceMatcher.

    Calculates a deterministic similarity ratio between 0.0 and 1.0
    using Python's standard library difflib.SequenceMatcher.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.SEMANTIC_SIMILARITY

    @property
    def name(self) -> str:
        return "string_similarity"

    def evaluate(
        self,
        case: EvaluationCase,
        prediction: str | None,
        **kwargs: Any,
    ) -> MetricScore:
        """Evaluate string similarity between prediction and expected output.

        Uses difflib.SequenceMatcher.ratio() which returns a float in [0, 1]
        where 1 means identical sequences.

        Args:
            case: Evaluation case with expected_output.
            prediction: The actual prediction to evaluate.
            **kwargs: Ignored.

        Returns:
            MetricScore with similarity ratio as value.
        """
        expected = case.expected_output

        if expected is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_expected_output"},
                metadata={"case_id": str(case.id)},
            )

        if prediction is None:
            return self._create_score(
                value=0.0,
                details={"reason": "no_prediction"},
                metadata={"case_id": str(case.id), "expected": expected},
            )

        similarity = difflib.SequenceMatcher(None, prediction, expected).ratio()

        return self._create_score(
            value=similarity,
            details={
                "similarity": similarity,
                "prediction_length": len(prediction),
                "expected_length": len(expected),
            },
            metadata={"case_id": str(case.id)},
        )
