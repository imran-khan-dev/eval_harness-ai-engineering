"""Exception hierarchy for Eval Harness."""


class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(EvaluationError):
    """Raised when configuration is invalid or missing."""

    pass


class ValidationError(EvaluationError):
    """Raised when data validation fails."""

    pass


class TraceError(EvaluationError):
    """Raised when trace recording or processing fails."""

    pass


class DatasetError(EvaluationError):
    """Raised when dataset operations fail."""

    pass


class MetricError(EvaluationError):
    """Raised when metric computation fails."""

    pass


class ReportingError(EvaluationError):
    """Raised when report generation fails."""

    pass
