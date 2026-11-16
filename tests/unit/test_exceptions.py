"""Tests for core exceptions."""


from eval_harness.core.exceptions import (
    ConfigurationError,
    DatasetError,
    EvaluationError,
    MetricError,
    ReportingError,
    TraceError,
    ValidationError,
)


class TestEvaluationError:
    """Tests for base EvaluationError."""

    def test_basic_error(self) -> None:
        err = EvaluationError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.details == {}

    def test_error_with_details(self) -> None:
        err = EvaluationError("Failed", details={"code": 500, "retry": True})
        assert "Failed" in str(err)
        assert err.details == {"code": 500, "retry": True}

    def test_inheritance(self) -> None:
        err = EvaluationError("test")
        assert isinstance(err, Exception)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_inherits_from_evaluation_error(self) -> None:
        err = ConfigurationError("Invalid config")
        assert isinstance(err, EvaluationError)
        assert isinstance(err, ConfigurationError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_inherits_from_evaluation_error(self) -> None:
        err = ValidationError("Invalid data")
        assert isinstance(err, EvaluationError)


class TestTraceError:
    """Tests for TraceError."""

    def test_inherits_from_evaluation_error(self) -> None:
        err = TraceError("Trace failed")
        assert isinstance(err, EvaluationError)


class TestDatasetError:
    """Tests for DatasetError."""

    def test_inherits_from_evaluation_error(self) -> None:
        err = DatasetError("Dataset error")
        assert isinstance(err, EvaluationError)


class TestMetricError:
    """Tests for MetricError."""

    def test_inherits_from_evaluation_error(self) -> None:
        err = MetricError("Metric error")
        assert isinstance(err, EvaluationError)


class TestReportingError:
    """Tests for ReportingError."""

    def test_inherits_from_evaluation_error(self) -> None:
        err = ReportingError("Report error")
        assert isinstance(err, EvaluationError)


class TestExceptionHierarchy:
    """Tests for exception hierarchy relationships."""

    def test_all_inherit_from_base(self) -> None:
        errors = [
            ConfigurationError("config"),
            ValidationError("validation"),
            TraceError("trace"),
            DatasetError("dataset"),
            MetricError("metric"),
            ReportingError("reporting"),
        ]
        for err in errors:
            assert isinstance(err, EvaluationError)

    def test_catch_all_with_base(self) -> None:
        try:
            raise ConfigurationError("bad config")
        except EvaluationError as e:
            assert isinstance(e, ConfigurationError)
