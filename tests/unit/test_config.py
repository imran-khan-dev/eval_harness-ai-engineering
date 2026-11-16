"""Tests for EvaluationConfig."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_harness.core.config import EvaluationConfig


class TestEvaluationConfig:
    """Tests for EvaluationConfig model."""

    def test_minimal_config(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        config = EvaluationConfig(name="test", dataset_path=dataset_path)
        assert config.name == "test"
        assert config.dataset_path == dataset_path
        assert config.parallel_workers == 1
        assert config.timeout_seconds == 300.0
        assert config.enable_tracing is True
        assert config.cost_tracking_enabled is True

        dataset_path.unlink()

    def test_full_config(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        config = EvaluationConfig(
            name="full test",
            description="A full config",
            dataset_path=dataset_path,
            output_dir=Path("./custom_output"),
            parallel_workers=4,
            timeout_seconds=60.0,
            retry_attempts=2,
            enable_tracing=False,
            trace_sampling_rate=0.5,
            cost_tracking_enabled=False,
            metadata={"env": "test"},
        )
        assert config.name == "full test"
        assert config.description == "A full config"
        assert config.parallel_workers == 4
        assert config.timeout_seconds == 60.0
        assert config.retry_attempts == 2
        assert config.enable_tracing is False
        assert config.trace_sampling_rate == 0.5
        assert config.cost_tracking_enabled is False
        assert config.metadata == {"env": "test"}

        dataset_path.unlink()

    def test_invalid_dataset_path(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(name="test", dataset_path=Path("/nonexistent/path.json"))
        assert "does not exist" in str(exc_info.value)

    def test_invalid_parallel_workers(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, parallel_workers=0)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, parallel_workers=65)

        dataset_path.unlink()

    def test_invalid_timeout(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, timeout_seconds=-1)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, timeout_seconds=0)

        dataset_path.unlink()

    def test_invalid_sampling_rate(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, trace_sampling_rate=-0.1)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, trace_sampling_rate=1.1)

        dataset_path.unlink()

    def test_output_dir_creation(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_output"
            EvaluationConfig(
                name="test", dataset_path=dataset_path, output_dir=output_dir
            )
            assert output_dir.exists()

        dataset_path.unlink()

    def test_extra_fields_forbidden(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        with pytest.raises(ValidationError):
            EvaluationConfig(name="test", dataset_path=dataset_path, extra_field="value")

        dataset_path.unlink()

    def test_validate_assignment(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"cases": []}')
            dataset_path = Path(f.name)

        config = EvaluationConfig(name="test", dataset_path=dataset_path)
        config.parallel_workers = 8
        assert config.parallel_workers == 8

        with pytest.raises(ValidationError):
            config.parallel_workers = 100

        dataset_path.unlink()
