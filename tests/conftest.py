"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_file(temp_dir):
    """Create a sample dataset file."""
    dataset_file = temp_dir / "dataset.json"
    dataset_file.write_text(
        '{"name": "test", "cases": [{"input": "test", "expected_output": "answer"}]}'
    )
    return dataset_file
