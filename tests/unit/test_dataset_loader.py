"""Tests for dataset loading and validation."""

import contextlib
import json
from uuid import uuid4

import pytest

from eval_harness.core.exceptions import DatasetError, ValidationError
from eval_harness.datasets import (
    DatasetLoadPolicy,
    EvaluationCase,
    EvaluationDataset,
    check_duplicate_case_ids,
    compute_dataset_hash,
    load_json_dataset,
    load_jsonl_dataset,
    validate_cases,
    validate_dataset_structure,
)


class TestValidateDatasetStructure:
    """Tests for validate_dataset_structure function."""

    def test_valid_dataset(self) -> None:
        data = {"name": "test", "cases": []}
        validate_dataset_structure(data, "test.json")

    def test_missing_name(self) -> None:
        data = {"cases": []}
        with pytest.raises(ValidationError) as exc_info:
            validate_dataset_structure(data, "test.json")
        assert "missing required 'name' field" in str(exc_info.value)

    def test_missing_cases(self) -> None:
        data = {"name": "test"}
        with pytest.raises(ValidationError) as exc_info:
            validate_dataset_structure(data, "test.json")
        assert "missing required 'cases' field" in str(exc_info.value)

    def test_cases_not_list(self) -> None:
        data = {"name": "test", "cases": "not a list"}
        with pytest.raises(ValidationError) as exc_info:
            validate_dataset_structure(data, "test.json")
        assert "cases' must be a list" in str(exc_info.value)

    def test_not_dict(self) -> None:
        data = ["not", "a", "dict"]
        with pytest.raises(ValidationError) as exc_info:
            validate_dataset_structure(data, "test.json")
        assert "must be a JSON object" in str(exc_info.value)


class TestValidateCases:
    """Tests for validate_cases function."""

    def test_valid_cases(self) -> None:
        cases = [
            {"input": "Case 1"},
            {"input": "Case 2", "expected_output": "Answer"},
        ]
        validated = validate_cases(cases, "test.json")
        assert len(validated) == 2
        assert all(isinstance(c, EvaluationCase) for c in validated)

    def test_invalid_case_missing_input(self) -> None:
        cases = [{"expected_output": "Answer"}]
        with pytest.raises(ValidationError) as exc_info:
            validate_cases(cases, "test.json")
        assert "Failed to validate" in str(exc_info.value)

    def test_invalid_case_empty_input(self) -> None:
        cases = [{"input": ""}]
        with pytest.raises(ValidationError) as exc_info:
            validate_cases(cases, "test.json")
        assert "Failed to validate" in str(exc_info.value)

    def test_error_details_include_index(self) -> None:
        cases = [{"input": "Valid"}, {"expected_output": "No input"}]
        with pytest.raises(ValidationError) as exc_info:
            validate_cases(cases, "test.json")
        details = exc_info.value.details
        assert "errors" in details
        assert any(e["index"] == 1 for e in details["errors"])


class TestCheckDuplicateCaseIds:
    """Tests for check_duplicate_case_ids function."""

    def test_no_duplicates(self) -> None:
        cases = [
            EvaluationCase(input="Case 1"),
            EvaluationCase(input="Case 2"),
        ]
        check_duplicate_case_ids(cases, "test.json")

    def test_duplicate_ids(self) -> None:
        case_id = uuid4()
        cases = [
            EvaluationCase(input="Case 1", id=case_id),
            EvaluationCase(input="Case 2", id=case_id),
        ]
        with pytest.raises(ValidationError) as exc_info:
            check_duplicate_case_ids(cases, "test.json")
        assert "duplicate case ID" in str(exc_info.value)
        assert exc_info.value.details["duplicates"][0]["case_id"] == str(case_id)

    def test_multiple_duplicates(self) -> None:
        id1 = uuid4()
        id2 = uuid4()
        cases = [
            EvaluationCase(input="Case 1", id=id1),
            EvaluationCase(input="Case 2", id=id1),
            EvaluationCase(input="Case 3", id=id2),
            EvaluationCase(input="Case 4", id=id2),
        ]
        with pytest.raises(ValidationError) as exc_info:
            check_duplicate_case_ids(cases, "test.json")
        assert len(exc_info.value.details["duplicates"]) == 2


class TestComputeDatasetHash:
    """Tests for compute_dataset_hash function."""

    def test_deterministic_hash(self) -> None:
        dataset = EvaluationDataset(
            name="test",
            cases=[
                EvaluationCase(input="Case 1", expected_output="Answer 1"),
                EvaluationCase(input="Case 2", expected_output="Answer 2"),
            ],
        )
        hash1 = compute_dataset_hash(dataset)
        hash2 = compute_dataset_hash(dataset)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self) -> None:
        dataset1 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
        )
        dataset2 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 2")],
        )
        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 != hash2

    def test_same_content_different_order_same_hash(self) -> None:
        case1 = EvaluationCase(input="Case 1", id=uuid4())
        case2 = EvaluationCase(input="Case 2", id=uuid4())

        dataset1 = EvaluationDataset(name="test", cases=[case1, case2])
        dataset2 = EvaluationDataset(name="test", cases=[case2, case1])

        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 == hash2

    def test_metadata_changes_hash(self) -> None:
        dataset1 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
            metadata={"key": "value1"},
        )
        dataset2 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
            metadata={"key": "value2"},
        )
        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 != hash2

    def test_tags_changes_hash(self) -> None:
        dataset1 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
            tags=["tag1"],
        )
        dataset2 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
            tags=["tag2"],
        )
        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 != hash2

    def test_version_changes_hash(self) -> None:
        dataset1 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
            version="1.0.0",
        )
        dataset2 = EvaluationDataset(
            name="test",
            cases=[EvaluationCase(input="Case 1")],
            version="2.0.0",
        )
        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 != hash2


class TestLoadJsonDataset:
    """Tests for load_json_dataset function."""

    def test_load_valid_json(self, tmp_path) -> None:
        dataset_file = tmp_path / "dataset.json"
        data = {
            "name": "test_dataset",
            "description": "Test dataset",
            "version": "1.0.0",
            "metadata": {"author": "test"},
            "tags": ["unit_test"],
            "cases": [
                {"input": "Case 1", "expected_output": "Answer 1"},
                {"input": "Case 2", "tags": ["tag1"]},
            ],
        }
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)

        assert dataset.name == "test_dataset"
        assert dataset.description == "Test dataset"
        assert dataset.version == "1.0.0"
        assert dataset.metadata["author"] == "test"
        assert "content_hash" in dataset.metadata
        assert dataset.tags == ["unit_test"]
        assert len(dataset.cases) == 2
        assert dataset.cases[0].input == "Case 1"
        assert dataset.cases[0].expected_output == "Answer 1"
        assert dataset.cases[1].tags == ["tag1"]
        assert "content_hash" in dataset.metadata

    def test_load_minimal_json(self, tmp_path) -> None:
        dataset_file = tmp_path / "minimal.json"
        data = {
            "name": "minimal",
            "cases": [{"input": "Single case"}],
        }
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)

        assert dataset.name == "minimal"
        assert len(dataset.cases) == 1
        assert dataset.cases[0].input == "Single case"

    def test_file_not_found(self) -> None:
        with pytest.raises(DatasetError) as exc_info:
            load_json_dataset("nonexistent.json")
        assert "not found" in str(exc_info.value)
        assert exc_info.value.details["error"] == "file_not_found"

    def test_invalid_json(self, tmp_path) -> None:
        dataset_file = tmp_path / "invalid.json"
        dataset_file.write_text("{ invalid json }")

        with pytest.raises(DatasetError) as exc_info:
            load_json_dataset(dataset_file)
        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.details["error"] == "json_decode"

    def test_missing_name_field(self, tmp_path) -> None:
        dataset_file = tmp_path / "no_name.json"
        data = {"cases": [{"input": "Case 1"}]}
        dataset_file.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        assert "missing required 'name' field" in str(exc_info.value)

    def test_missing_cases_field(self, tmp_path) -> None:
        dataset_file = tmp_path / "no_cases.json"
        data = {"name": "test"}
        dataset_file.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        assert "missing required 'cases' field" in str(exc_info.value)

    def test_cases_not_list(self, tmp_path) -> None:
        dataset_file = tmp_path / "bad_cases.json"
        data = {"name": "test", "cases": "not a list"}
        dataset_file.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        assert "cases' must be a list" in str(exc_info.value)

    def test_invalid_case_in_json(self, tmp_path) -> None:
        dataset_file = tmp_path / "bad_case.json"
        data = {"name": "test", "cases": [{"expected_output": "No input"}]}
        dataset_file.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        assert "Failed to validate" in str(exc_info.value)

    def test_duplicate_case_ids_in_json(self, tmp_path) -> None:
        case_id = str(uuid4())
        dataset_file = tmp_path / "duplicate.json"
        data = {
            "name": "test",
            "cases": [
                {"id": case_id, "input": "Case 1"},
                {"id": case_id, "input": "Case 2"},
            ],
        }
        dataset_file.write_text(json.dumps(data))

        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        assert "duplicate case ID" in str(exc_info.value)

    def test_empty_dataset_allowed_by_default(self, tmp_path) -> None:
        dataset_file = tmp_path / "empty.json"
        data = {"name": "empty", "cases": []}
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)
        assert len(dataset.cases) == 0

    def test_empty_dataset_not_allowed_with_policy(self, tmp_path) -> None:
        dataset_file = tmp_path / "empty.json"
        data = {"name": "empty", "cases": []}
        dataset_file.write_text(json.dumps(data))

        policy = DatasetLoadPolicy(allow_empty=False)
        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file, policy=policy)
        assert "empty dataset not allowed" in str(exc_info.value)

    def test_hash_not_computed_when_disabled(self, tmp_path) -> None:
        dataset_file = tmp_path / "nohash.json"
        data = {"name": "test", "cases": [{"input": "Case 1"}]}
        dataset_file.write_text(json.dumps(data))

        policy = DatasetLoadPolicy(require_hash=False)
        dataset = load_json_dataset(dataset_file, policy=policy)
        assert "content_hash" not in dataset.metadata

    def test_preserves_case_metadata_and_tags(self, tmp_path) -> None:
        dataset_file = tmp_path / "metadata.json"
        data = {
            "name": "test",
            "cases": [
                {
                    "input": "Case 1",
                    "metadata": {"source": "test"},
                    "tags": ["important", "regression"],
                }
            ],
        }
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)
        assert dataset.cases[0].metadata == {"source": "test"}
        assert dataset.cases[0].tags == ["important", "regression"]

    def test_preserves_expected_tool_calls(self, tmp_path) -> None:
        dataset_file = tmp_path / "tools.json"
        data = {
            "name": "test",
            "cases": [
                {
                    "input": "Case 1",
                    "expected_tool_calls": [
                        {"name": "search", "arguments": {"q": "test"}}
                    ],
                }
            ],
        }
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)
        assert len(dataset.cases[0].expected_tool_calls) == 1
        assert dataset.cases[0].expected_tool_calls[0].name == "search"
        assert dataset.cases[0].expected_tool_calls[0].arguments == {"q": "test"}

    def test_preserves_expected_trajectory(self, tmp_path) -> None:
        dataset_file = tmp_path / "trajectory.json"
        data = {
            "name": "test",
            "cases": [
                {
                    "input": "Case 1",
                    "expected_trajectory": {
                        "steps": [{"action": "search"}],
                        "description": "Search then answer",
                        "max_deviation": 0.1,
                    },
                }
            ],
        }
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)
        traj = dataset.cases[0].expected_trajectory
        assert traj is not None
        assert traj.steps == [{"action": "search"}]
        assert traj.description == "Search then answer"
        assert traj.max_deviation == 0.1

    def test_preserves_answerability(self, tmp_path) -> None:
        dataset_file = tmp_path / "answerability.json"
        data = {
            "name": "test",
            "cases": [
                {
                    "input": "Case 1",
                    "answerability": {
                        "is_answerable": False,
                        "confidence": 0.5,
                        "requires_tools": True,
                        "required_tools": ["search"],
                    },
                }
            ],
        }
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)
        ans = dataset.cases[0].answerability
        assert ans.is_answerable is False
        assert ans.confidence == 0.5
        assert ans.requires_tools is True
        assert ans.required_tools == ["search"]


class TestLoadJsonlDataset:
    """Tests for load_jsonl_dataset function."""

    def test_load_valid_jsonl(self, tmp_path) -> None:
        dataset_file = tmp_path / "dataset.jsonl"
        lines = [
            '{"input": "Case 1", "expected_output": "Answer 1"}',
            '{"input": "Case 2", "tags": ["tag1"]}',
        ]
        dataset_file.write_text("\n".join(lines))

        dataset = load_jsonl_dataset(dataset_file)

        assert dataset.name == "dataset"  # from filename
        assert len(dataset.cases) == 2
        assert dataset.cases[0].input == "Case 1"
        assert dataset.cases[0].expected_output == "Answer 1"
        assert dataset.cases[1].tags == ["tag1"]

    def test_load_jsonl_with_metadata(self, tmp_path) -> None:
        dataset_file = tmp_path / "dataset.jsonl"
        meta_line = json.dumps(
            {
                "_meta": {
                    "name": "my_dataset",
                    "description": "Test",
                    "version": "2.0.0",
                    "metadata": {"key": "value"},
                    "tags": ["meta_tag"],
                }
            }
        )
        lines = [
            meta_line,
            '{"input": "Case 1"}',
            '{"input": "Case 2"}',
        ]
        dataset_file.write_text("\n".join(lines))

        dataset = load_jsonl_dataset(dataset_file)

        assert dataset.name == "my_dataset"
        assert dataset.description == "Test"
        assert dataset.version == "2.0.0"
        assert dataset.metadata["key"] == "value"
        assert "content_hash" in dataset.metadata
        assert dataset.tags == ["meta_tag"]
        assert len(dataset.cases) == 2

    def test_file_not_found(self) -> None:
        with pytest.raises(DatasetError) as exc_info:
            load_jsonl_dataset("nonexistent.jsonl")
        assert "not found" in str(exc_info.value)

    def test_invalid_jsonl_line(self, tmp_path) -> None:
        dataset_file = tmp_path / "bad.jsonl"
        lines = [
            '{"input": "Case 1"}',
            '{ invalid json }',
        ]
        dataset_file.write_text("\n".join(lines))

        with pytest.raises(DatasetError) as exc_info:
            load_jsonl_dataset(dataset_file)
        assert "Invalid JSON on line 2" in str(exc_info.value)
        assert exc_info.value.details["line"] == 2

    def test_empty_lines_skipped(self, tmp_path) -> None:
        dataset_file = tmp_path / "empty_lines.jsonl"
        lines = [
            '{"input": "Case 1"}',
            "",
            "   ",
            '{"input": "Case 2"}',
        ]
        dataset_file.write_text("\n".join(lines))

        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 2

    def test_duplicate_ids_in_jsonl(self, tmp_path) -> None:
        case_id = str(uuid4())
        dataset_file = tmp_path / "dup.jsonl"
        lines = [
            f'{{"id": "{case_id}", "input": "Case 1"}}',
            f'{{"id": "{case_id}", "input": "Case 2"}}',
        ]
        dataset_file.write_text("\n".join(lines))

        with pytest.raises(ValidationError) as exc_info:
            load_jsonl_dataset(dataset_file)
        assert "duplicate case ID" in str(exc_info.value)

    def test_invalid_case_in_jsonl(self, tmp_path) -> None:
        dataset_file = tmp_path / "bad_case.jsonl"
        lines = [
            '{"input": "Case 1"}',
            '{"expected_output": "No input"}',
        ]
        dataset_file.write_text("\n".join(lines))

        with pytest.raises(ValidationError) as exc_info:
            load_jsonl_dataset(dataset_file)
        assert "Failed to validate" in str(exc_info.value)

    def test_empty_jsonl_allowed(self, tmp_path) -> None:
        dataset_file = tmp_path / "empty.jsonl"
        dataset_file.write_text("")

        dataset = load_jsonl_dataset(dataset_file)
        assert dataset.name == "empty"
        assert len(dataset.cases) == 0

    def test_jsonl_with_complex_cases(self, tmp_path) -> None:
        dataset_file = tmp_path / "complex.jsonl"
        lines = [
            json.dumps({
                "input": "Case 1",
                "expected_output": "Answer",
                "expected_tool_calls": [{"name": "tool1", "arguments": {"arg": "value"}}],
                "metadata": {"source": "test"},
                "tags": ["tag1"],
            }),
            json.dumps({
                "input": "Case 2",
                "expected_trajectory": {"steps": [{"action": "step1"}]},
                "answerability": {"is_answerable": True, "confidence": 0.9},
            }),
        ]
        dataset_file.write_text("\n".join(lines))

        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 2
        assert len(dataset.cases[0].expected_tool_calls) == 1
        assert dataset.cases[0].metadata == {"source": "test"}
        assert dataset.cases[1].expected_trajectory is not None
        assert dataset.cases[1].answerability.confidence == 0.9


class TestDatasetLoadPolicy:
    """Tests for DatasetLoadPolicy."""

    def test_default_policy(self) -> None:
        policy = DatasetLoadPolicy()
        assert policy.allow_empty is True
        assert policy.require_hash is True
        assert policy.strip_whitespace is True

    def test_custom_policy(self) -> None:
        policy = DatasetLoadPolicy(allow_empty=False, require_hash=False)
        assert policy.allow_empty is False
        assert policy.require_hash is False
        assert policy.strip_whitespace is True  # default


class TestLargeDataset:
    """Tests for loading larger datasets."""

    def test_load_100_cases(self, tmp_path) -> None:
        dataset_file = tmp_path / "large.json"
        cases = [{"input": f"Case {i}", "expected_output": f"Answer {i}"} for i in range(100)]
        data = {"name": "large", "cases": cases}
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(dataset_file)
        assert len(dataset.cases) == 100

    def test_load_1000_cases(self, tmp_path) -> None:
        dataset_file = tmp_path / "xlarge.jsonl"
        lines = [json.dumps({"input": f"Case {i}"}) for i in range(1000)]
        dataset_file.write_text("\n".join(lines))

        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 1000


class TestLoadFromStringPath:
    """Tests that loaders accept both string and Path objects."""

    def test_json_accepts_string_path(self, tmp_path) -> None:
        dataset_file = tmp_path / "test.json"
        data = {"name": "test", "cases": [{"input": "Case 1"}]}
        dataset_file.write_text(json.dumps(data))

        dataset = load_json_dataset(str(dataset_file))
        assert dataset.name == "test"

    def test_jsonl_accepts_string_path(self, tmp_path) -> None:
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"input": "Case 1"}')

        dataset = load_jsonl_dataset(str(dataset_file))
        assert len(dataset.cases) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_json_with_explicit_id(self, tmp_path) -> None:
        """Test loading JSON with explicit dataset ID."""
        dataset_file = tmp_path / "with_id.json"
        explicit_id = str(uuid4())
        data = {
            "id": explicit_id,
            "name": "test",
            "cases": [{"input": "Case 1"}],
        }
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        dataset = load_json_dataset(dataset_file)
        assert str(dataset.id) == explicit_id

    def test_jsonl_meta_on_non_first_line(self, tmp_path) -> None:
        """Test JSONL where _meta appears on line 2 (treated as case, fails validation).

        The _meta key is only recognized on the first line. On subsequent lines,
        it's treated as a regular case and will fail validation if it doesn't
        have the required "input" field.
        """
        dataset_file = tmp_path / "meta_line2.jsonl"
        lines = [
            '{"input": "Case 1"}',
            '{"_meta": {"name": "should_not_be_used"}}',
            '{"input": "Case 2"}',
        ]
        dataset_file.write_text("\n".join(lines), encoding="utf-8")

        # Should fail validation because line 2 doesn't have "input" field
        with pytest.raises(ValidationError) as exc_info:
            load_jsonl_dataset(dataset_file)
        assert "input" in str(exc_info.value).lower()
        assert "field required" in str(exc_info.value).lower()

    def test_unicode_content(self, tmp_path) -> None:
        """Test loading dataset with Unicode content."""
        dataset_file = tmp_path / "unicode.json"
        data = {
            "name": "unicode_test",
            "cases": [
                {"input": "测试用例", "expected_output": "答案"},
                {"input": "café", "expected_output": "café"},
                {"input": "🤖 AI", "expected_output": "🤖"},
            ],
        }
        dataset_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        dataset = load_json_dataset(dataset_file)
        assert len(dataset.cases) == 3
        assert dataset.cases[0].input == "测试用例"
        assert dataset.cases[0].expected_output == "答案"
        assert dataset.cases[1].input == "café"
        assert dataset.cases[2].input == "🤖 AI"

    def test_unicode_jsonl(self, tmp_path) -> None:
        """Test loading JSONL with Unicode content."""
        dataset_file = tmp_path / "unicode.jsonl"
        lines = [
            json.dumps({"input": "测试用例", "expected_output": "答案"}, ensure_ascii=False),
            json.dumps({"input": "café", "expected_output": "café"}, ensure_ascii=False),
        ]
        dataset_file.write_text("\n".join(lines), encoding="utf-8")

        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 2
        assert dataset.cases[0].input == "测试用例"
        assert dataset.cases[1].input == "café"

    def test_case_with_null_optional_fields(self, tmp_path) -> None:
        """Test case with explicit null values for optional fields.

        Pydantic rejects None for fields with dict/list type annotations,
        so this test verifies that validation fails appropriately.
        """
        dataset_file = tmp_path / "null_fields.json"
        data = {
            "name": "test",
            "cases": [
                {
                    "input": "Case 1",
                    "expected_output": None,
                    "metadata": None,
                    "tags": None,
                }
            ],
        }
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        # Should fail validation because None is not valid for dict/list fields
        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        error_str = str(exc_info.value).lower()
        assert "metadata" in error_str or "tags" in error_str
        assert "dict" in error_str or "list" in error_str

    def test_jsonl_trailing_newline(self, tmp_path) -> None:
        """Test JSONL with trailing newline."""
        dataset_file = tmp_path / "trailing.jsonl"
        lines = ['{"input": "Case 1"}', '{"input": "Case 2"}', ""]
        dataset_file.write_text("\n".join(lines))

        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 2

    def test_very_long_jsonl_line(self, tmp_path) -> None:
        """Test JSONL with a very long line."""
        dataset_file = tmp_path / "long_line.jsonl"
        long_input = "x" * 100000  # 100KB line
        line = json.dumps({"input": long_input})
        dataset_file.write_text(line)

        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 1
        assert len(dataset.cases[0].input) == 100000

    def test_hash_with_unicode_content(self) -> None:
        """Test that hash is deterministic with Unicode content."""
        # Use explicit UUID to ensure deterministic hash
        test_id = uuid4()
        dataset1 = EvaluationDataset(
            name="test",
            cases=[
                EvaluationCase(id=test_id, input="测试", expected_output="答案"),
            ],
        )
        dataset2 = EvaluationDataset(
            name="test",
            cases=[
                EvaluationCase(id=test_id, input="测试", expected_output="答案"),
            ],
        )
        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_hash_with_special_characters(self) -> None:
        """Test hash with special JSON characters."""
        test_id = uuid4()
        case_input = '{"key": "value"}'
        case_output = '{"nested": true}'
        dataset1 = EvaluationDataset(
            name="test",
            cases=[
                EvaluationCase(id=test_id, input=case_input, expected_output=case_output),
            ],
        )
        dataset2 = EvaluationDataset(
            name="test",
            cases=[
                EvaluationCase(id=test_id, input=case_input, expected_output=case_output),
            ],
        )
        hash1 = compute_dataset_hash(dataset1)
        hash2 = compute_dataset_hash(dataset2)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_strip_whitespace_policy_not_implemented(self, tmp_path) -> None:
        """Test that strip_whitespace policy is accepted but currently has no effect.

        This test documents the current behavior - the policy parameter exists
        but the implementation doesn't use it yet.
        """
        dataset_file = tmp_path / "whitespace.json"
        data = {
            "name": "test",
            "cases": [{"input": "  Case with spaces  "}],
        }
        dataset_file.write_text(json.dumps(data))

        # With strip_whitespace=True (default)
        policy_on = DatasetLoadPolicy(strip_whitespace=True)
        dataset_on = load_json_dataset(dataset_file, policy=policy_on)

        # With strip_whitespace=False
        policy_off = DatasetLoadPolicy(strip_whitespace=False)
        dataset_off = load_json_dataset(dataset_file, policy=policy_off)

        # Currently both behave the same (policy not implemented)
        assert dataset_on.cases[0].input == dataset_off.cases[0].input
        assert dataset_on.cases[0].input == "  Case with spaces  "


class TestErrorHandling:
    """Tests for specific error handling scenarios."""

    def test_json_invalid_id_format(self, tmp_path) -> None:
        """Test JSON with invalid UUID format for dataset ID."""
        dataset_file = tmp_path / "bad_id.json"
        data = {
            "id": "not-a-valid-uuid",
            "name": "test",
            "cases": [{"input": "Case 1"}],
        }
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(Exception) as exc_info:
            load_json_dataset(dataset_file)
        # UUID validation error should mention uuid or id
        error_str = str(exc_info.value).lower()
        assert "uuid" in error_str or "id" in error_str

    def test_case_invalid_id_format(self, tmp_path) -> None:
        """Test case with invalid UUID format."""
        dataset_file = tmp_path / "bad_case_id.json"
        data = {
            "name": "test",
            "cases": [{"id": "not-a-valid-uuid", "input": "Case 1"}],
        }
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValidationError) as exc_info:
            load_json_dataset(dataset_file)
        error_str = str(exc_info.value).lower()
        assert "uuid" in error_str or "id" in error_str

    def test_jsonl_invalid_meta_format(self, tmp_path) -> None:
        """Test JSONL with invalid _meta format (not a dict).

        When _meta is not a dict, it's not recognized as metadata and
        the record is treated as a case, which fails validation due to
        missing required "input" field.
        """
        dataset_file = tmp_path / "bad_meta.jsonl"
        lines = [
            '{"_meta": "not-an-object"}',
            '{"input": "Case 1"}',
        ]
        dataset_file.write_text("\n".join(lines), encoding="utf-8")

        # Should fail validation because first line is treated as case without "input"
        with pytest.raises(Exception) as exc_info:
            load_jsonl_dataset(dataset_file)
        error_str = str(exc_info.value).lower()
        assert "input" in error_str or "field required" in error_str

    def test_io_error_on_read(self, tmp_path) -> None:
        """Test handling of IO errors during file read.

        Note: This test is platform-dependent and may not work on all systems.
        On Windows, permission errors are harder to trigger reliably.
        """
        import sys
        if sys.platform == "win32":
            pytest.skip("File permission tests not reliable on Windows")

        dataset_file = tmp_path / "permission.json"
        data = {"name": "test", "cases": [{"input": "Case 1"}]}
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        # Make file unreadable (may not work on Windows or in all environments)
        try:
            dataset_file.chmod(0o000)
            with pytest.raises(DatasetError) as exc_info:
                load_json_dataset(dataset_file)
            assert exc_info.value.details["error"] == "io_error"
        except (PermissionError, OSError):
            # If we can't change permissions, skip the test
            pytest.skip("Cannot test file permissions in this environment")
        finally:
            with contextlib.suppress(Exception):
                dataset_file.chmod(0o644)


class TestPerformance:
    """Tests for performance characteristics."""

    def test_large_dataset_hash_performance(self) -> None:
        """Test that hashing a large dataset completes in reasonable time."""
        import time

        # Create dataset with 1000 cases
        cases = [
            EvaluationCase(
                input=f"Case {i}",
                expected_output=f"Answer {i}",
                metadata={"index": i},
                tags=[f"tag{i % 10}"],
            )
            for i in range(1000)
        ]
        dataset = EvaluationDataset(name="perf_test", cases=cases)

        start = time.perf_counter()
        hash1 = compute_dataset_hash(dataset)
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0
        assert len(hash1) == 64

    def test_jsonl_streaming_memory(self, tmp_path) -> None:
        """Test that JSONL loading doesn't load entire file into memory at once.

        This is a basic test - we just verify it works with a larger file.
        True streaming verification would require memory profiling.
        """
        dataset_file = tmp_path / "streaming.jsonl"
        # Use valid fields that exist in the model (input, expected_output, metadata, tags)
        case_data = {"input": "Case", "expected_output": "x" * 1000}
        lines = [json.dumps({**case_data, "input": f"Case {i}"}) for i in range(5000)]
        dataset_file.write_text("\n".join(lines), encoding="utf-8")

        # Should handle 5000 cases with 1KB each without issues
        dataset = load_jsonl_dataset(dataset_file)
        assert len(dataset.cases) == 5000
