from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from eval_harness.runner.system import SystemUnderEvaluation
from eval_harness.datasets.loader import load_json_dataset, load_jsonl_dataset
from eval_harness.datasets.models import EvaluationDataset
from eval_harness.runner.experiment import ExperimentRunner


class CLIConfig(BaseModel):
    system_factory: str
    system_config: dict[str, Any] = Field(default_factory=dict)
    dataset_path: str
    dataset_format: str
    experiment_name: str | None = None
    evaluators: list[str] = Field(default_factory=list)
    baseline_path: str | None = None
    quality_gate: dict[str, Any] = Field(default_factory=dict)
    save_baseline_path: str | None = None
    save_baseline: bool = False
    overwrite_baseline: bool = False
    report_output_dir: str = "."
    json_report_filename: str = "report.json"
    markdown_report_filename: str = "report.md"

    model_config = {"extra": "forbid"}

    @classmethod
    def from_file(cls, path: str | Path) -> "CLIConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.model_validate(data)
        base = Path(path).resolve().parent
        dp = Path(cfg.dataset_path)
        if not dp.is_absolute():
            cfg.dataset_path = str((base / dp).resolve())
        return cfg


def load_system_factory(import_path: str, system_config: dict[str, Any]) -> SystemUnderEvaluation:
    if ":" not in import_path:
        raise ValueError("import path must be module:factory")
    module_name, factory_name = import_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(f"cannot import module {module_name}: {exc}") from exc
    try:
        factory = getattr(module, factory_name)
    except AttributeError as exc:
        raise AttributeError(f"factory {factory_name} not found") from exc
    if not callable(factory):
        raise TypeError("factory is not callable")
    try:
        system = factory(**system_config)
    except Exception as exc:
        raise RuntimeError(f"factory raised: {exc}") from exc
    if not (hasattr(system, "run") and callable(getattr(system, "run"))):
        raise TypeError("factory did not return SystemUnderEvaluation")
    from typing import cast
    return cast(SystemUnderEvaluation, system)


def load_dataset(cfg: "CLIConfig") -> EvaluationDataset:
    fmt = cfg.dataset_format.lower()
    if fmt == "json":
        return load_json_dataset(cfg.dataset_path)
    if fmt in ("jsonl", "json_lines"):
        return load_jsonl_dataset(cfg.dataset_path)
    raise ValueError(f"unsupported dataset format: {cfg.dataset_format}")


def _make_evaluators(names: list[str]) -> list[Any]:
    import eval_harness.evaluators as ev
    out: list[Any] = []
    for n in names:
        cls = getattr(ev, n, None)
        if cls is None:
            raise ValueError(f"unknown evaluator: {n}")
        out.append(cls())
    return out


def run_experiment(cfg: "CLIConfig", baseline_result: Any = None) -> tuple["CLIConfig", EvaluationDataset, Any, Any, str, str]:
    dataset = load_dataset(cfg)
    system = load_system_factory(cfg.system_factory, cfg.system_config)
    evaluators = _make_evaluators(cfg.evaluators)
    runner = ExperimentRunner(system, evaluators=evaluators)
    result = runner.run_and_evaluate(dataset, config_name=cfg.experiment_name or "cli")
    from eval_harness.baselines.comparator import BaselineComparator
    from eval_harness.regression.models import QualityGateConfig
    from eval_harness.regression.tester import RegressionTester
    comparison = None
    loaded_baseline = baseline_result
    if loaded_baseline is None and cfg.baseline_path:
        from eval_harness.baselines.artifacts import load_baseline
        loaded_baseline = load_baseline(cfg.baseline_path)
    if loaded_baseline is not None:
        comparison = BaselineComparator().compare(loaded_baseline, result)
    gate_config = QualityGateConfig.model_validate(cfg.quality_gate) if cfg.quality_gate else QualityGateConfig()
    gate_result = RegressionTester(gate_config).evaluate(result, loaded_baseline)
    from pathlib import Path
    from eval_harness.reporting.json_report import JSONReportGenerator
    from eval_harness.reporting.markdown_report import MarkdownReportGenerator
    outdir = Path(cfg.report_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / cfg.json_report_filename
    md_path = outdir / cfg.markdown_report_filename
    JSONReportGenerator().write_json(result, str(json_path))
    MarkdownReportGenerator().write_markdown(result, str(md_path))
    if cfg.save_baseline and cfg.save_baseline_path and gate_result.passed:
        from eval_harness.baselines.artifacts import save_baseline
        save_baseline(result, cfg.save_baseline_path, overwrite=cfg.overwrite_baseline)
    return cfg, dataset, result, gate_result, str(json_path), str(md_path)
