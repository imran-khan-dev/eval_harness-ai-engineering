p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
# add fields
s=s.replace(
    "    evaluators: list[str] = Field(default_factory=list)\n",
    "    evaluators: list[str] = Field(default_factory=list)\n    baseline_path: str | None = None\n    quality_gate: dict[str, Any] = Field(default_factory=dict)\n",
)
# update run_experiment signature and logic
old='''def run_experiment(cfg: "CLIConfig") -> tuple["CLIConfig", EvaluationDataset, Any, str, str]:
    dataset = load_dataset(cfg)
    system = load_system_factory(cfg.system_factory, cfg.system_config)
    evaluators = _make_evaluators(cfg.evaluators)
    runner = ExperimentRunner(system, evaluators=evaluators)
    result = runner.run_and_evaluate(dataset, config_name=cfg.experiment_name or "cli")
    from pathlib import Path
    from eval_harness.reporting.json_report import JSONReportGenerator
    from eval_harness.reporting.markdown_report import MarkdownReportGenerator
    outdir = Path(cfg.report_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / cfg.json_report_filename
    md_path = outdir / cfg.markdown_report_filename
    JSONReportGenerator().write_json(result, str(json_path))
    MarkdownReportGenerator().write_markdown(result, str(md_path))
    return cfg, dataset, result, str(json_path), str(md_path)'''
new='''def run_experiment(cfg: "CLIConfig", baseline_result: Any = None) -> tuple["CLIConfig", EvaluationDataset, Any, Any, str, str]:
    dataset = load_dataset(cfg)
    system = load_system_factory(cfg.system_factory, cfg.system_config)
    evaluators = _make_evaluators(cfg.evaluators)
    runner = ExperimentRunner(system, evaluators=evaluators)
    result = runner.run_and_evaluate(dataset, config_name=cfg.experiment_name or "cli")
    from eval_harness.baselines.comparator import BaselineComparator
    from eval_harness.regression.models import QualityGateConfig
    from eval_harness.regression.tester import RegressionTester
    comparison = None
    if baseline_result is not None:
        comparison = BaselineComparator().compare(baseline_result, result)
    gate_config = QualityGateConfig.model_validate(cfg.quality_gate) if cfg.quality_gate else QualityGateConfig()
    gate_result = RegressionTester(gate_config).evaluate(result, baseline_result)
    from pathlib import Path
    from eval_harness.reporting.json_report import JSONReportGenerator
    from eval_harness.reporting.markdown_report import MarkdownReportGenerator
    outdir = Path(cfg.report_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / cfg.json_report_filename
    md_path = outdir / cfg.markdown_report_filename
    JSONReportGenerator().write_json(result, str(json_path))
    MarkdownReportGenerator().write_markdown(result, str(md_path))
    return cfg, dataset, result, gate_result, str(json_path), str(md_path)'''
s=s.replace(old,new)
open(p,"w",encoding="utf-8").write(s)
print("done")
