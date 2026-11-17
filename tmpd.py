p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
# add fields to CLIConfig
s=s.replace(
    "    experiment_name: str | None = None\n",
    "    experiment_name: str | None = None\n    evaluators: list[str] = []\n    report_output_dir: str = \".\"\n    json_report_filename: str = \"report.json\"\n    markdown_report_filename: str = \"report.md\"\n",
)
# add evaluator mapping and import
s=s.replace(
    "from eval_harness.runner.experiment import ExperimentRunner",
    "from eval_harness.runner.experiment import ExperimentRunner\nfrom eval_harness.evaluators import *",
)
# update run_experiment
old='''def run_experiment(cfg: "CLIConfig"):
    dataset = load_dataset(cfg)
    system = load_system_factory(cfg.system_factory, cfg.system_config)
    runner = ExperimentRunner(system)
    result = runner.run(dataset)
    return cfg, dataset, result'''
new='''EVALUATOR_MAP = {
    "exact_match": "ExactMatchEvaluator",
}

def _make_evaluators(names):
    import eval_harness.evaluators as ev
    out = []
    for n in names:
        cls = getattr(ev, n, None)
        if cls is None:
            # try direct class name
            cls = getattr(ev, n, None)
        if cls is None:
            raise ValueError(f"unknown evaluator: {n}")
        out.append(cls())
    return out

def run_experiment(cfg: "CLIConfig"):
    dataset = load_dataset(cfg)
    system = load_system_factory(cfg.system_factory, cfg.system_config)
    evaluators = _make_evaluators(cfg.evaluators)
    runner = ExperimentRunner(system, evaluators=evaluators)
    result = runner.run_and_evaluate(dataset, config_name=cfg.experiment_name or "cli")
    # reports
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
s=s.replace(old,new)
open(p,"w",encoding="utf-8").write(s)
print("done")
