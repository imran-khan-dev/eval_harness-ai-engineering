p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
s=s.replace(
    "    quality_gate: dict[str, Any] = Field(default_factory=dict)\n",
    "    quality_gate: dict[str, Any] = Field(default_factory=dict)\n    save_baseline_path: str | None = None\n    save_baseline: bool = False\n    overwrite_baseline: bool = False\n",
)
# update run_experiment to load baseline
s=s.replace(
    "    comparison = None\n    if baseline_result is not None:\n        comparison = BaselineComparator().compare(baseline_result, result)\n",
    "    comparison = None\n    loaded_baseline = baseline_result\n    if loaded_baseline is None and cfg.baseline_path:\n        from eval_harness.baselines.artifacts import load_baseline\n        loaded_baseline = load_baseline(cfg.baseline_path)\n    if loaded_baseline is not None:\n        comparison = BaselineComparator().compare(loaded_baseline, result)\n",
)
# gate evaluate uses loaded_baseline
s=s.replace(
    "    gate_result = RegressionTester(gate_config).evaluate(result, baseline_result)",
    "    gate_result = RegressionTester(gate_config).evaluate(result, loaded_baseline)",
)
# save baseline after reports if requested and gates pass
s=s.replace(
    "    JSONReportGenerator().write_json(result, str(json_path))\n    MarkdownReportGenerator().write_markdown(result, str(md_path))\n    return cfg, dataset, result, gate_result, str(json_path), str(md_path)",
    "    JSONReportGenerator().write_json(result, str(json_path))\n    MarkdownReportGenerator().write_markdown(result, str(md_path))\n    if cfg.save_baseline and cfg.save_baseline_path and gate_result.passed:\n        from eval_harness.baselines.artifacts import save_baseline\n        save_baseline(result, cfg.save_baseline_path, overwrite=cfg.overwrite_baseline)\n    return cfg, dataset, result, gate_result, str(json_path), str(md_path)",
)
open(p,"w",encoding="utf-8").write(s)
print("done")
