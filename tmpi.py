p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
old='''    comparison = None
    loaded_baseline = baseline_result
    if loaded_baseline is None and cfg.baseline_path:
        from eval_harness.baselines.artifacts import load_baseline
        loaded_baseline = load_baseline(cfg.baseline_path)
    if loaded_baseline is not None:
        comparison = BaselineComparator().compare(loaded_baseline, result)
    if loaded_baseline is not None:
        comparison = BaselineComparator().compare(loaded_baseline, result)
'''
# The above may not match; do targeted replacement
s=s.replace(
"""    loaded_baseline = baseline_result
    if loaded_baseline is None and cfg.baseline_path:
        from eval_harness.baselines.artifacts import load_baseline
        loaded_baseline = load_baseline(cfg.baseline_path)
    if loaded_baseline is not None:
        comparison = BaselineComparator().compare(loaded_baseline, result)
""",
"""    loaded_baseline = baseline_result
    if loaded_baseline is None and cfg.baseline_path:
        from eval_harness.baselines.artifacts import load_baseline
        loaded_baseline = load_baseline(cfg.baseline_path)
""")
open(p,"w",encoding="utf-8").write(s)
print("done")
