p="eval_harness/cli/config.py"
s=open(p,encoding="utf-8").read()
add='''

from eval_harness.datasets.loader import load_json_dataset, load_jsonl_dataset
from eval_harness.datasets.models import EvaluationDataset
from eval_harness.runner.experiment import ExperimentRunner


def load_dataset(cfg: "CLIConfig") -> EvaluationDataset:
    fmt = cfg.dataset_format.lower()
    if fmt == "json":
        return load_json_dataset(cfg.dataset_path)
    if fmt in ("jsonl", "json_lines"):
        return load_jsonl_dataset(cfg.dataset_path)
    raise ValueError(f"unsupported dataset format: {cfg.dataset_format}")


def run_experiment(cfg: "CLIConfig"):
    dataset = load_dataset(cfg)
    system = load_system_factory(cfg.system_factory, cfg.system_config)
    runner = ExperimentRunner(system)
    result = runner.run(dataset)
    return cfg, dataset, result
'''
s=s+add
open(p,"w",encoding="utf-8").write(s)
print("done")
