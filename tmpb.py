p="eval_harness/cli/main.py"
s=open(p,encoding="utf-8").read()
new='''from __future__ import annotations

import argparse
import sys

from eval_harness.cli.config import CLIConfig, run_experiment

EXIT_SUCCESS = 0
EXIT_GATE_FAIL = 1
EXIT_CONFIG_ERROR = 2
EXIT_EXECUTION_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval-harness")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            cfg, dataset, result = run_experiment(CLIConfig.from_file(args.config))
        except Exception as exc:
            print(f"Error: {exc}")
            return EXIT_CONFIG_ERROR
        print(f"Experiment: {cfg.experiment_name}")
        print(f"Dataset cases: {len(dataset.cases) if hasattr(dataset, 'cases') else 'n/a'}")
        print(f"Total: {result.total_cases} Success: {result.successful_cases} Failed: {result.failed_cases}")
        print(f"Status: {result.status}")
        return EXIT_SUCCESS
    return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    sys.exit(main())
'''
open(p,"w",encoding="utf-8").write(new)
print("done")
