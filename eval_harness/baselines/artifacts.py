from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_harness.metrics.models import ExperimentResult

ARTIFACT_VERSION = 1


def save_baseline(result: ExperimentResult, path: str | Path, overwrite: bool = False) -> None:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"baseline already exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": ARTIFACT_VERSION, "experiment": result.model_dump(mode="json")}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def load_baseline(path: str | Path) -> ExperimentResult:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError("invalid baseline artifact")
    if data["version"] != ARTIFACT_VERSION:
        raise ValueError(f"unsupported baseline artifact version: {data['version']}")
    return ExperimentResult.model_validate(data["experiment"])
