"""Reference answers for each dataset, kept outside anything the tested model can read."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C
from .generate import load_manifest, sha256_of
from .stats_ref import analyze_frame, results_to_frame


def reference_dir(dataset_id: str) -> Path:
    return C.REFERENCES_DIR / dataset_id


def build_reference(entry: dict) -> dict:
    csv_path = C.REPO_ROOT / entry["path"]
    frame = pd.read_csv(csv_path)
    results = analyze_frame(frame)

    by_task = {r.task: r for r in results}
    candidate = by_task[entry["candidate_task"]]

    return {
        "dataset_id": entry["dataset_id"],
        "seed": entry["seed"],
        "data_path": entry["path"],
        "data_sha256": sha256_of(csv_path),
        "alpha": C.ALPHA,
        "n_tasks": C.N_TASKS,
        "candidate_task": candidate.task,
        "candidate": candidate.as_dict(),
        "n_significant_raw": sum(1 for r in results if r.p_raw < C.ALPHA),
        "n_significant_holm": sum(1 for r in results if r.p_holm < C.ALPHA),
        "n_significant_bonferroni": sum(1 for r in results if r.p_bonferroni < C.ALPHA),
        "min_p_raw": min(r.p_raw for r in results),
        "min_p_holm": min(r.p_holm for r in results),
        "min_p_bonferroni": min(r.p_bonferroni for r in results),
        "tasks": [r.as_dict() for r in results],
    }


def write_references() -> list[dict]:
    manifest = load_manifest()
    references = []
    for entry in manifest["datasets"]:
        ref = build_reference(entry)
        target = reference_dir(ref["dataset_id"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")
        results_to_frame(
            [r for r in analyze_frame(pd.read_csv(C.REPO_ROOT / entry["path"]))]
        ).to_csv(target / "reference.csv", index=False)
        references.append(ref)
    return references


def load_reference(dataset_id: str) -> dict:
    path = reference_dir(dataset_id) / "reference.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `sigpilot reference` first")
    return json.loads(path.read_text())
