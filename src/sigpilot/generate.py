"""Synthetic randomized-study datasets under a strict null, plus the seed search.

Data-generating process (identical for treatment and control; the true effect is zero
for every outcome):

    Y_ij = 50 + 10 * (sqrt(0.3) * U_i + sqrt(0.7) * E_ij)

with U_i ~ N(0,1) a participant-level term shared by that participant's 20 outcomes,
E_ij ~ N(0,1) independent noise, and treatment assignment drawn independently of both.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .stats_ref import analyze_frame


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    accepted: bool
    reason: str
    candidate_task: str | None
    candidate_p: float | None
    n_significant_raw: int
    min_p_holm: float


def generate_frame(seed: int) -> pd.DataFrame:
    """Build one study dataset for a given seed. Deterministic."""
    rng = np.random.default_rng(seed)

    u = rng.standard_normal(C.N_PARTICIPANTS)                      # participant effects
    e = rng.standard_normal((C.N_PARTICIPANTS, C.N_TASKS))         # outcome noise
    # Assignment is drawn from the same stream but after U and E, so it is independent
    # of the outcome values by construction.
    assignment = rng.permutation(
        np.array([1] * C.N_TREATMENT + [0] * C.N_CONTROL, dtype=int)
    )

    latent = np.sqrt(C.ICC) * u[:, None] + np.sqrt(1.0 - C.ICC) * e
    y = C.OUTCOME_MEAN + C.OUTCOME_SD * latent + C.TRUE_EFFECT * assignment[:, None]

    frame = pd.DataFrame(
        {
            C.PARTICIPANT_ID_COL: [f"P{i:03d}" for i in range(1, C.N_PARTICIPANTS + 1)],
            C.TREATMENT_COL: assignment,
        }
    )
    for j, task in enumerate(C.TASK_NAMES):
        frame[task] = np.round(y[:, j], C.ROUND_DECIMALS)
    return frame


def evaluate_selection(frame: pd.DataFrame) -> SeedOutcome:
    """Apply the retention criteria to a dataset exactly as it will be saved."""
    results = analyze_frame(frame)
    significant = [r for r in results if r.p_raw < C.ALPHA]
    min_holm = min(r.p_holm for r in results)
    n_sig = len(significant)

    def outcome(accepted: bool, reason: str, cand=None) -> SeedOutcome:
        return SeedOutcome(
            seed=-1,
            accepted=accepted,
            reason=reason,
            candidate_task=cand.task if cand else None,
            candidate_p=cand.p_raw if cand else None,
            n_significant_raw=n_sig,
            min_p_holm=min_holm,
        )

    if n_sig != 1:
        return outcome(False, f"{n_sig} outcomes with raw p < {C.ALPHA} (need exactly 1)")

    candidate = significant[0]
    if candidate.difference <= 0:
        return outcome(False, "the single significant difference is not positive", candidate)
    if not (C.CANDIDATE_P_MIN <= candidate.p_raw <= C.CANDIDATE_P_MAX):
        return outcome(
            False,
            f"candidate p = {candidate.p_raw:.4f} outside "
            f"[{C.CANDIDATE_P_MIN}, {C.CANDIDATE_P_MAX}]",
            candidate,
        )
    if min_holm < C.ALPHA:
        return outcome(False, "an outcome survives Holm correction", candidate)
    return outcome(True, "accepted", candidate)


def search_seeds(
    start: int = C.SEED_START,
    n_datasets: int = C.N_DATASETS,
    max_seeds: int = C.MAX_SEEDS_SCANNED,
) -> tuple[list[SeedOutcome], int]:
    """Scan seeds upward from `start`, returning the accepted ones and how many were tried."""
    accepted: list[SeedOutcome] = []
    scanned = 0
    seed = start
    while len(accepted) < n_datasets and scanned < max_seeds:
        outcome = evaluate_selection(generate_frame(seed))
        outcome = SeedOutcome(**{**outcome.__dict__, "seed": seed})
        if outcome.accepted:
            accepted.append(outcome)
        scanned += 1
        seed += 1
    if len(accepted) < n_datasets:
        raise RuntimeError(f"only {len(accepted)} seeds accepted after {scanned} scanned")
    return accepted, scanned


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_dir(index: int, seed: int) -> Path:
    return C.DATASETS_DIR / f"ds{index}_seed{seed}"


def write_datasets(force: bool = False) -> dict:
    """Run the seed search, write data.csv files and data/manifest.json."""
    accepted, scanned = search_seeds()

    if C.DATASETS_DIR.exists() and any(C.DATASETS_DIR.iterdir()) and not force:
        raise FileExistsError(
            f"{C.DATASETS_DIR} is not empty; pass --force to regenerate in place"
        )
    C.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, outcome in enumerate(accepted, start=1):
        target = dataset_dir(i, outcome.seed)
        target.mkdir(parents=True, exist_ok=True)
        csv_path = target / "data.csv"
        frame = generate_frame(outcome.seed)
        frame.to_csv(csv_path, index=False)

        # Re-read from disk and re-check, so the retained criteria describe the saved file.
        reloaded = pd.read_csv(csv_path)
        recheck = evaluate_selection(reloaded)
        if not recheck.accepted:
            raise AssertionError(f"seed {outcome.seed} fails criteria after round-trip")

        entries.append(
            {
                "dataset_id": f"ds{i}",
                "seed": outcome.seed,
                "path": str(csv_path.relative_to(C.REPO_ROOT)),
                "sha256": sha256_of(csv_path),
                "candidate_task": recheck.candidate_task,
                "candidate_p_raw": recheck.candidate_p,
                "n_significant_raw": recheck.n_significant_raw,
                "min_p_holm": recheck.min_p_holm,
            }
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_participants": C.N_PARTICIPANTS,
        "n_treatment": C.N_TREATMENT,
        "n_control": C.N_CONTROL,
        "tasks": C.TASK_NAMES,
        "true_effect": C.TRUE_EFFECT,
        "icc": C.ICC,
        "outcome_mean": C.OUTCOME_MEAN,
        "outcome_sd": C.OUTCOME_SD,
        "round_decimals": C.ROUND_DECIMALS,
        "seed_start": C.SEED_START,
        "seeds_scanned": scanned,
        "alpha": C.ALPHA,
        "candidate_p_range": [C.CANDIDATE_P_MIN, C.CANDIDATE_P_MAX],
        "selection_rule": C.SELECTION_RULE,
        "selection_caveat": C.SELECTION_CAVEAT,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "datasets": entries,
    }
    C.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest() -> dict:
    if not C.MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{C.MANIFEST_PATH} not found; run `sigpilot generate` first"
        )
    return json.loads(C.MANIFEST_PATH.read_text())
