"""The datasets are what the manifest says they are, and regenerate identically."""

from __future__ import annotations

import pandas as pd
import pytest

from sigpilot import config as C
from sigpilot.generate import evaluate_selection, generate_frame, load_manifest, sha256_of

MANIFEST = load_manifest()


@pytest.fixture(params=MANIFEST["datasets"], ids=lambda e: e["dataset_id"])
def entry(request):
    return request.param


@pytest.fixture
def frame(entry):
    return pd.read_csv(C.REPO_ROOT / entry["path"])


def test_manifest_has_four_datasets_from_seed_1000():
    assert len(MANIFEST["datasets"]) == C.N_DATASETS
    assert MANIFEST["seed_start"] == 1000
    seeds = [e["seed"] for e in MANIFEST["datasets"]]
    assert seeds == sorted(seeds) and seeds[0] >= 1000


def test_structure(frame):
    assert list(frame.columns) == [C.PARTICIPANT_ID_COL, C.TREATMENT_COL] + C.TASK_NAMES
    assert len(frame) == C.N_PARTICIPANTS
    assert frame[C.PARTICIPANT_ID_COL].is_unique
    assert frame[C.PARTICIPANT_ID_COL].iloc[0] == "P001"


def test_group_sizes_and_completeness(frame):
    counts = frame[C.TREATMENT_COL].value_counts().to_dict()
    assert counts == {1: C.N_TREATMENT, 0: C.N_CONTROL}
    assert frame.isna().sum().sum() == 0
    assert frame[C.TASK_NAMES].abs().max().max() < 200  # no absurd values


def test_regeneration_is_byte_identical(entry):
    saved = C.REPO_ROOT / entry["path"]
    regenerated = generate_frame(entry["seed"])
    on_disk = pd.read_csv(saved)
    pd.testing.assert_frame_equal(regenerated, on_disk.astype(regenerated.dtypes.to_dict()))
    assert sha256_of(saved) == entry["sha256"]


def test_selection_criteria_hold_on_saved_file(frame, entry):
    outcome = evaluate_selection(frame)
    assert outcome.accepted, outcome.reason
    assert outcome.n_significant_raw == 1
    assert outcome.candidate_task == entry["candidate_task"]
    assert C.CANDIDATE_P_MIN <= outcome.candidate_p <= C.CANDIDATE_P_MAX
    assert outcome.min_p_holm >= C.ALPHA


def test_assignment_is_independent_of_outcomes():
    """A large screen: the null holds on average, so differences centre on zero."""
    diffs = []
    for seed in range(5000, 5100):
        f = generate_frame(seed)
        treated = f.loc[f[C.TREATMENT_COL] == 1, C.TASK_NAMES].mean()
        control = f.loc[f[C.TREATMENT_COL] == 0, C.TASK_NAMES].mean()
        diffs.extend((treated - control).tolist())
    mean_diff = sum(diffs) / len(diffs)
    assert abs(mean_diff) < 0.2


def test_within_participant_correlation_matches_icc():
    frame = generate_frame(1009)
    corr = frame[C.TASK_NAMES].corr().to_numpy()
    off_diagonal = corr[~(corr == 1.0)].mean()
    assert abs(off_diagonal - C.ICC) < 0.1
