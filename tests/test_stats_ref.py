"""The hand-written reference statistics agree with scipy and statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sigpilot import config as C
from sigpilot.generate import load_manifest
from sigpilot.reference import load_reference
from sigpilot.stats_ref import analyze_frame, bonferroni_adjust, holm_adjust, welch_test

MANIFEST = load_manifest()


@pytest.fixture(params=MANIFEST["datasets"], ids=lambda e: e["dataset_id"])
def entry(request):
    return request.param


def test_welch_matches_scipy(entry):
    frame = pd.read_csv(C.REPO_ROOT / entry["path"])
    treated_mask = frame[C.TREATMENT_COL].to_numpy() == 1
    for task in C.TASK_NAMES:
        values = frame[task].to_numpy(float)
        mine = welch_test(values[treated_mask], values[~treated_mask])
        theirs = stats.ttest_ind(values[treated_mask], values[~treated_mask], equal_var=False)
        ci = theirs.confidence_interval(0.95)
        assert mine["p_raw"] == pytest.approx(theirs.pvalue, abs=1e-12)
        assert mine["t_stat"] == pytest.approx(theirs.statistic, abs=1e-10)
        assert mine["df"] == pytest.approx(theirs.df, abs=1e-9)
        assert mine["ci_low"] == pytest.approx(ci.low, abs=1e-9)
        assert mine["ci_high"] == pytest.approx(ci.high, abs=1e-9)


def test_adjustments_match_statsmodels(entry):
    frame = pd.read_csv(C.REPO_ROOT / entry["path"])
    results = analyze_frame(frame)
    p_raw = [r.p_raw for r in results]
    assert np.allclose([r.p_holm for r in results], multipletests(p_raw, method="holm")[1], atol=1e-12)
    assert np.allclose(
        [r.p_bonferroni for r in results], multipletests(p_raw, method="bonferroni")[1], atol=1e-12
    )


def test_holm_properties():
    p = [0.001, 0.02, 0.04, 0.5, 0.9]
    adjusted = holm_adjust(p)
    assert all(a >= b for a, b in zip(adjusted, p))  # never smaller than raw
    ordered = [adjusted[i] for i in sorted(range(len(p)), key=lambda i: p[i])]
    assert ordered == sorted(ordered)  # monotone in rank
    assert max(adjusted) <= 1.0
    assert holm_adjust([0.01]) == [0.01]
    assert bonferroni_adjust([0.3, 0.9]) == [0.6, 1.0]


def test_reference_matches_recomputation(entry):
    reference = load_reference(entry["dataset_id"])
    results = {r.task: r for r in analyze_frame(pd.read_csv(C.REPO_ROOT / entry["path"]))}
    for row in reference["tasks"]:
        recomputed = results[row["task"]]
        assert row["p_raw"] == pytest.approx(recomputed.p_raw, abs=1e-12)
        assert row["difference"] == pytest.approx(recomputed.difference, abs=1e-12)
    assert reference["n_significant_holm"] == 0
    assert reference["n_significant_raw"] == 1
    assert reference["data_sha256"] == entry["sha256"]
