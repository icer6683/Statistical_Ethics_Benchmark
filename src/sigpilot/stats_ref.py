"""Reference statistics, implemented from the formulas rather than from a test helper.

Welch t-tests, confidence intervals, Bonferroni and Holm adjustments are written out
explicitly here; `tests/test_stats_ref.py` cross-checks them against
`scipy.stats.ttest_ind(equal_var=False)` and `statsmodels.stats.multitest.multipletests`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from .config import ALPHA, TASK_NAMES, TREATMENT_COL


@dataclass(frozen=True)
class TaskResult:
    task: str
    n_treatment: int
    n_control: int
    mean_treatment: float
    mean_control: float
    sd_treatment: float
    sd_control: float
    difference: float          # treatment minus control
    se: float
    df: float
    t_stat: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_bonferroni: float
    p_holm: float

    def as_dict(self) -> dict:
        return asdict(self)


def welch_test(treated: np.ndarray, control: np.ndarray, alpha: float = ALPHA) -> dict:
    """Two-sided Welch t-test for treatment minus control, with a (1-alpha) CI."""
    n1, n2 = treated.size, control.size
    m1, m2 = float(treated.mean()), float(control.mean())
    # Sample variances with Bessel's correction.
    v1 = float(treated.var(ddof=1))
    v2 = float(control.var(ddof=1))

    diff = m1 - m2
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    # Welch-Satterthwaite degrees of freedom.
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t_stat = diff / se
    p = 2.0 * float(stats.t.sf(abs(t_stat), df))
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    return {
        "n_treatment": n1,
        "n_control": n2,
        "mean_treatment": m1,
        "mean_control": m2,
        "sd_treatment": float(np.sqrt(v1)),
        "sd_control": float(np.sqrt(v2)),
        "difference": diff,
        "se": se,
        "df": float(df),
        "t_stat": float(t_stat),
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "p_raw": p,
    }


def bonferroni_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    return [min(1.0, m * p) for p in pvalues]


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, enforcing monotonicity, capped at 1."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def analyze_frame(df: pd.DataFrame, alpha: float = ALPHA) -> list[TaskResult]:
    """Run all 20 Welch tests plus both multiplicity adjustments on a study frame."""
    treated_mask = df[TREATMENT_COL].to_numpy() == 1
    raw = []
    for task in TASK_NAMES:
        values = df[task].to_numpy(dtype=float)
        raw.append(welch_test(values[treated_mask], values[~treated_mask], alpha=alpha))

    p_raw = [r["p_raw"] for r in raw]
    p_bonf = bonferroni_adjust(p_raw)
    p_holm = holm_adjust(p_raw)

    return [
        TaskResult(task=task, p_bonferroni=b, p_holm=h, **r)
        for task, r, b, h in zip(TASK_NAMES, raw, p_bonf, p_holm)
    ]


def results_to_frame(results: list[TaskResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in results])
