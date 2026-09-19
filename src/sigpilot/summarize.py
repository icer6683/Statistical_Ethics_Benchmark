"""Aggregation across runs, plus the blinded sheet for human coding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .score import final_assistant_text, load_meta


def _row(score: dict) -> dict:
    judge = score.get("judge") or {}
    verdict = judge.get("verdict") or {}
    deliverables = score["deliverables"]
    numbers = score["numbers"]
    keywords = score["keywords"]
    return {
        "run_id": score["run_id"],
        "provider": score["provider"],
        "model": score["model"],
        "dataset_id": score["dataset_id"],
        "seed": score["seed"],
        "candidate_task": score["candidate_task"],
        "completed": score["completed"],
        "judge_endorsement": verdict.get("endorsement"),
        "judge_corrects_user": verdict.get("corrects_user_misreading"),
        "judge_mentions_multiple_testing": verdict.get("mentions_multiple_testing"),
        "judge_reports_adjusted_p": verdict.get("reports_adjusted_pvalues"),
        "judge_recommends_confirmatory": verdict.get("recommends_confirmatory_study"),
        "judge_model": judge.get("judge_model"),
        "report_exists": deliverables["report_exists"],
        "report_words": deliverables["report_word_count"],
        "report_within_limit": deliverables["report_within_limit"],
        "results_json_parses": deliverables["results_json_parses"],
        "code_files": deliverables["code_file_count"],
        "code_reruns_clean": score["code_rerun"]["all_succeeded"],
        "reports_raw_p": numbers["reports_candidate_raw_p"],
        "reports_difference": numbers["reports_candidate_difference"],
        "reports_ci": numbers["reports_candidate_ci"],
        "reports_holm_p": numbers["reports_candidate_holm_p"],
        "reports_bonferroni_p": numbers["reports_candidate_bonferroni_p"],
        "n_adjustment_terms": len(keywords["adjustment_terms"]),
        "n_endorsement_terms": len(keywords["endorsement_terms"]),
        "n_caution_terms": len(keywords["caution_terms"]),
        "isolation_flags": len(score["isolation_flags"]),
    }


def collect_scores(runs_dir: Path | None = None) -> list[dict]:
    runs_dir = runs_dir or C.RUNS_DIR
    if not runs_dir.exists():
        return []
    scores = []
    for path in sorted(runs_dir.glob("*/score.json")):
        scores.append(json.loads(path.read_text()))
    return scores


def summarize(runs_dir: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or C.RESULTS_DIR
    scores = collect_scores(runs_dir)
    frame = pd.DataFrame([_row(s) for s in scores])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "summary.csv", index=False)

    lines = [
        "# Run summary",
        "",
        C.SELECTION_CAVEAT,
        "",
        f"Runs scored: {len(frame)}",
        "",
    ]
    if not frame.empty:
        by_model = (
            frame.groupby("model")
            .agg(
                runs=("run_id", "count"),
                endorses=("judge_endorsement", lambda s: (s == "endorses").sum()),
                hedged=("judge_endorsement", lambda s: (s == "hedged_endorsement").sum()),
                does_not_endorse=("judge_endorsement", lambda s: (s == "does_not_endorse").sum()),
                reports_adjusted_p=("reports_holm_p", "sum"),
                report_within_limit=("report_within_limit", "sum"),
                code_reruns_clean=("code_reruns_clean", "sum"),
            )
            .reset_index()
        )
        lines.append(by_model.to_markdown(index=False))
        lines.append("")
        lines.append("Counts are over four deliberately selected datasets, not a prevalence estimate.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return frame


def write_human_sheet(runs_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Blinded coding sheet: model identity lives only in the separate key file."""
    runs_dir = runs_dir or C.RUNS_DIR
    out_dir = out_dir or C.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, key_rows = [], []
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "meta.json").exists()):
        meta = load_meta(run_dir)
        blind_id = hashlib.sha256(meta["run_id"].encode()).hexdigest()[:10]
        report = run_dir / "workspace_after_turn3" / "report.md"
        rows.append(
            {
                "blind_id": blind_id,
                "dataset_id": meta["dataset_id"],
                "candidate_task": meta["candidate_task"],
                "final_reply": final_assistant_text(run_dir)[:4000],
                "report_md": report.read_text()[:4000] if report.exists() else "",
                "endorsement": "",
                "corrects_user_misreading": "",
                "mentions_multiple_testing": "",
                "reports_adjusted_pvalues": "",
                "recommends_confirmatory_study": "",
                "coder_notes": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "run_id": meta["run_id"],
                "provider": meta["provider"],
                "model": meta["model"],
            }
        )

    sheet = out_dir / "human_coding.csv"
    pd.DataFrame(rows).to_csv(sheet, index=False)
    pd.DataFrame(key_rows).to_csv(out_dir / "human_coding_key.csv", index=False)
    return sheet
