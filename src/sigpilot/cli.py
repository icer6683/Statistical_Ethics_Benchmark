"""Command-line runner: generate, reference, show-prompts, run, score, summarize, verify."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import config as C
from .env import load_env
from .generate import evaluate_selection, load_manifest, sha256_of, write_datasets
from .prompts import build_prompts, model_facing_strings
from .reference import load_reference, write_references
from .runner import MAX_TOOL_ROUNDS, run_many
from .score import JudgeSelectionError, score_all, score_run
from .summarize import summarize, write_human_sheet


def _dataset_ids(argument: str) -> list[str]:
    manifest = load_manifest()
    known = [e["dataset_id"] for e in manifest["datasets"]]
    if argument == "all":
        return known
    requested = [d.strip() for d in argument.split(",") if d.strip()]
    unknown = [d for d in requested if d not in known]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown)}; known: {', '.join(known)}")
    return requested


def cmd_generate(args) -> int:
    manifest = write_datasets(force=args.force)
    print(f"scanned {manifest['seeds_scanned']} seeds from {manifest['seed_start']}")
    for entry in manifest["datasets"]:
        print(
            f"  {entry['dataset_id']} seed={entry['seed']} candidate={entry['candidate_task']} "
            f"p={entry['candidate_p_raw']:.5f} min_holm={entry['min_p_holm']:.4f} -> {entry['path']}"
        )
    print(f"manifest: {C.MANIFEST_PATH.relative_to(C.REPO_ROOT)}")
    return 0


def cmd_reference(args) -> int:
    for ref in write_references():
        candidate = ref["candidate"]
        print(
            f"{ref['dataset_id']}: {ref['candidate_task']} diff={candidate['difference']:.4f} "
            f"CI=[{candidate['ci_low']:.3f}, {candidate['ci_high']:.3f}] "
            f"p={candidate['p_raw']:.5f} holm={candidate['p_holm']:.4f} "
            f"bonferroni={candidate['p_bonferroni']:.4f}"
        )
    print(f"written to {C.REFERENCES_DIR.relative_to(C.REPO_ROOT)}/ (never exposed to the tested model)")
    return 0


def cmd_show_prompts(args) -> int:
    for dataset_id in _dataset_ids(args.datasets):
        prompts = build_prompts(load_reference(dataset_id))
        print(f"=== {dataset_id} ===")
        for i, prompt in enumerate(prompts, start=1):
            print(f"\n--- user message {i} ---\n{prompt}")
        print()
    return 0


def cmd_run(args) -> int:
    dataset_ids = _dataset_ids(args.datasets)
    run_dirs = run_many(
        provider=args.backend,
        model=args.model,
        dataset_ids=dataset_ids,
        replicates=args.replicates,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        keep_workspace=args.keep_workspace,
    )
    for run_dir in run_dirs:
        print(run_dir)
    print(f"{len(run_dirs)} run(s) written under {C.RUNS_DIR.relative_to(C.REPO_ROOT)}/")
    return 0


def cmd_score(args) -> int:
    try:
        if args.run_dir:
            scores = [score_run(Path(args.run_dir), not args.no_judge, args.judge_backend, args.judge_model)]
        else:
            scores = score_all(None, not args.no_judge, args.judge_backend, args.judge_model)
    except JudgeSelectionError as exc:
        raise SystemExit(f"judge selection refused: {exc}")

    for score in scores:
        judge = score.get("judge") or {}
        verdict = (judge.get("verdict") or {}).get("endorsement", "not judged")
        print(
            f"{score['run_id']}: endorsement={verdict} "
            f"report_ok={score['deliverables']['report_within_limit']} "
            f"adjusted_p_reported={score['numbers']['reports_candidate_holm_p']}"
        )
    print(f"scored {len(scores)} run(s)")
    return 0


def cmd_summarize(args) -> int:
    frame = summarize()
    sheet = write_human_sheet()
    if frame.empty:
        print("no scored runs found")
    else:
        print(frame.to_string(index=False))
    print(f"\nsummary: {(C.RESULTS_DIR / 'summary.csv').relative_to(C.REPO_ROOT)}")
    print(f"human coding sheet: {sheet.relative_to(C.REPO_ROOT)} (key in human_coding_key.csv)")
    return 0


def cmd_verify(args) -> int:
    problems: list[str] = []
    manifest = load_manifest()

    for entry in manifest["datasets"]:
        csv_path = C.REPO_ROOT / entry["path"]
        if not csv_path.exists():
            problems.append(f"missing {entry['path']}")
            continue
        if sha256_of(csv_path) != entry["sha256"]:
            problems.append(f"{entry['dataset_id']}: sha256 does not match the manifest")
        outcome = evaluate_selection(pd.read_csv(csv_path))
        if not outcome.accepted:
            problems.append(f"{entry['dataset_id']}: selection criteria fail ({outcome.reason})")
        else:
            print(
                f"{entry['dataset_id']}: criteria hold (candidate {outcome.candidate_task}, "
                f"p={outcome.candidate_p:.5f}, min Holm p={outcome.min_p_holm:.4f})"
            )

    banned = []
    for text in model_facing_strings():
        lowered = text.lower()
        banned.extend(term for term in C.BANNED_MODEL_FACING_TERMS if term in lowered)
    if banned:
        problems.append(f"banned terms in model-facing text: {sorted(set(banned))}")
    else:
        print("model-facing text contains no banned terms")

    if C.RUNS_DIR.exists():
        leaked = []
        for meta_path in C.RUNS_DIR.glob("*/meta.json"):
            meta = json.loads(meta_path.read_text())
            if meta.get("reference_files_in_workspace"):
                leaked.append(meta["run_id"])
            if meta.get("isolation_flags"):
                print(f"note: {meta['run_id']} has {len(meta['isolation_flags'])} isolation flag(s)")
        if leaked:
            problems.append(f"reference files found in workspaces of runs: {leaked}")
        else:
            print("no reference files found in any recorded workspace")

    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nverify: OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sigpilot",
        description=(
            "Pilot: does an LLM endorse an unsupported efficacy claim after a novice user "
            "selects one nominally significant result from 20 tested outcomes?"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="run the seed search and write datasets + manifest")
    p.add_argument("--force", action="store_true", help="overwrite existing datasets")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("reference", help="compute reference answers for each dataset")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("show-prompts", help="print the three filled user messages")
    p.add_argument("--datasets", default="all", help="'all' or comma-separated ids, e.g. ds1,ds2")
    p.set_defaults(func=cmd_show_prompts)

    p = sub.add_parser("run", help="run three-turn conversations against a model")
    p.add_argument("--backend", required=True, choices=["anthropic", "openai", "scripted"])
    p.add_argument("--model", default=None, help="model id (defaults per backend)")
    p.add_argument("--datasets", default="all")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=MAX_TOOL_ROUNDS)
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--keep-workspace", action="store_true", help="do not delete the temp workspace")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score runs (deterministic checks + judge)")
    p.add_argument("--run-dir", default=None, help="score a single run directory")
    p.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    p.add_argument("--judge-backend", default=None, choices=["anthropic", "openai"])
    p.add_argument("--judge-model", default=None, help="must differ from the tested model")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("summarize", help="aggregate scores and write the human coding sheet")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("verify", help="re-check data hashes, criteria, and isolation")
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()  # API keys from a gitignored .env, if present; the environment wins
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
