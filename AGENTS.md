# Repository conventions

This repository runs a pilot experiment on LLMs. Two things break the experiment silently,
so check them before any change:

## 1. Never leak the design into model-facing text

Model-facing text is: `prompts.SYSTEM_PROMPT`, the three prompts, and every tool name,
description and parameter description in `tools.py`. It must not mention multiplicity
corrections, Holm or Bonferroni, statistical ethics, evaluation, scoring, or the existence
of a benchmark. `config.BANNED_MODEL_FACING_TERMS` lists the forbidden substrings;
`tests/test_prompts_and_isolation.py` and `sigpilot verify` enforce it. Add a term to that
list rather than relying on review.

The three prompts are fixed study material. Do not reword them, reorder them, add a fourth,
or paraphrase them in code comments that end up in the prompt. Their wording is quoted in
`docs/design.md`.

## 2. Never let reference answers reach a workspace

`references/` holds the correct answers. A run workspace is a temp directory outside the
repository containing only `data.csv`. Do not copy, symlink or path-reference anything from
`references/`, `data/manifest.json` or the repository into a workspace, and do not relax the
path guards in `workspace.py` or the environment stripping in `tools.py`.

## Other conventions

- **Data is generated, not edited.** Never hand-edit files under `data/`. Change the DGP or
  the selection rule in `generate.py` / `config.py`, then re-run `sigpilot generate` and
  `sigpilot reference`; the manifest hashes must be regenerated with them.
- **Preserve existing runs.** `results/runs/` is a record of real API calls that cost money.
  Do not delete or rewrite run directories; new runs get new directories.
- **Reference statistics stay independent.** `stats_ref.py` implements Welch, Bonferroni and
  Holm from the formulas on purpose. Do not replace it with library calls — the library
  results are the cross-check in `tests/test_stats_ref.py`, and they cannot be both.
- **A model never judges itself.** Keep the `select_judge` guard; a judge defaults to the
  other provider.
- **Report counts, not prevalence.** These four datasets are selected challenge cases. Any
  summary that quotes a rate must carry the caveat in `config.SELECTION_CAVEAT`.

## Working setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q      # 59 tests, no network, no API keys needed
.venv/bin/sigpilot verify          # hashes, criteria, model-facing text, isolation
```

Offline end-to-end check (no API calls):

```bash
.venv/bin/sigpilot run --backend scripted --model scripted-correcting --datasets ds1
.venv/bin/sigpilot score --no-judge && .venv/bin/sigpilot summarize
```
