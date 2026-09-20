"""Token accounting for recorded runs, so spend stays visible.

Every assistant step stores the provider's own `usage` block in the transcript, so cost
is read back from the runs themselves rather than estimated.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config as C

# US dollars per million tokens. Anthropic list prices; cache writes bill at 1.25x the
# input rate and cache reads at 0.1x. Update here if prices change.
ANTHROPIC_PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def _normalize_usage(usage: dict) -> dict:
    """Map either provider's usage block onto one set of names.

    Anthropic reports `input_tokens` / `output_tokens` plus explicit cache counters;
    OpenAI reports `prompt_tokens` / `completion_tokens` with cached tokens nested in
    `prompt_tokens_details` (its caching is automatic, so there is no write counter).
    """
    if "prompt_tokens" in usage:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return {
            "input_tokens": (usage.get("prompt_tokens") or 0) - cached,
            "output_tokens": usage.get("completion_tokens") or 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached,
        }
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens") or 0,
    }


def usage_for_run(run_dir: Path) -> dict:
    """Total token usage for one run, from the transcript's recorded usage blocks."""
    totals = {
        "steps": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    provider = model = None
    transcript = run_dir / "transcript.jsonl"
    if not transcript.exists():
        return {**totals, "provider": None, "model": None}

    for line in transcript.read_text().splitlines():
        record = json.loads(line)
        if record["type"] == "meta":
            provider, model = record.get("provider"), record.get("model")
        elif record["type"] == "assistant_step" and record.get("usage"):
            totals["steps"] += 1
            for key, value in _normalize_usage(record["usage"]).items():
                totals[key] += value
    return {**totals, "provider": provider, "model": model}


def cost_for_usage(usage: dict) -> float | None:
    """Dollar cost, or None when no price list covers the model."""
    price = ANTHROPIC_PRICING.get(usage.get("model") or "")
    if not price:
        return None
    return (
        usage["input_tokens"] / 1e6 * price["input"]
        + usage["output_tokens"] / 1e6 * price["output"]
        + usage["cache_creation_input_tokens"] / 1e6 * price["input"] * CACHE_WRITE_MULTIPLIER
        + usage["cache_read_input_tokens"] / 1e6 * price["input"] * CACHE_READ_MULTIPLIER
    )


def uncached_cost_for_usage(usage: dict) -> float | None:
    """What the same run would have cost with no caching, for comparison."""
    price = ANTHROPIC_PRICING.get(usage.get("model") or "")
    if not price:
        return None
    all_input = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    return all_input / 1e6 * price["input"] + usage["output_tokens"] / 1e6 * price["output"]


def report(runs_dir: Path | None = None) -> list[dict]:
    """Per-model token and cost totals across all recorded runs."""
    runs_dir = runs_dir or C.RUNS_DIR
    if not runs_dir.exists():
        return []

    by_model: dict[str, dict] = {}
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        usage = usage_for_run(run_dir)
        if not usage["model"] or usage["provider"] == "scripted":
            continue
        entry = by_model.setdefault(
            usage["model"],
            {
                "model": usage["model"],
                "provider": usage["provider"],
                "runs": 0,
                "steps": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )
        entry["runs"] += 1
        for key in (
            "steps",
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            entry[key] += usage[key]

    rows = []
    for entry in by_model.values():
        cost = cost_for_usage(entry)
        uncached = uncached_cost_for_usage(entry)
        billed_input = (
            entry["input_tokens"]
            + entry["cache_creation_input_tokens"]
            + entry["cache_read_input_tokens"]
        )
        rows.append(
            {
                **entry,
                "cost_usd": cost,
                "cost_without_cache_usd": uncached,
                "cache_hit_rate": (
                    entry["cache_read_input_tokens"] / billed_input if billed_input else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda r: -(r["cost_usd"] or 0))
