"""The three-turn conversation driver.

Exactly three user messages are sent per conversation: two setup messages and one critical
message. After each, the model runs a tool loop until it stops calling tools. The model's
actual replies and the actual tool outputs stay in the history unchanged — nothing is
summarized, rewritten or trimmed between turns, and context length is not manipulated.
"""

from __future__ import annotations

import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .backends import make_backend
from .generate import load_manifest
from .prompts import SYSTEM_PROMPT, build_prompts
from .reference import load_reference
from .tools import execute_tool
from .workspace import assert_no_references, create_workspace, snapshot

MAX_TOOL_ROUNDS = 40


def run_id_for(provider: str, model: str, dataset_id: str, replicate: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "-")
    return f"{stamp}_{provider}_{safe_model}_{dataset_id}_r{replicate}_{uuid.uuid4().hex[:6]}"


class TranscriptWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w")

    def write(self, record: dict) -> None:
        record["logged_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.handle.write(json.dumps(record, default=str) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_conversation(
    provider: str,
    model: str | None,
    dataset_id: str,
    replicate: int = 1,
    max_rounds: int = MAX_TOOL_ROUNDS,
    max_tokens: int = 8000,
    runs_dir: Path | None = None,
    keep_workspace: bool = False,
) -> Path:
    """Run one full three-turn conversation and return its run directory."""
    manifest = load_manifest()
    entry = next(e for e in manifest["datasets"] if e["dataset_id"] == dataset_id)
    reference = load_reference(dataset_id)
    prompts = build_prompts(reference)

    backend_kwargs: dict = {"system": SYSTEM_PROMPT, "max_tokens": max_tokens}
    if provider == "scripted":
        backend_kwargs["candidate_task"] = reference["candidate_task"]
    backend = make_backend(provider, model, **backend_kwargs)

    run_dir = (runs_dir or C.RUNS_DIR) / run_id_for(provider, backend.model, dataset_id, replicate)
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = TranscriptWriter(run_dir / "transcript.jsonl")

    workspace = create_workspace(C.REPO_ROOT / entry["path"])
    isolation_flags: list[dict] = []
    truncation_events = 0
    round_cap_hits: list[int] = []
    step_counter = 0

    meta = {
        "run_id": run_dir.name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider,
        "model": backend.model,
        "max_tokens": max_tokens,
        "max_tool_rounds": max_rounds,
        "dataset_id": dataset_id,
        "seed": entry["seed"],
        "data_sha256": entry["sha256"],
        "candidate_task": reference["candidate_task"],
        "workspace": str(workspace),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    transcript.write({"type": "meta", **meta})
    for i, prompt in enumerate(prompts, start=1):
        transcript.write({"type": "prompt_text", "turn": i, "text": prompt})

    try:
        for turn, prompt in enumerate(prompts, start=1):
            backend.append_user(prompt)
            transcript.write({"type": "user_message", "turn": turn, "text": prompt})

            for round_index in range(max_rounds):
                step = backend.step()
                step_counter += 1
                transcript.write(
                    {
                        "type": "assistant_step",
                        "turn": turn,
                        "round": round_index,
                        "text": step.text,
                        "stop_reason": step.stop_reason,
                        "usage": step.usage,
                        "tool_calls": [
                            {"id": c.call_id, "name": c.name, "arguments": c.arguments}
                            for c in step.tool_calls
                        ],
                    }
                )
                if not step.tool_calls:
                    break

                results = []
                for call in step.tool_calls:
                    result = execute_tool(workspace, call.name, call.arguments, step=step_counter)
                    results.append((call, result))
                    if result.truncated:
                        truncation_events += 1
                    if result.isolation_flag:
                        isolation_flags.append(
                            {"turn": turn, "tool": call.name, "arguments": call.arguments}
                        )
                    transcript.write(
                        {
                            "type": "tool_result",
                            "turn": turn,
                            "round": round_index,
                            "tool": call.name,
                            "call_id": call.call_id,
                            "output": result.output,
                            "is_error": result.is_error,
                            "truncated": result.truncated,
                            "isolation_flag": result.isolation_flag,
                        }
                    )
                backend.append_tool_results(results)
            else:
                round_cap_hits.append(turn)
                transcript.write({"type": "round_cap_reached", "turn": turn, "cap": max_rounds})

            snapshot(workspace, run_dir / f"workspace_after_turn{turn}")

        leaked = assert_no_references(workspace)
        meta.update(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "isolation_flags": isolation_flags,
                "reference_files_in_workspace": leaked,
                "truncation_events": truncation_events,
                "round_cap_hits": round_cap_hits,
                "completed": True,
            }
        )
    except Exception as exc:
        meta.update(
            {
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "completed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        transcript.write({"type": "error", "error": meta["error"]})
        raise
    finally:
        (run_dir / "messages_final.json").write_text(
            json.dumps(backend.history(), indent=2, default=str) + "\n"
        )
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        transcript.write({"type": "run_end", "completed": meta.get("completed", False)})
        transcript.close()
        if keep_workspace:
            print(f"workspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    return run_dir


def run_many(
    provider: str,
    model: str | None,
    dataset_ids: list[str],
    replicates: int = 1,
    **kwargs,
) -> list[Path]:
    run_dirs = []
    for dataset_id in dataset_ids:
        for replicate in range(1, replicates + 1):
            run_dirs.append(
                run_conversation(provider, model, dataset_id, replicate=replicate, **kwargs)
            )
    return run_dirs
