"""Model-facing text, prompt filling, and the workspace/tool isolation guards."""

from __future__ import annotations

import os
import re

import pytest

from sigpilot import config as C
from sigpilot.generate import load_manifest
from sigpilot.prompts import PROMPT_1, PROMPT_2, build_prompts, model_facing_strings
from sigpilot.reference import load_reference
from sigpilot.tools import execute_tool
from sigpilot.workspace import PathEscape, create_workspace, list_workspace, resolve_in_workspace

DATASET_IDS = [e["dataset_id"] for e in load_manifest()["datasets"]]


@pytest.fixture(params=DATASET_IDS)
def reference(request):
    return load_reference(request.param)


@pytest.fixture
def workspace(tmp_path_factory):
    entry = load_manifest()["datasets"][0]
    return create_workspace(C.REPO_ROOT / entry["path"])


# --- prompts ---------------------------------------------------------------


def test_exactly_three_prompts_two_setup_then_critical(reference):
    prompts = build_prompts(reference)
    assert len(prompts) == 3
    assert prompts[0] == PROMPT_1 and prompts[1] == PROMPT_2
    assert "I'm not experienced with statistics" in prompts[2]


def test_placeholders_filled_from_reference(reference):
    prompt = build_prompts(reference)[2]
    candidate = reference["candidate"]
    assert "{" not in prompt and "}" not in prompt
    assert prompt.startswith(reference["candidate_task"])
    assert f"{candidate['difference']:.2f} points" in prompt
    assert f"p = {candidate['p_raw']:.3f}" in prompt

    # The quoted numbers must be the real ones, not placeholders left in place.
    quoted_p = float(re.search(r"p = (\d+\.\d+)", prompt).group(1))
    assert quoted_p == pytest.approx(candidate["p_raw"], abs=5e-4)
    assert 0.005 <= quoted_p <= 0.04


def test_model_facing_text_has_no_banned_terms():
    for text in model_facing_strings():
        lowered = text.lower()
        for term in C.BANNED_MODEL_FACING_TERMS:
            assert term not in lowered, f"{term!r} leaks in: {text[:80]}"


def test_prompt_three_does_not_hint_at_the_answer(reference):
    prompt = build_prompts(reference)[2].lower()
    for hint in ["adjust", "correct for", "20 tests", "false positive", "chance"]:
        assert hint not in prompt


# --- workspace isolation ---------------------------------------------------


def test_workspace_contains_only_data_csv(workspace):
    assert list_workspace(workspace) == ["data.csv"]
    assert C.REPO_ROOT not in workspace.parents


def test_path_escapes_are_rejected(workspace):
    for bad in ["../secret.txt", "../../references/ds1/reference.json", "/etc/passwd"]:
        with pytest.raises(PathEscape):
            resolve_in_workspace(workspace, bad)
    result = execute_tool(workspace, "read_file", {"path": "../../references/ds1/reference.json"})
    assert result.is_error and result.isolation_flag


def test_reference_access_attempt_is_flagged(workspace):
    code = f"print(open('{C.REFERENCES_DIR}/ds1/reference.json').read())"
    result = execute_tool(workspace, "run_python", {"code": code})
    assert result.isolation_flag


def test_run_python_has_no_api_keys_and_can_use_pandas(workspace, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    code = (
        "import os, pandas as pd\n"
        "print('keys:', [k for k in os.environ if 'API_KEY' in k])\n"
        "print('rows:', len(pd.read_csv('data.csv')))\n"
    )
    result = execute_tool(workspace, "run_python", {"code": code})
    assert not result.is_error
    assert "keys: []" in result.output
    assert "rows: 200" in result.output
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-should-not-leak"  # parent env untouched


def test_write_and_read_roundtrip(workspace):
    write = execute_tool(workspace, "write_file", {"path": "notes/report.md", "content": "hello"})
    assert not write.is_error
    read = execute_tool(workspace, "read_file", {"path": "notes/report.md"})
    assert read.output == "hello"
    assert "notes/report.md" in list_workspace(workspace)
