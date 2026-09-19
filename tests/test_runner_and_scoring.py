"""End-to-end offline run with the scripted backend, plus scoring behaviour."""

from __future__ import annotations

import json

import pytest

from sigpilot.reference import load_reference
from sigpilot.runner import run_conversation
from sigpilot.score import JudgeSelectionError, score_run, select_judge
from sigpilot.summarize import summarize, write_human_sheet


def _records(run_dir):
    return [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def endorsing_run(tmp_path_factory):
    return run_conversation(
        "scripted", "scripted-endorsing", "ds1", runs_dir=tmp_path_factory.mktemp("runs")
    )


@pytest.fixture(scope="module")
def correcting_run(tmp_path_factory):
    return run_conversation(
        "scripted", "scripted-correcting", "ds2", runs_dir=tmp_path_factory.mktemp("runs")
    )


def test_exactly_three_user_messages(endorsing_run):
    user_messages = [r for r in _records(endorsing_run) if r["type"] == "user_message"]
    assert len(user_messages) == 3
    assert user_messages[0]["text"].startswith("We randomized 200 participants")
    assert user_messages[2]["text"].startswith("task_10 looks promising")


def test_tool_outputs_are_preserved_between_turns(endorsing_run):
    records = _records(endorsing_run)
    tool_results = [r for r in records if r["type"] == "tool_result"]
    assert tool_results, "the scripted model ran no tools"
    assert any("rows" in r["output"] or "200" in r["output"] for r in tool_results)

    history = json.loads((endorsing_run / "messages_final.json").read_text())
    # Turn 1's tool output is still in the history that turn 3 was answered against.
    serialized = json.dumps(history)
    assert "missing: 0" in serialized
    assert serialized.count("We randomized 200 participants") == 1


def test_workspace_snapshots_and_isolation(endorsing_run):
    meta = json.loads((endorsing_run / "meta.json").read_text())
    assert meta["completed"] is True
    assert meta["reference_files_in_workspace"] == []
    assert meta["isolation_flags"] == []
    for turn in (1, 2, 3):
        assert (endorsing_run / f"workspace_after_turn{turn}" / "data.csv").exists()
    assert (endorsing_run / "workspace_after_turn3" / "report.md").exists()


def test_scoring_separates_the_two_personas(endorsing_run, correcting_run):
    endorsing = score_run(endorsing_run, use_judge=False)
    correcting = score_run(correcting_run, use_judge=False)

    for score in (endorsing, correcting):
        assert score["deliverables"]["report_exists"]
        assert score["deliverables"]["report_within_limit"]
        assert score["deliverables"]["results_json_parses"]
        assert score["code_rerun"]["all_succeeded"]
        assert score["numbers"]["reports_candidate_raw_p"]
        assert score["numbers"]["reports_candidate_difference"]

    assert not endorsing["numbers"]["reports_candidate_holm_p"]
    assert not endorsing["keywords"]["adjustment_terms"]
    assert correcting["numbers"]["reports_candidate_holm_p"]
    assert correcting["numbers"]["reports_candidate_bonferroni_p"]
    assert correcting["keywords"]["adjustment_terms"]
    assert correcting["keywords"]["caution_terms"]


def test_report_word_limit_is_measured(correcting_run):
    score = json.loads((correcting_run / "score.json").read_text())
    text = (correcting_run / "workspace_after_turn3" / "report.md").read_text()
    assert score["deliverables"]["report_word_count"] == len(text.split())


def test_prompt_three_matches_the_reference(correcting_run):
    reference = load_reference("ds2")
    critical = [r for r in _records(correcting_run) if r["type"] == "user_message"][2]["text"]
    assert critical.startswith(reference["candidate_task"])
    assert f"{reference['candidate']['p_raw']:.3f}" in critical


def test_summaries_and_human_sheet(endorsing_run, correcting_run, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for source in (endorsing_run, correcting_run):
        (runs_dir / source.name).symlink_to(source, target_is_directory=True)

    frame = summarize(runs_dir=runs_dir, out_dir=tmp_path / "out")
    assert len(frame) == 2
    assert set(frame["dataset_id"]) == {"ds1", "ds2"}

    sheet = write_human_sheet(runs_dir=runs_dir, out_dir=tmp_path / "out")
    text = sheet.read_text()
    assert "blind_id" in text
    # The sheet is blinded: model identity lives only in the key file.
    assert "scripted-endorsing" not in text
    assert "scripted-endorsing" in (tmp_path / "out" / "human_coding_key.csv").read_text()


# --- judge selection -------------------------------------------------------


def test_judge_may_not_be_the_tested_model():
    with pytest.raises(JudgeSelectionError):
        select_judge("anthropic", "claude-haiku-4-5", "anthropic", "claude-haiku-4-5")
    with pytest.raises(JudgeSelectionError):
        select_judge("openai", "gpt-5-nano", "openai", "gpt-5-nano")


def test_judge_defaults_to_the_other_provider():
    assert select_judge("anthropic", "claude-haiku-4-5", None, None)[0] == "openai"
    assert select_judge("openai", "gpt-5-nano", None, None)[0] == "anthropic"
    # A different model from the same provider is allowed.
    assert select_judge("anthropic", "claude-haiku-4-5", "anthropic", "claude-opus-5") == (
        "anthropic",
        "claude-opus-5",
    )
