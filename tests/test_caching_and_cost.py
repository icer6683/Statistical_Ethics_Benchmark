"""Prompt caching is actually requested, and recorded usage is accounted correctly."""

from __future__ import annotations

import copy
import json
import sys
import types

import pytest

from sigpilot.cost import cost_for_usage, report, uncached_cost_for_usage, usage_for_run


class _FakeUsage:
    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self):
        return dict(self._data)


class _FakeBlock:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        # Snapshot: the backend passes its live message list, which the real SDK
        # serializes immediately but a reference here would keep mutating.
        self.calls.append(copy.deepcopy(kwargs))
        return types.SimpleNamespace(
            id="msg_fake",
            content=[_FakeBlock({"type": "text", "text": "hello"})],
            stop_reason="end_turn",
            usage=_FakeUsage(input_tokens=10, output_tokens=5),
        )


@pytest.fixture
def anthropic_backend(monkeypatch):
    """An AnthropicBackend whose client records requests instead of sending them."""
    fake_messages = _FakeMessages()
    fake_module = types.SimpleNamespace(
        Anthropic=lambda **kwargs: types.SimpleNamespace(messages=fake_messages)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    from sigpilot.backends.anthropic_backend import AnthropicBackend

    backend = AnthropicBackend(model="claude-haiku-4-5", system="sys")
    return backend, fake_messages


def test_requests_enable_prompt_caching(anthropic_backend):
    backend, fake = anthropic_backend
    backend.append_user("first question")
    backend.step()
    backend.append_user("second question")
    backend.step()

    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["cache_control"] == {"type": "ephemeral"}


def test_cached_prefix_stays_byte_stable(anthropic_backend):
    """History is append-only: a cached prefix is never rewritten by a later turn."""
    backend, fake = anthropic_backend
    backend.append_user("first question")
    backend.step()
    first_prefix = json.dumps(fake.calls[0]["messages"])

    backend.append_user("second question")
    backend.step()
    second_prefix = json.dumps(fake.calls[1]["messages"][: len(fake.calls[0]["messages"])])

    assert second_prefix == first_prefix
    assert fake.calls[0]["system"] == fake.calls[1]["system"]
    assert fake.calls[0]["tools"] == fake.calls[1]["tools"]


# --- cost accounting -------------------------------------------------------


def test_cost_uses_cache_multipliers():
    usage = {
        "model": "claude-haiku-4-5",
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
    }
    # 1.00 input + 5.00 output + 1.25 cache write + 0.10 cache read
    assert cost_for_usage(usage) == pytest.approx(7.35)
    # Without caching all four million input-side tokens bill at the input rate.
    assert uncached_cost_for_usage(usage) == pytest.approx(3 * 1.00 + 5.00)


def test_unknown_model_has_no_price():
    assert cost_for_usage({"model": "gpt-5-nano", "input_tokens": 1}) is None


def _write_run(tmp_path, name, provider, model, usages):
    run_dir = tmp_path / name
    run_dir.mkdir()
    lines = [json.dumps({"type": "meta", "provider": provider, "model": model})]
    for usage in usages:
        lines.append(json.dumps({"type": "assistant_step", "turn": 1, "usage": usage}))
    (run_dir / "transcript.jsonl").write_text("\n".join(lines) + "\n")
    return run_dir


def test_usage_for_run_reads_anthropic_and_openai_shapes(tmp_path):
    anthropic_run = _write_run(
        tmp_path,
        "a",
        "anthropic",
        "claude-haiku-4-5",
        [{"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 900}],
    )
    openai_run = _write_run(
        tmp_path,
        "o",
        "openai",
        "gpt-5-nano",
        [
            {
                "prompt_tokens": 1000,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 400},
            }
        ],
    )

    a = usage_for_run(anthropic_run)
    assert (a["input_tokens"], a["output_tokens"], a["cache_read_input_tokens"]) == (100, 20, 900)

    o = usage_for_run(openai_run)
    assert o["input_tokens"] == 600  # prompt tokens minus the cached portion
    assert o["output_tokens"] == 30
    assert o["cache_read_input_tokens"] == 400

    rows = {r["model"]: r for r in report(runs_dir=tmp_path)}
    assert rows["claude-haiku-4-5"]["cache_hit_rate"] == pytest.approx(0.9)
    assert rows["gpt-5-nano"]["cost_usd"] is None
