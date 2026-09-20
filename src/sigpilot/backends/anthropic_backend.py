"""Anthropic backend (default tested model: claude-haiku-4-5).

A manual tool-use loop rather than the SDK tool runner: the pilot needs the raw message
history in provider-native form, one assistant step at a time, and the same control flow
for both providers. Server-side refusal fallbacks are deliberately not enabled — silently
swapping in another model would change the model under test.
"""

from __future__ import annotations

import json

from ..tools import TOOL_SPECS
from .base import AssistantStep, Backend, ToolCall, with_retries

DEFAULT_MODEL = "claude-haiku-4-5"

# A hung request must not stall a whole batch: cap each call and let `with_retries`
# handle the retry, rather than the SDK's 10-minute default.
REQUEST_TIMEOUT_SECONDS = 300.0
SDK_MAX_RETRIES = 1


class AnthropicBackend(Backend):
    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, system: str = "", tools=None, max_tokens: int = 8000):
        import anthropic  # imported lazily so offline runs need no SDK

        super().__init__(model=model, system=system, tools=tools or TOOL_SPECS, max_tokens=max_tokens)
        self.client = anthropic.Anthropic(
            timeout=REQUEST_TIMEOUT_SECONDS, max_retries=SDK_MAX_RETRIES
        )
        self.messages: list[dict] = []

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> AssistantStep:
        # Top-level auto-caching: each request caches the tail of the history, so the
        # next one reads the whole prefix at ~0.1x instead of resending it at full
        # price. An agentic loop resends the entire conversation on every tool call,
        # so without this the cost of a run grows with the square of its tool calls.
        # The cached prefix must stay byte-stable: the system prompt, the tool list
        # and the message history are all fixed once written, and nothing volatile
        # (timestamps, ids) is inserted into them.
        response = with_retries(
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                cache_control={"type": "ephemeral"},
                system=self.system,
                tools=self.tools,
                messages=self.messages,
            )
        )
        content = [block.model_dump() for block in response.content]
        self.messages.append({"role": "assistant", "content": content})

        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        tool_calls = [
            ToolCall(call_id=b["id"], name=b["name"], arguments=b.get("input") or {})
            for b in content
            if b.get("type") == "tool_use"
        ]
        usage = response.usage.model_dump() if response.usage else {}
        return AssistantStep(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage=usage,
            raw={"content": content, "id": response.id},
        )

    def append_tool_results(self, results) -> None:
        # All results for one assistant turn go back in a single user message.
        blocks = []
        for call, result in results:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.call_id,
                    "content": result.output or "(no output)",
                    "is_error": result.is_error,
                }
            )
        self.messages.append({"role": "user", "content": blocks})

    def history(self) -> list:
        return json.loads(json.dumps(self.messages, default=str))
