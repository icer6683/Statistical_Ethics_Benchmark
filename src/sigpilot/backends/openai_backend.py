"""OpenAI backend (default tested model: gpt-5-nano).

Chat Completions with function calling. Reasoning-family models reject `temperature` and
`max_tokens`, so neither is sent; `max_completion_tokens` is used instead and dropped
automatically if a given model rejects it.
"""

from __future__ import annotations

import json

from ..tools import openai_tool_specs
from .base import AssistantStep, Backend, ToolCall, with_retries

DEFAULT_MODEL = "gpt-5-nano"

# See the Anthropic backend: bound each request so a hung call cannot stall a batch.
REQUEST_TIMEOUT_SECONDS = 300.0
SDK_MAX_RETRIES = 1


class OpenAIBackend(Backend):
    provider = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, system: str = "", tools=None, max_tokens: int = 8000):
        from openai import OpenAI  # imported lazily so offline runs need no SDK

        super().__init__(model=model, system=system, tools=tools or openai_tool_specs(), max_tokens=max_tokens)
        self.client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=SDK_MAX_RETRIES)
        self.messages: list[dict] = [{"role": "system", "content": system}] if system else []
        self._send_max_completion_tokens = True

    def _create(self):
        kwargs = dict(model=self.model, messages=self.messages, tools=self.tools)
        if self._send_max_completion_tokens:
            kwargs["max_completion_tokens"] = self.max_tokens
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._send_max_completion_tokens and "max_completion_tokens" in str(exc):
                self._send_max_completion_tokens = False
                return self.client.chat.completions.create(
                    model=self.model, messages=self.messages, tools=self.tools
                )
            raise

    def append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def step(self) -> AssistantStep:
        response = with_retries(self._create)
        choice = response.choices[0]
        message = choice.message

        assistant: dict = {"role": "assistant", "content": message.content or ""}
        tool_calls = []
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"__invalid_json__": tc.function.arguments}
                tool_calls.append(ToolCall(call_id=tc.id, name=tc.function.name, arguments=arguments))
        self.messages.append(assistant)

        usage = response.usage.model_dump() if response.usage else {}
        return AssistantStep(
            text=message.content or "",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage=usage,
            raw={"id": response.id},
        )

    def append_tool_results(self, results) -> None:
        for call, result in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.output or "(no output)",
                }
            )

    def history(self) -> list:
        return json.loads(json.dumps(self.messages, default=str))
