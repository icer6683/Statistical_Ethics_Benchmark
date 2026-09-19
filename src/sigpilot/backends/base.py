"""Backend interface shared by the tested-model providers.

A backend owns the provider-specific message history. The runner only sends user messages,
asks for one assistant step at a time, and hands back tool results — so the actual
assistant replies and actual tool outputs stay in the history verbatim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class AssistantStep:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class Backend:
    """Provider adapter. Subclasses keep history in their own native format."""

    provider: str = "base"

    def __init__(self, model: str, system: str, tools: list[dict], max_tokens: int = 8000):
        self.model = model
        self.system = system
        self.tools = tools
        self.max_tokens = max_tokens

    # --- conversation ---------------------------------------------------
    def append_user(self, text: str) -> None:
        raise NotImplementedError

    def step(self) -> AssistantStep:
        raise NotImplementedError

    def append_tool_results(self, results: list[tuple[ToolCall, object]]) -> None:
        raise NotImplementedError

    def history(self) -> list:
        raise NotImplementedError

    # --- helpers ---------------------------------------------------------
    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "max_tokens": self.max_tokens}


def with_retries(call, attempts: int = 4, base_delay: float = 2.0):
    """Retry transient API failures with exponential backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # provider SDKs raise their own types; treat uniformly
            message = f"{type(exc).__name__}: {exc}"
            retryable = any(
                token in message.lower()
                for token in ("rate limit", "429", "overloaded", "timeout", "connection", "500", "502", "503", "529")
            )
            last = exc
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    raise last  # pragma: no cover
