"""Backend registry for tested models and for the judge."""

from __future__ import annotations

from .base import AssistantStep, Backend, ToolCall

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-nano",
    "scripted": "scripted-endorsing",
}

# Which provider judges a run by default, so a model never judges itself.
CROSS_PROVIDER_JUDGE = {"anthropic": "openai", "openai": "anthropic", "scripted": "anthropic"}

DEFAULT_JUDGE_MODELS = {"anthropic": "claude-opus-5", "openai": "gpt-5"}


def make_backend(provider: str, model: str | None = None, **kwargs) -> Backend:
    model = model or DEFAULT_MODELS[provider]
    if provider == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model, **kwargs)
    if provider == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(model=model, **kwargs)
    if provider == "scripted":
        from .scripted import ScriptedBackend

        return ScriptedBackend(model=model, **kwargs)
    raise ValueError(f"unknown provider: {provider}")


__all__ = [
    "AssistantStep",
    "Backend",
    "ToolCall",
    "make_backend",
    "DEFAULT_MODELS",
    "DEFAULT_JUDGE_MODELS",
    "CROSS_PROVIDER_JUDGE",
]
