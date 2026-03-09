"""OpenAI provider implementation."""

from __future__ import annotations

from agentapi.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Provider for OpenAI Chat Completions API."""

    def __init__(self, *, api_key: str, model: str) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.openai.com/v1",
        )
