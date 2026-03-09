"""OpenRouter provider implementation."""

from __future__ import annotations

from agentapi.providers.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """Provider for OpenRouter OpenAI-compatible API."""

    def __init__(self, *, api_key: str, model: str) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={
                "HTTP-Referer": "https://github.com",
                "X-Title": "agentapi",
            },
        )
