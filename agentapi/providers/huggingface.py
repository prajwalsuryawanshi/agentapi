"""Hugging Face provider implementation."""

from __future__ import annotations

from agentapi.providers.openai_compatible import OpenAICompatibleProvider


class HuggingFaceProvider(OpenAICompatibleProvider):
    """Provider for Hugging Face Inference API (Serverless or Dedicated Endpoints)."""

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        url = base_url or "https://api-inference.huggingface.co/v1"
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=url,
        )
