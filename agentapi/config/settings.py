"""Environment-based settings for AgentAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _parse_fallback_providers(value: str | None) -> list[str]:
    if not value:
        return []
    return [provider.strip().lower() for provider in value.split(",") if provider.strip()]


@dataclass(frozen=True)
class Settings:
    """Minimal environment config used by providers."""

    openai_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    default_provider: str
    fallback_providers: list[str] = field(default_factory=list)


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        default_provider=os.getenv("DEFAULT_PROVIDER", "openai"),
        fallback_providers=_parse_fallback_providers(os.getenv("FALLBACK_PROVIDERS")),
    )