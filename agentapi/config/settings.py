"""Environment-based settings for AgentAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Minimal environment config used by providers."""

    openai_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    default_provider: str


def _validate_api_key(key: str | None, provider_name: str) -> str | None:
    """Validate and normalize API keys."""
    
    if key is None:
        return None

    key = key.strip()

    if not key:
        raise ValueError(
            f"{provider_name} API key is configured but empty. "
            f"Check your environment variables or .env file."
        )

    if len(key) < 8 and key is not None:
        raise ValueError(
            f"{provider_name} API key appears invalid (too short)."
        )

    return key


def get_settings() -> Settings:
    return Settings(
        openai_api_key=_validate_api_key(
            os.getenv("OPENAI_API_KEY"),
            "OpenAI",
        ),
        gemini_api_key=_validate_api_key(
            os.getenv("GEMINI_API_KEY"),
            "Gemini",
        ),
        openrouter_api_key=_validate_api_key(
            os.getenv("OPENROUTER_API_KEY"),
            "OpenRouter",
        ),
        default_provider=os.getenv(
            "DEFAULT_PROVIDER",
            "openai",
        ),
    )
