"""Environment-based settings for AgentAPI."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Registry of currently supported provider names.
# When a new provider is implemented, add its name here.
# ─────────────────────────────────────────────
SUPPORTED_PROVIDERS: frozenset[str] = frozenset({
    "openai",
    "gemini",
    "openrouter",
})


def _validate_provider(value: str | None) -> str:
    """
    Validate a provider name against the supported provider registry.

    Raises a clear, actionable ValueError if:
    - The value is None or empty (DEFAULT_PROVIDER not set)
    - The value is not in SUPPORTED_PROVIDERS

    Returns the validated provider name (lowercased and stripped).
    """
    if not value or not value.strip():
        raise ValueError(
            "\n\nDEFAULT_PROVIDER is not configured.\n"
            "Please set it in your .env file.\n\n"
            "Supported providers:\n"
            + "\n".join(f"  - {p}" for p in sorted(SUPPORTED_PROVIDERS))
            + "\n\nExample:\n  DEFAULT_PROVIDER=openai\n"
        )

    normalized = value.strip().lower()

    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f'\n\nUnsupported provider: "{value}"\n\n'
            "Supported providers:\n"
            + "\n".join(f"  - {p}" for p in sorted(SUPPORTED_PROVIDERS))
            + "\n\nCheck your .env file and update DEFAULT_PROVIDER.\n"
            + (
                f'\nNote: "{normalized}" is on the roadmap but not yet available.\n'
                if normalized in {"anthropic", "claude", "mistral", "cohere"}
                else ""
            )
        )

    return normalized


class Settings:
    """
    Application settings loaded from environment variables.
    Provider validation is performed eagerly at instantiation time
    only when DEFAULT_PROVIDER is explicitly accessed.
    """

    def __init__(self) -> None:
        self._default_provider: str | None = None
        self._raw_provider: str | None = os.getenv("DEFAULT_PROVIDER")

    @property
    def default_provider(self) -> str:
        """
        Returns the validated default provider name.
        Validation is lazy — runs on first access so that importing
        settings does not break environments where provider is set inline.
        """
        if self._default_provider is None:
            self._default_provider = _validate_provider(self._raw_provider)
        return self._default_provider

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    @property
    def openrouter_api_key(self) -> str | None:
        return os.getenv("OPENROUTER_API_KEY")


# Global settings instance — used throughout the framework
settings = Settings()