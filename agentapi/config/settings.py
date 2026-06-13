"""Environment-based settings for AgentAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()

SUPPORTED_PROVIDERS = ("openai", "gemini", "openrouter")


@dataclass(frozen=True)
class Settings:
    """Minimal environment config used by providers."""

    openai_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    default_provider: str

    def __post_init__(self) -> None:
        """Validate the default provider at initialization time."""
        provider = (self.default_provider or "").lower()

        if provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(
                f"Invalid default_provider: {self.default_provider!r}. "
                f"Expected one of: {supported}."
            )

        object.__setattr__(self, "default_provider", provider)


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        default_provider=os.getenv("DEFAULT_PROVIDER", "openai"),
    )
