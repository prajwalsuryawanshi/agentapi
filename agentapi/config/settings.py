"""Environment-based settings for AgentAPI."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"


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

    openai_api_key: str | None
    gemini_api_key: str | None
    openrouter_api_key: str | None
    huggingface_api_key: str | None
    huggingface_base_url: str | None
    default_provider: str

    def __post_init__(self) -> None:
        """Validate the default provider at initialization time."""
        provider = (self.default_provider or "").lower()

        try:
            provider = Provider(provider).value
        except ValueError:
            supported = ", ".join(p.value for p in Provider)
            raise ValueError(
                f"Invalid default_provider: {self.default_provider!r}. "
                f"Expected one of: {supported}."
            ) from None

        object.__setattr__(self, "default_provider", provider)


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
        huggingface_api_key=_validate_api_key(
            os.getenv("HUGGINGFACE_API_KEY"),
            "HuggingFace",
        ),
        huggingface_base_url=os.getenv("HUGGINGFACE_BASE_URL"),
        default_provider=os.getenv(
            "DEFAULT_PROVIDER",
            "openai",
        ),
    )
