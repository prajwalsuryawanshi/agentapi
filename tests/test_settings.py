"""Tests for settings validation."""

import pytest

from agentapi.config.settings import Settings, SUPPORTED_PROVIDERS


@pytest.mark.parametrize("provider", SUPPORTED_PROVIDERS)
def test_settings_accepts_supported_provider(provider):
    """Supported providers should be accepted and normalized to lowercase."""
    settings = Settings(
        openai_api_key=None,
        gemini_api_key=None,
        openrouter_api_key=None,
        default_provider=provider.upper(),
    )

    assert settings.default_provider == provider


def test_settings_rejects_invalid_provider():
    """Unsupported providers should fail fast with a clear error."""
    with pytest.raises(ValueError, match="Invalid default_provider"):
        Settings(
            openai_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            default_provider="claude",
        )
