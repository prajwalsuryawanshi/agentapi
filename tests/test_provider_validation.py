# tests/test_provider_validation.py
"""
Tests for DEFAULT_PROVIDER validation in agentapi.config.settings.

Covers:
- Valid provider names are accepted
- Missing/empty values raise ValueError with helpful message
- Unsupported provider names raise ValueError with helpful message
- Roadmap providers (anthropic) produce a specific hint
- Case and whitespace normalization works correctly
"""

import pytest
from agentapi.config.settings import _validate_provider, SUPPORTED_PROVIDERS


class TestValidateProvider:

    # ── Valid inputs ──────────────────────────────────────────────────────

    def test_valid_openai(self):
        assert _validate_provider("openai") == "openai"

    def test_valid_gemini(self):
        assert _validate_provider("gemini") == "gemini"

    def test_valid_openrouter(self):
        assert _validate_provider("openrouter") == "openrouter"

    def test_case_insensitive_upper(self):
        """Provider names should be normalized to lowercase."""
        assert _validate_provider("OpenAI") == "openai"

    def test_case_insensitive_mixed(self):
        assert _validate_provider("GEMINI") == "gemini"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be ignored."""
        assert _validate_provider("  openai  ") == "openai"

    def test_all_supported_providers_accepted(self):
        """Every entry in SUPPORTED_PROVIDERS must pass validation."""
        for provider in SUPPORTED_PROVIDERS:
            assert _validate_provider(provider) == provider

    # ── Missing / empty inputs ────────────────────────────────────────────

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="DEFAULT_PROVIDER is not configured"):
            _validate_provider(None)

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="DEFAULT_PROVIDER is not configured"):
            _validate_provider("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="DEFAULT_PROVIDER is not configured"):
            _validate_provider("   ")

    # ── Unsupported providers ─────────────────────────────────────────────

    def test_unsupported_provider_raises_value_error(self):
        with pytest.raises(ValueError, match='Unsupported provider: "mistral"'):
            _validate_provider("mistral")

    def test_unsupported_provider_lists_valid_options(self):
        """Error message must list all supported providers."""
        with pytest.raises(ValueError) as exc_info:
            _validate_provider("cohere")
        error_message = str(exc_info.value)
        for provider in SUPPORTED_PROVIDERS:
            assert provider in error_message

    def test_typographical_error_hyphen(self):
        """open-ai should not be accepted."""
        with pytest.raises(ValueError):
            _validate_provider("open-ai")

    def test_typographical_error_space(self):
        """'Open AI' with space should not be accepted."""
        with pytest.raises(ValueError):
            _validate_provider("Open AI")

    def test_gemini_typo(self):
        with pytest.raises(ValueError):
            _validate_provider("gemni")

    # ── Roadmap provider hint ─────────────────────────────────────────────

    def test_anthropic_includes_roadmap_hint(self):
        """
        'anthropic' is on the roadmap — error should include a
        specific hint that it's coming but not yet available.
        """
        with pytest.raises(ValueError) as exc_info:
            _validate_provider("anthropic")
        assert "roadmap" in str(exc_info.value).lower() or \
               "not yet available" in str(exc_info.value)

    def test_claude_includes_roadmap_hint(self):
        with pytest.raises(ValueError) as exc_info:
            _validate_provider("claude")
        assert "not yet available" in str(exc_info.value)


class TestSupportedProvidersRegistry:

    def test_supported_providers_is_frozenset(self):
        assert isinstance(SUPPORTED_PROVIDERS, frozenset)

    def test_required_providers_present(self):
        """Core providers must always be in the registry."""
        assert "openai" in SUPPORTED_PROVIDERS
        assert "gemini" in SUPPORTED_PROVIDERS
        assert "openrouter" in SUPPORTED_PROVIDERS

    def test_anthropic_not_yet_supported(self):
        """Anthropic is roadmap — must NOT be in supported set yet."""
        assert "anthropic" not in SUPPORTED_PROVIDERS