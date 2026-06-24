"""Tests for agentapi.config.settings — anthropic_api_key field."""

import os
import pytest
from agentapi.config.settings import Settings, get_settings


def test_settings_has_anthropic_api_key_field():
    s = Settings(
        openai_api_key="sk-openai",
        gemini_api_key="sk-gemini",
        openrouter_api_key="sk-openrouter",
        anthropic_api_key="sk-anthropic",
        default_provider="openai",
    )
    assert s.anthropic_api_key == "sk-anthropic"


def test_settings_anthropic_api_key_defaults_to_none():
    s = Settings(
        openai_api_key=None,
        gemini_api_key=None,
        openrouter_api_key=None,
        anthropic_api_key=None,
        default_provider="openai",
    )
    assert s.anthropic_api_key is None


def test_get_settings_reads_anthropic_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    s = get_settings()
    assert s.anthropic_api_key == "sk-test-anthropic"


def test_get_settings_anthropic_api_key_none_when_env_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = get_settings()
    assert s.anthropic_api_key is None


def test_settings_is_frozen():
    s = Settings(
        openai_api_key=None,
        gemini_api_key=None,
        openrouter_api_key=None,
        anthropic_api_key=None,
        default_provider="openai",
    )
    with pytest.raises(Exception):
        s.anthropic_api_key = "should-fail"