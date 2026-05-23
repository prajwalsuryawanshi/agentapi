"""Tests for Anthropic provider settings wiring."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from agentapi.agent.agent import Agent
from agentapi.config.settings import get_settings
from agentapi.errors import AgentConfigurationError
from agentapi.providers.base import BaseProvider


class FakeAnthropicProvider(BaseProvider):
    """Small provider double so tests do not require the anthropic package."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def chat(self, messages, tools=None, tool_calling=None):
        raise NotImplementedError

    async def stream(self, messages, tools=None, tool_calling=None):
        raise NotImplementedError


@pytest.fixture
def fake_anthropic_module(monkeypatch):
    module = types.ModuleType("agentapi.providers.anthropic")
    module.AnthropicProvider = FakeAnthropicProvider
    monkeypatch.setitem(sys.modules, "agentapi.providers.anthropic", module)


def test_get_settings_loads_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")

    settings = get_settings()

    assert settings.anthropic_api_key == "anthropic-test-key"


def test_anthropic_provider_uses_settings_api_key(fake_anthropic_module):
    agent = Agent(system_prompt="You are helpful.", provider="anthropic")

    provider = agent._create_provider(
        SimpleNamespace(anthropic_api_key="settings-anthropic-key")
    )

    assert isinstance(provider, FakeAnthropicProvider)
    assert provider.api_key == "settings-anthropic-key"


def test_anthropic_provider_reports_missing_key(fake_anthropic_module):
    agent = Agent(system_prompt="You are helpful.", provider="anthropic")

    with pytest.raises(AgentConfigurationError, match="ANTHROPIC_API_KEY"):
        agent._create_provider(SimpleNamespace(anthropic_api_key=None))
