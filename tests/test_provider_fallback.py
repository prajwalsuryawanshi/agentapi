import pytest

from agentapi.agent.agent import Agent, AgentAPIProviderError
from agentapi.providers.base import BaseProvider, ProviderResponse


class FailingProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        raise RuntimeError("primary failed")

    async def stream(self, messages, *, tools=None, tool_calling=None):
        raise RuntimeError("primary stream failed")
        yield


class WorkingProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(content="fallback worked", tool_calls=[], raw_message={})

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield "fallback "
        yield "worked"


class AlwaysFailProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        raise RuntimeError("provider failed")

    async def stream(self, messages, *, tools=None, tool_calling=None):
        raise RuntimeError("provider stream failed")
        yield


@pytest.mark.anyio
async def test_primary_provider_success_does_not_need_fallback():
    agent = Agent(system_prompt="test", provider=WorkingProvider())

    result = await agent.run("hello")

    assert result == "fallback worked"


@pytest.mark.anyio
async def test_fallback_provider_used_when_primary_fails():
    Agent.register_provider("primary_fail", lambda agent, settings, model: FailingProvider())
    Agent.register_provider("fallback_ok", lambda agent, settings, model: WorkingProvider())

    agent = Agent(system_prompt="test", provider="primary_fail")
    object.__setattr__(agent._settings, "fallback_providers", ["fallback_ok"])

    result = await agent.run("hello")

    assert result == "fallback worked"


@pytest.mark.anyio
async def test_error_raised_when_all_providers_fail():
    Agent.register_provider("primary_fail_all", lambda agent, settings, model: AlwaysFailProvider())
    Agent.register_provider("fallback_fail_all", lambda agent, settings, model: AlwaysFailProvider())

    agent = Agent(system_prompt="test", provider="primary_fail_all")
    object.__setattr__(agent._settings, "fallback_providers", ["fallback_fail_all"])

    with pytest.raises(AgentAPIProviderError):
        await agent.run("hello")


@pytest.mark.anyio
async def test_stream_fallback_provider_used_when_primary_fails():
    Agent.register_provider("primary_stream_fail", lambda agent, settings, model: FailingProvider())
    Agent.register_provider("fallback_stream_ok", lambda agent, settings, model: WorkingProvider())

    agent = Agent(system_prompt="test", provider="primary_stream_fail")
    object.__setattr__(agent._settings, "fallback_providers", ["fallback_stream_ok"])

    tokens = []
    async for token in agent._stream_generator("hello"):
        tokens.append(token)

    assert "".join(tokens) == "fallback worked"