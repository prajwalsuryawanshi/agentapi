"""Tests for tool-call authorization hooks."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from agentapi import Agent, tool
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall


class ToolCallingProvider(BaseProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.calls += 1
        self.messages.append(list(messages))
        if self.calls == 1:
            return ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="call-1", name="dangerous_action", arguments='{"value": "x"}')],
                raw_message={},
            )
        return ProviderResponse(content="final", tool_calls=[], raw_message={})

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


def test_authorize_tool_allows_execution() -> None:
    executed: list[str] = []
    provider = ToolCallingProvider()

    @tool
    def dangerous_action(value: str) -> str:
        executed.append(value)
        return f"ran:{value}"

    agent = Agent(
        system_prompt="safe",
        provider=provider,
        tools=[dangerous_action],
        authorize_tool=lambda call: call.name == "dangerous_action",
    )

    response = asyncio.run(agent.run("please act"))

    assert response == "final"
    assert executed == ["x"]
    assert provider.messages[1][-1]["content"] == "ran:x"


def test_authorize_tool_denies_with_custom_message() -> None:
    executed: list[str] = []
    provider = ToolCallingProvider()

    @tool
    def dangerous_action(value: str) -> str:
        executed.append(value)
        return f"ran:{value}"

    agent = Agent(
        system_prompt="safe",
        provider=provider,
        tools=[dangerous_action],
        authorize_tool=lambda call: "Tool blocked by policy.",
    )

    response = asyncio.run(agent.run("please act"))

    assert response == "final"
    assert executed == []
    assert provider.messages[1][-1]["content"] == "Tool blocked by policy."


def test_authorize_tool_false_denies_with_default_message() -> None:
    executed: list[str] = []
    provider = ToolCallingProvider()

    @tool
    def dangerous_action(value: str) -> str:
        executed.append(value)
        return f"ran:{value}"

    agent = Agent(
        system_prompt="safe",
        provider=provider,
        tools=[dangerous_action],
        authorize_tool=lambda call: False,
    )

    response = asyncio.run(agent.run("please act"))

    assert response == "final"
    assert executed == []
    assert provider.messages[1][-1]["content"] == "Tool call 'dangerous_action' was denied by authorize_tool."
