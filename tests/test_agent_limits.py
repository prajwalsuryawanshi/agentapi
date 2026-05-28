"""Tests for Agent input and history safety limits."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest
from fastapi.testclient import TestClient

from agentapi import Agent, AgentAPI, InMemoryMemory
from agentapi.errors import AgentPayloadTooLargeError
from agentapi.providers.base import BaseProvider, ProviderResponse


class RecordingProvider(BaseProvider):
    """Provider double that records payloads without making network calls."""

    def __init__(self) -> None:
        self.chat_calls = 0
        self.stream_calls = 0
        self.last_messages: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.chat_calls += 1
        self.last_messages = list(messages)
        return ProviderResponse(content="ok", tool_calls=[], raw_message={})

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        self.last_messages = list(messages)

        async def _tokens() -> AsyncIterator[str]:
            yield "ok"

        return _tokens()


def test_oversized_run_rejects_before_provider_call() -> None:
    provider = RecordingProvider()
    agent = Agent(system_prompt="safe", provider=provider, max_message_chars=5)

    with pytest.raises(AgentPayloadTooLargeError) as exc:
        asyncio.run(agent.run("too long"))

    assert exc.value.status_code == 413
    assert provider.chat_calls == 0


def test_disabled_message_limit_allows_large_input() -> None:
    provider = RecordingProvider()
    agent = Agent(system_prompt="safe", provider=provider, max_message_chars=None)

    response = asyncio.run(agent.run("x" * 10_000))

    assert response == "ok"
    assert provider.chat_calls == 1


def test_history_limit_preserves_system_prompt_and_recent_messages() -> None:
    memory = InMemoryMemory()
    for index in range(5):
        memory.add({"role": "user", "content": f"message {index}"})

    provider = RecordingProvider()
    agent = Agent(
        system_prompt="system stays",
        memory=memory,
        provider=provider,
        max_history_messages=2,
    )

    asyncio.run(agent.run("new"))

    assert provider.last_messages == [
        {"role": "system", "content": "system stays"},
        {"role": "user", "content": "message 3"},
        {"role": "user", "content": "message 4"},
        {"role": "user", "content": "new"},
    ]


def test_history_limit_none_keeps_full_history() -> None:
    memory = InMemoryMemory()
    for index in range(3):
        memory.add({"role": "user", "content": f"message {index}"})

    provider = RecordingProvider()
    agent = Agent(
        system_prompt="system stays",
        memory=memory,
        provider=provider,
        max_history_messages=None,
    )

    asyncio.run(agent.run("new"))

    assert provider.last_messages == [
        {"role": "system", "content": "system stays"},
        {"role": "user", "content": "message 0"},
        {"role": "user", "content": "message 1"},
        {"role": "user", "content": "message 2"},
        {"role": "user", "content": "new"},
    ]


def test_stream_rejects_before_provider_call() -> None:
    provider = RecordingProvider()
    agent = Agent(system_prompt="safe", provider=provider, max_message_chars=3)

    with pytest.raises(AgentPayloadTooLargeError):
        agent.stream("four")

    assert provider.stream_calls == 0


def test_chat_endpoint_returns_413_for_oversized_input() -> None:
    provider = RecordingProvider()
    agent = Agent(system_prompt="safe", provider=provider, max_message_chars=3)
    app = AgentAPI()

    @app.chat("/chat")
    async def chat(message: str) -> str:
        return await agent.run(message)

    response = TestClient(app).post("/chat", params={"message": "four"})

    assert response.status_code == 413
    assert response.json()["error"].startswith("Message is 4 characters")
    assert provider.chat_calls == 0
