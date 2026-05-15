"""Structured event hook coverage for provider and tool lifecycle events."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from agentapi import Agent, BaseProvider, tool
from agentapi.providers.base import ProviderResponse, ToolCall


class FakeProvider(BaseProvider):
    def __init__(self, responses: list[ProviderResponse] | None = None, *, should_fail: bool = False) -> None:
        self.responses = responses or []
        self.should_fail = should_fail

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        if self.should_fail:
            raise RuntimeError("provider unavailable")
        return self.responses.pop(0)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if self.should_fail:
            raise RuntimeError("stream unavailable")
        yield "hel"
        yield "lo"


@tool
def add(left: int, right: int) -> int:
    """Add two numbers."""

    return left + right


def test_run_emits_provider_and_tool_events_without_payloads() -> None:
    events: list[dict[str, Any]] = []
    provider = FakeProvider(
        [
            ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="call_1", name="add", arguments='{"left": 2, "right": 3}')],
                raw_message={},
            ),
            ProviderResponse(content="done", tool_calls=[], raw_message={}),
        ]
    )
    agent = Agent(
        system_prompt="You are helpful",
        provider=provider,
        tools=[add],
        event_hooks=[events.append],
    )

    result = asyncio.run(agent.run("please add 2 and 3"))

    assert result == "done"
    event_names = [event["event"] for event in events]
    assert event_names == [
        "provider.chat.start",
        "provider.chat.end",
        "tool.call.start",
        "tool.call.end",
        "provider.chat.start",
        "provider.chat.end",
    ]
    assert events[0]["provider"] == "fakeprovider"
    assert events[0]["model"] == "gpt-4o-mini"
    assert events[0]["tool_names"] == ["add"]
    assert events[1]["tool_call_count"] == 1
    assert events[3]["tool_name"] == "add"
    assert events[3]["output_length"] == 1
    assert all("content" not in event for event in events)
    assert all("arguments" not in event for event in events)


def test_async_event_hooks_are_supported() -> None:
    events: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        events.append(event)

    agent = Agent(
        system_prompt="You are helpful",
        provider=FakeProvider([ProviderResponse(content="ok", tool_calls=[], raw_message={})]),
    )
    agent.add_event_hook(collect)

    assert asyncio.run(agent.run("hello")) == "ok"
    assert [event["event"] for event in events] == ["provider.chat.start", "provider.chat.end"]


def test_failing_event_hook_does_not_break_agent_execution() -> None:
    events: list[dict[str, Any]] = []
    failures: list[str] = []

    def fail(event: dict[str, Any]) -> None:
        failures.append(event["event"])
        raise RuntimeError("hook unavailable")

    agent = Agent(
        system_prompt="You are helpful",
        provider=FakeProvider([ProviderResponse(content="ok", tool_calls=[], raw_message={})]),
        event_hooks=[fail, events.append],
    )

    assert asyncio.run(agent.run("hello")) == "ok"
    assert failures == ["provider.chat.start", "provider.chat.end"]
    assert [event["event"] for event in events] == ["provider.chat.start", "provider.chat.end"]


def test_provider_errors_emit_structured_error_event() -> None:
    events: list[dict[str, Any]] = []
    agent = Agent(
        system_prompt="You are helpful",
        provider=FakeProvider(should_fail=True),
        event_hooks=[events.append],
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(agent.run("hello"))

    assert [event["event"] for event in events] == ["provider.chat.start", "provider.chat.error"]
    assert events[-1]["error_type"] == "RuntimeError"
    assert "duration_ms" in events[-1]


def test_stream_emits_latency_and_token_counts() -> None:
    events: list[dict[str, Any]] = []
    agent = Agent(
        system_prompt="You are helpful",
        provider=FakeProvider(),
        event_hooks=[events.append],
    )

    async def collect_stream() -> str:
        chunks = []
        async for token in agent.stream("hello"):
            chunks.append(token)
        return "".join(chunks)

    assert asyncio.run(collect_stream()) == "hello"
    assert [event["event"] for event in events] == ["provider.stream.start", "provider.stream.end"]
    assert events[-1]["token_count"] == 2
    assert events[-1]["content_length"] == 5
