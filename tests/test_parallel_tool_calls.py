"""Tests for concurrent tool-call execution."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from agentapi import Agent
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall


class FakeProvider(BaseProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(content="", tool_calls=[], raw_message={})

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


def test_execute_tool_calls_runs_async_tools_concurrently_in_call_order():
    events: list[str] = []

    async def slow_first() -> str:
        events.append("first:start")
        await asyncio.sleep(0.15)
        events.append("first:end")
        return "first"

    async def fast_second() -> str:
        events.append("second:start")
        await asyncio.sleep(0.01)
        events.append("second:end")
        return "second"

    async def medium_third() -> str:
        events.append("third:start")
        await asyncio.sleep(0.08)
        events.append("third:end")
        return "third"

    async def run_tool_calls() -> tuple[list[dict[str, Any]], float]:
        agent = Agent(
            system_prompt="test",
            provider=FakeProvider(),
            tools=[slow_first, fast_second, medium_third],
        )
        conversation_messages: list[dict[str, Any]] = []
        calls = [
            ToolCall(id="call_1", name="slow_first", arguments="{}"),
            ToolCall(id="call_2", name="fast_second", arguments="{}"),
            ToolCall(id="call_3", name="medium_third", arguments="{}"),
        ]

        started_at = time.perf_counter()
        await agent._execute_tool_calls(calls, conversation_messages)
        elapsed = time.perf_counter() - started_at
        return conversation_messages, elapsed

    messages, elapsed = asyncio.run(run_tool_calls())

    assert [message["tool_call_id"] for message in messages] == ["call_1", "call_2", "call_3"]
    assert [message["content"] for message in messages] == ["first", "second", "third"]
    assert events.index("second:start") < events.index("first:end")
    assert events.index("third:start") < events.index("first:end")
    assert elapsed < 0.30


def test_execute_tool_calls_honors_parallel_tool_calls_opt_out():
    events: list[str] = []

    async def slow_first() -> str:
        events.append("first:start")
        await asyncio.sleep(0.04)
        events.append("first:end")
        return "first"

    async def fast_second() -> str:
        events.append("second:start")
        await asyncio.sleep(0.01)
        events.append("second:end")
        return "second"

    async def run_tool_calls() -> list[dict[str, Any]]:
        agent = Agent(
            system_prompt="test",
            provider=FakeProvider(),
            tools=[slow_first, fast_second],
            tool_calling={"parallel_tool_calls": False},
        )
        conversation_messages: list[dict[str, Any]] = []
        calls = [
            ToolCall(id="call_1", name="slow_first", arguments="{}"),
            ToolCall(id="call_2", name="fast_second", arguments="{}"),
        ]

        await agent._execute_tool_calls(calls, conversation_messages)
        return conversation_messages

    messages = asyncio.run(run_tool_calls())

    assert [message["tool_call_id"] for message in messages] == ["call_1", "call_2"]
    assert [message["content"] for message in messages] == ["first", "second"]
    assert events == ["first:start", "first:end", "second:start", "second:end"]
