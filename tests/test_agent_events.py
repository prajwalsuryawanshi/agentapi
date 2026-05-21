from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from agentapi.agent.agent import Agent
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall


class ToolLoopProvider(BaseProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="lookup_order",
                        arguments='{"order_id": "A-42"}',
                    )
                ],
                raw_message={},
            )
        return ProviderResponse(
            content="Order A-42 is shipped",
            tool_calls=[],
            raw_message={},
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"


class StreamingProvider(BaseProvider):
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
        yield "hello"
        yield " world"


def lookup_order(order_id: str) -> str:
    return f"{order_id}: shipped"


def test_agent_run_emits_provider_and_tool_events_without_payloads() -> None:
    events: list[dict[str, Any]] = []
    agent = Agent(
        system_prompt="You are helpful.",
        provider=ToolLoopProvider(),
        tools=[lookup_order],
        event_handler=events.append,
    )

    response = asyncio.run(agent.run("Where is A-42?"))

    assert response == "Order A-42 is shipped"
    assert [event["event"] for event in events] == [
        "provider_call_start",
        "provider_call_end",
        "tool_execution_start",
        "tool_execution_end",
        "provider_call_start",
        "provider_call_end",
    ]
    assert events[0]["mode"] == "run"
    assert events[1]["tool_call_count"] == 1
    assert events[2]["tool_name"] == "lookup_order"
    assert events[3]["output_length"] == len("A-42: shipped")
    assert all("messages" not in event for event in events)
    assert all("arguments" not in event for event in events)


def test_stream_response_emits_streaming_metrics() -> None:
    events: list[dict[str, Any]] = []
    agent = Agent(
        system_prompt="You are helpful.",
        provider=StreamingProvider(),
        event_handler=events.append,
    )

    async def collect() -> list[str]:
        response = agent.stream("Say hello")
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())

    assert "".join(chunks) == "data: hello\n\ndata:  world\n\n"
    assert [event["event"] for event in events] == [
        "provider_call_start",
        "provider_call_end",
    ]
    assert events[0]["mode"] == "stream"
    assert events[1]["token_count"] == 2
    assert events[1]["content_length"] == len("hello world")


def test_event_handler_errors_do_not_break_agent_run() -> None:
    def broken_handler(event: dict[str, Any]) -> None:
        raise RuntimeError(f"failed on {event['event']}")

    agent = Agent(
        system_prompt="You are helpful.",
        provider=ToolLoopProvider(),
        tools=[lookup_order],
        event_handler=broken_handler,
    )

    assert asyncio.run(agent.run("Where is A-42?")) == "Order A-42 is shipped"


def test_async_event_handler_is_supported() -> None:
    events: list[dict[str, Any]] = []

    async def async_handler(event: dict[str, Any]) -> None:
        events.append(event)

    agent = Agent(
        system_prompt="You are helpful.",
        provider=ToolLoopProvider(),
        tools=[lookup_order],
        event_handler=async_handler,
    )

    assert asyncio.run(agent.run("Where is A-42?")) == "Order A-42 is shipped"
    assert [event["event"] for event in events] == [
        "provider_call_start",
        "provider_call_end",
        "tool_execution_start",
        "tool_execution_end",
        "provider_call_start",
        "provider_call_end",
    ]
