"""Integration coverage for provider tool-calling payloads and loops."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from agentapi import Agent, tool
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.openai import OpenAIProvider
from agentapi.providers.openrouter import OpenRouterProvider


class FakeToolProvider(BaseProvider):
    """Provider fixture that asks for one tool call and then returns a final answer."""

    def __init__(self, tool_call: ToolCall) -> None:
        self.tool_call = tool_call
        self.requests: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_calling": tool_calling,
            }
        )
        if len(self.requests) == 1:
            return ProviderResponse(content="", tool_calls=[self.tool_call], raw_message={})
        return ProviderResponse(content="final answer", tool_calls=[], raw_message={})

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeAsyncClient:
    requests: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        self.requests.append({"url": url, **kwargs})
        return FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_order",
                                        "arguments": json.dumps({"order_id": "A-42"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )


@tool
def add(left: int, right: int) -> int:
    """Add two numbers."""

    return left + right


@pytest.mark.asyncio
async def test_agent_executes_tool_call_round_trip() -> None:
    provider = FakeToolProvider(
        ToolCall(id="call-add", name="add", arguments=json.dumps({"left": 2, "right": 3}))
    )
    agent = Agent(system_prompt="Use tools when helpful", provider=provider, tools=[add])

    result = await agent.run("What is 2 + 3?")

    assert result == "final answer"
    assert provider.requests[0]["tools"][0]["function"]["name"] == "add"
    second_request_messages = provider.requests[1]["messages"]
    assert second_request_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-add",
        "name": "add",
        "content": "5",
    }


@pytest.mark.asyncio
async def test_agent_records_tool_execution_failure_and_continues() -> None:
    provider = FakeToolProvider(ToolCall(id="call-bad", name="add", arguments="{bad json"))
    agent = Agent(system_prompt="Use tools when helpful", provider=provider, tools=[add])

    result = await agent.run("Try a malformed call")

    assert result == "final answer"
    tool_message = provider.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-bad"
    assert tool_message["content"].startswith("Tool execution failed:")


@pytest.mark.asyncio
async def test_agent_reports_unregistered_tool_call() -> None:
    provider = FakeToolProvider(ToolCall(id="call-missing", name="missing_tool", arguments="{}"))
    agent = Agent(system_prompt="Use tools when helpful", provider=provider, tools=[add])

    await agent.run("Call an unavailable tool")

    assert provider.requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-missing",
        "name": "missing_tool",
        "content": "Tool 'missing_tool' is not registered",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        OpenAIProvider(api_key="test-key", model="gpt-test"),
        OpenRouterProvider(api_key="test-key", model="openrouter-test"),
    ],
)
async def test_openai_compatible_providers_send_tool_schema_and_parse_call(
    monkeypatch: pytest.MonkeyPatch,
    provider: OpenAIProvider | OpenRouterProvider,
) -> None:
    FakeAsyncClient.requests = []
    monkeypatch.setattr("agentapi.providers.openai_compatible.httpx.AsyncClient", FakeAsyncClient)

    response = await provider.chat(
        [{"role": "user", "content": "Find order A-42"}],
        tools=[add.__agentapi_tool_schema__],
        tool_calling={"tool_choice": "auto", "parallel_tool_calls": False},
    )

    request = FakeAsyncClient.requests[0]
    assert request["json"]["tools"][0]["function"]["name"] == "add"
    assert request["json"]["tool_choice"] == "auto"
    assert request["json"]["parallel_tool_calls"] is False
    assert response.tool_calls == [
        ToolCall(id="call-1", name="lookup_order", arguments=json.dumps({"order_id": "A-42"}))
    ]


def test_gemini_provider_builds_tool_payload_and_extracts_function_calls() -> None:
    provider = GeminiProvider(api_key="test-key", model="gemini-test")

    payload = provider._build_payload(
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Find order A-42"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "add", "arguments": json.dumps({"left": 2, "right": 3})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "add", "content": "5"},
        ],
        tools=[add.__agentapi_tool_schema__],
        tool_calling={"mode": "auto", "allowed_function_names": ["add"]},
    )

    assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful"
    assert payload["tools"][0]["function_declarations"][0]["name"] == "add"
    assert payload["toolConfig"]["functionCallingConfig"] == {
        "mode": "AUTO",
        "allowedFunctionNames": ["add"],
    }
    assert payload["contents"][1]["parts"][0]["functionCall"]["name"] == "add"
    assert payload["contents"][2]["parts"][1]["functionResponse"]["response"] == {"result": 5}

    calls = provider._extract_tool_calls(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"id": "call-2", "name": "add", "args": {"left": 4, "right": 6}}}
                        ]
                    }
                }
            ]
        }
    )

    assert calls == [ToolCall(id="call-2", name="add", arguments=json.dumps({"left": 4, "right": 6}))]
