"""Integration tests for provider tool-calling loops."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from agentapi import Agent, BaseProvider, tool
from agentapi.providers.base import ProviderResponse, ToolCall
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.openai import OpenAIProvider
from agentapi.providers.openrouter import OpenRouterProvider


class ScriptedToolProvider(BaseProvider):
    """Provider fixture that asks for a tool, then returns a final response."""

    def __init__(self, first_response: ProviderResponse) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = [
            first_response,
            ProviderResponse(
                content="The weather in Pune is sunny.",
                tool_calls=[],
                raw_message={"role": "assistant"},
            ),
        ]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.calls.append({"messages": messages, "tools": tools, "tool_calling": tool_calling})
        return self._responses.pop(0)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    return f"Weather in {city}: sunny"


@pytest.mark.asyncio
async def test_agent_executes_tool_call_and_returns_final_response() -> None:
    provider = ScriptedToolProvider(
        ProviderResponse(
            content="",
            tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"city": "Pune"}')],
            raw_message={"role": "assistant"},
        )
    )
    agent = Agent(system_prompt="You are a weather assistant", provider=provider, tools=[get_weather])

    result = await agent.run("How is the weather in Pune?")

    assert result == "The weather in Pune is sunny."
    assert len(provider.calls) == 2
    assert provider.calls[0]["tools"][0]["function"]["name"] == "get_weather"
    assert provider.calls[0]["tool_calling"] == {"tool_choice": "auto", "parallel_tool_calls": True}

    second_call_messages = provider.calls[1]["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert second_call_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "get_weather",
        "content": "Weather in Pune: sunny",
    }
    assert agent.memory.messages[-1]["content"] == "The weather in Pune is sunny."


@pytest.mark.asyncio
async def test_agent_returns_tool_failure_to_provider_before_final_response() -> None:
    provider = ScriptedToolProvider(
        ProviderResponse(
            content="",
            tool_calls=[ToolCall(id="call_bad", name="get_weather", arguments="{bad json")],
            raw_message={"role": "assistant"},
        )
    )
    agent = Agent(system_prompt="You are a weather assistant", provider=provider, tools=[get_weather])

    result = await agent.run("Use a malformed tool call")

    assert result == "The weather in Pune is sunny."
    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_bad"
    assert tool_message["content"].startswith("Tool execution failed:")


class FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in for deterministic provider tests."""

    def __init__(self, response_data: dict[str, Any], captured: dict[str, Any], **kwargs: Any) -> None:
        self._response_data = response_data
        self._captured = captured
        self._captured["client_kwargs"] = kwargs

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        self._captured.update({"url": url, "headers": headers, "params": params, "json": json})
        request = httpx.Request("POST", url)
        return httpx.Response(200, json=self._response_data, request=request)


def _patch_async_client(
    monkeypatch: pytest.MonkeyPatch,
    response_data: dict[str, Any],
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeAsyncClient:
        return FakeAsyncClient(response_data, captured, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_factory", "expected_url", "expected_header"),
    [
        (
            lambda: OpenAIProvider(api_key="test-key", model="gpt-test"),
            "https://api.openai.com/v1/chat/completions",
            None,
        ),
        (
            lambda: OpenRouterProvider(api_key="test-key", model="openrouter-test"),
            "https://openrouter.ai/api/v1/chat/completions",
            {"X-Title": "agentapi"},
        ),
    ],
)
async def test_openai_compatible_providers_send_tools_and_parse_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    provider_factory: Callable[[], BaseProvider],
    expected_url: str,
    expected_header: dict[str, str] | None,
) -> None:
    captured = _patch_async_client(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'},
                            }
                        ],
                    }
                }
            ]
        },
    )
    provider = provider_factory()
    tools = [get_weather.__agentapi_tool_schema__]

    response = await provider.chat(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tool_calling={"tool_choice": "auto", "parallel_tool_calls": True},
    )

    assert captured["url"] == expected_url
    assert captured["json"]["tools"] == tools
    assert captured["json"]["tool_choice"] == "auto"
    assert captured["json"]["parallel_tool_calls"] is True
    if expected_header:
        for key, value in expected_header.items():
            assert captured["headers"][key] == value
    assert response.tool_calls == [ToolCall(id="call_123", name="get_weather", arguments='{"city": "Pune"}')]


@pytest.mark.asyncio
async def test_gemini_provider_sends_function_declarations_and_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_async_client(
        monkeypatch,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "gemini_call_1",
                                    "name": "get_weather",
                                    "args": {"city": "Pune"},
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )
    provider = GeminiProvider(api_key="test-key", model="gemini-test")
    tools = [get_weather.__agentapi_tool_schema__]

    response = await provider.chat(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tool_calling={"mode": "AUTO"},
    )

    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    assert captured["params"] == {"key": "test-key"}
    assert captured["json"]["tools"][0]["function_declarations"][0]["name"] == "get_weather"
    assert captured["json"]["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    assert response.tool_calls == [
        ToolCall(id="gemini_call_1", name="get_weather", arguments=json.dumps({"city": "Pune"}))
    ]
