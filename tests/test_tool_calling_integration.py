"""Deterministic tool-calling loop integration tests."""

from __future__ import annotations

import json
import asyncio
from typing import Any

import httpx
import pytest

from agentapi import Agent, tool
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.openai import OpenAIProvider
from agentapi.providers.openrouter import OpenRouterProvider


def _json_request(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order status."""

    return f"order {order_id} ships tomorrow"


def _mock_async_client(monkeypatch: pytest.MonkeyPatch, module: Any, transport: httpx.MockTransport) -> None:
    original_async_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)


@pytest.mark.parametrize(
    ("provider_cls", "provider_module"),
    [
        (OpenAIProvider, "agentapi.providers.openai_compatible"),
        (OpenRouterProvider, "agentapi.providers.openai_compatible"),
    ],
)
def test_openai_compatible_provider_tool_loop_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    provider_cls: type[OpenAIProvider] | type[OpenRouterProvider],
    provider_module: str,
) -> None:
    """OpenAI and OpenRouter should parse tool calls and send tool results back."""

    asyncio.run(
        _run_openai_compatible_provider_tool_loop_round_trip(monkeypatch, provider_cls, provider_module)
    )


async def _run_openai_compatible_provider_tool_loop_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    provider_cls: type[OpenAIProvider] | type[OpenRouterProvider],
    provider_module: str,
) -> None:

    import importlib

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _json_request(request)
        requests.append(payload)

        if len(requests) == 1:
            assert payload["tools"][0]["function"]["name"] == "lookup_order"
            assert payload["tool_choice"] == "auto"
            assert payload["parallel_tool_calls"] is True
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-openai-1",
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
                },
            )

        assert payload["messages"][-2]["tool_calls"][0]["id"] == "call-openai-1"
        assert payload["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "call-openai-1",
            "name": "lookup_order",
            "content": "order A-42 ships tomorrow",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": "A-42 ships tomorrow."}}]})

    module = importlib.import_module(provider_module)
    _mock_async_client(monkeypatch, module, httpx.MockTransport(handler))

    agent = Agent(
        system_prompt="You resolve support questions.",
        provider=provider_cls(api_key="test-key", model="test-model"),
        tools=[lookup_order],
    )

    result = await agent.run("When does order A-42 ship?")

    assert result == "A-42 ships tomorrow."
    assert len(requests) == 2
    assert agent.memory.messages == [
        {"role": "user", "content": "When does order A-42 ship?"},
        {"role": "assistant", "content": "A-42 ships tomorrow."},
    ]


def test_gemini_provider_tool_loop_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini should translate Agent messages to functionCall/functionResponse payloads."""

    asyncio.run(_run_gemini_provider_tool_loop_round_trip(monkeypatch))


async def _run_gemini_provider_tool_loop_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:

    import agentapi.providers.gemini as gemini_module

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _json_request(request)
        requests.append(payload)

        if len(requests) == 1:
            declarations = payload["tools"][0]["function_declarations"]
            assert declarations[0]["name"] == "lookup_order"
            assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call-gemini-1",
                                            "name": "lookup_order",
                                            "args": {"order_id": "G-7"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

        parts = requests[-1]["contents"][-1]["parts"]
        assert {
            "functionResponse": {
                "name": "lookup_order",
                "response": {"result": "order G-7 ships tomorrow"},
                "id": "call-gemini-1",
            }
        } in parts
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "G-7 ships tomorrow."}]}}]},
        )

    _mock_async_client(monkeypatch, gemini_module, httpx.MockTransport(handler))

    agent = Agent(
        system_prompt="You resolve support questions.",
        provider=GeminiProvider(api_key="test-key", model="gemini-test"),
        tools=[lookup_order],
        tool_calling={"mode": "AUTO"},
    )

    result = await agent.run("When does order G-7 ship?")

    assert result == "G-7 ships tomorrow."
    assert len(requests) == 2


class ScriptedProvider(BaseProvider):
    """Provider fixture that returns a fixed sequence of responses."""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        self.requests.append(messages)
        return self.responses.pop(0)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ):
        if False:
            yield ""


def test_tool_loop_reports_malformed_tool_arguments() -> None:
    """Malformed provider tool arguments should be surfaced without aborting the loop."""

    asyncio.run(_run_tool_loop_reports_malformed_tool_arguments())


async def _run_tool_loop_reports_malformed_tool_arguments() -> None:

    provider = ScriptedProvider(
        [
            ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="bad-json-1", name="lookup_order", arguments="{bad json")],
                raw_message={},
            ),
            ProviderResponse(
                content="I could not call the tool.",
                tool_calls=[],
                raw_message={},
            ),
        ]
    )
    agent = Agent(system_prompt="You resolve support questions.", provider=provider, tools=[lookup_order])

    result = await agent.run("When does the order ship?")

    assert result == "I could not call the tool."
    tool_result = provider.requests[1][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "bad-json-1"
    assert tool_result["name"] == "lookup_order"
    assert tool_result["content"].startswith("Tool execution failed:")
