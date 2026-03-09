"""Shared implementation for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from agentapi.errors import AgentProviderError
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall


class OpenAICompatibleProvider(BaseProvider):
    """Minimal async provider for /chat/completions compatible APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for provider initialization")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        tool_calling = tool_calling or {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_calling.get("tool_choice", "auto")
            if "parallel_tool_calls" in tool_calling:
                payload["parallel_tool_calls"] = bool(tool_calling["parallel_tool_calls"])

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip()[:500]
                raise AgentProviderError(
                    f"Provider request failed ({exc.response.status_code}) for model '{self.model}'. "
                    f"Response: {detail}",
                    status_code=exc.response.status_code,
                ) from exc
            except httpx.RequestError as exc:
                raise AgentProviderError(
                    f"Provider network error for model '{self.model}': {exc}",
                    status_code=502,
                ) from exc

        message = data["choices"][0]["message"]
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=call.get("id", ""),
                name=call.get("function", {}).get("name", ""),
                arguments=call.get("function", {}).get("arguments", "{}"),
            )
            for call in raw_tool_calls
        ]

        return ProviderResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            raw_message=message,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        tool_calling = tool_calling or {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_calling.get("tool_choice", "auto")
            if "parallel_tool_calls" in tool_calling:
                payload["parallel_tool_calls"] = bool(tool_calling["parallel_tool_calls"])

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue

                        delta = choices[0].get("delta") or {}
                        token = delta.get("content")
                        if token:
                            yield token
            except httpx.HTTPStatusError as exc:
                detail = await self._safe_error_detail(exc.response)
                raise AgentProviderError(
                    f"Provider stream request failed ({exc.response.status_code}) for model '{self.model}'. "
                    f"Response: {detail}",
                    status_code=exc.response.status_code,
                ) from exc
            except httpx.RequestError as exc:
                raise AgentProviderError(
                    f"Provider stream network error for model '{self.model}': {exc}",
                    status_code=502,
                ) from exc

    async def _safe_error_detail(self, response: httpx.Response) -> str:
        try:
            raw = await response.aread()
            if raw:
                return raw.decode(errors="replace").strip()[:500]
        except Exception:
            pass

        try:
            text = response.text
            if text:
                return text.strip()[:500]
        except Exception:
            pass

        return response.reason_phrase or "Unknown error"
