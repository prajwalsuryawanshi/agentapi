"""Gemini provider implementation."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from agentapi.errors import AgentProviderError
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall


class GeminiProvider(BaseProvider):
    """Provider for Gemini generateContent and streamGenerateContent APIs."""

    def __init__(self, *, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("API key is required for provider initialization")

        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        payload = self._build_payload(messages, tools=tools, tool_calling=tool_calling)

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                raise self._map_http_status_error(exc) from exc
            except httpx.RequestError as exc:
                raise AgentProviderError(
                    f"Gemini network error for model '{self.model}': {exc}",
                    status_code=502,
                ) from exc

        content = self._extract_text(data)
        tool_calls = self._extract_tool_calls(data)
        return ProviderResponse(content=content, tool_calls=tool_calls, raw_message=data)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, tools=tools, tool_calling=tool_calling)
        last_text = ""

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/models/{self.model}:streamGenerateContent",
                    params={"alt": "sse", "key": self.api_key},
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        data = line[len("data:") :].strip()
                        if not data or data == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        token_text = self._extract_text(chunk)
                        # Some Gemini streaming responses are cumulative; emit
                        # only the newly added suffix when possible.
                        if token_text.startswith(last_text):
                            token = token_text[len(last_text) :]
                        else:
                            token = token_text

                        if token_text:
                            last_text = token_text

                        if token:
                            yield token
            except httpx.HTTPStatusError as exc:
                detail = await self._safe_error_detail(exc.response)
                raise self._map_http_status_error(exc, detail=detail) from exc
            except httpx.RequestError as exc:
                raise AgentProviderError(
                    f"Gemini stream network error for model '{self.model}': {exc}",
                    status_code=502,
                ) from exc

    def _map_http_status_error(
        self,
        exc: httpx.HTTPStatusError,
        *,
        detail: str | None = None,
    ) -> AgentProviderError:
        if detail is None:
            detail = self._safe_error_detail_sync(exc.response)
        status = exc.response.status_code

        if status == 404:
            return AgentProviderError(
                "Gemini model not found or unavailable for this API version/key. "
                f"Tried model '{self.model}'. Try setting model='gemini-2.5-flash' or another available model. "
                f"Response: {detail}",
                status_code=404,
            )

        return AgentProviderError(
            f"Gemini request failed ({status}) for model '{self.model}'. Response: {detail}",
            status_code=status,
        )

    async def _safe_error_detail(self, response: httpx.Response) -> str:
        try:
            raw = await response.aread()
            if raw:
                return raw.decode(errors="replace").strip()[:500]
        except Exception:
            pass
        return self._safe_error_detail_sync(response)

    def _safe_error_detail_sync(self, response: httpx.Response) -> str:
        try:
            text = response.text
            if text:
                return text.strip()[:500]
        except Exception:
            pass
        return f"{response.reason_phrase or 'Unknown error'}"

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_prompt: str | None = None
        contents: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")

            if role == "system":
                text = str(message.get("content") or "")
                if text:
                    system_prompt = text
                continue

            parts: list[dict[str, Any]] = []
            text = str(message.get("content") or "")
            if text:
                parts.append({"text": text})

            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function_obj = call.get("function") or {}
                    call_args = function_obj.get("arguments") or "{}"
                    try:
                        args_obj = json.loads(call_args)
                    except json.JSONDecodeError:
                        args_obj = {}

                    function_call: dict[str, Any] = {
                        "name": function_obj.get("name", ""),
                        "args": args_obj,
                    }

                    call_id = call.get("id")
                    if call_id:
                        function_call["id"] = call_id

                    parts.append({"functionCall": function_call})

            if role == "tool":
                result_obj = self._to_function_response_payload(message.get("content"))
                function_response: dict[str, Any] = {
                    "name": message.get("name") or "tool",
                    "response": result_obj,
                }

                tool_call_id = message.get("tool_call_id")
                if tool_call_id:
                    function_response["id"] = tool_call_id

                parts.append({"functionResponse": function_response})

            if not parts:
                continue

            gemini_role = "model" if role == "assistant" else "user"

            contents.append(
                {
                    "role": gemini_role,
                    "parts": parts,
                }
            )

        payload: dict[str, Any] = {"contents": contents}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            declarations = self._to_function_declarations(tools)
            if declarations:
                payload["tools"] = [{"function_declarations": declarations}]

                tool_calling = tool_calling or {}
                function_calling_config: dict[str, Any] = {}

                mode = tool_calling.get("mode")
                if mode:
                    function_calling_config["mode"] = str(mode).upper()

                allowed_function_names = tool_calling.get("allowed_function_names")
                if allowed_function_names:
                    function_calling_config["allowedFunctionNames"] = list(allowed_function_names)

                if function_calling_config:
                    payload["toolConfig"] = {
                        "functionCallingConfig": function_calling_config,
                    }

        return payload

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]:
        candidates = data.get("candidates") or []
        if not candidates:
            return []

        parts = candidates[0].get("content", {}).get("parts") or []
        result: list[ToolCall] = []

        for part in parts:
            function_call = part.get("functionCall") or part.get("function_call")
            if not function_call:
                continue

            result.append(
                ToolCall(
                    id=function_call.get("id") or uuid.uuid4().hex,
                    name=function_call.get("name") or "",
                    arguments=json.dumps(function_call.get("args") or {}),
                )
            )

        return result

    def _to_function_declarations(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []

        for tool in tools:
            function_schema = tool.get("function")
            if not function_schema:
                continue

            declaration = {
                "name": function_schema.get("name", ""),
                "description": function_schema.get("description", ""),
                "parameters": function_schema.get("parameters", {"type": "object", "properties": {}}),
            }
            declarations.append(declaration)

        return declarations

    def _to_function_response_payload(self, content: Any) -> dict[str, Any]:
        if content is None:
            return {"result": None}

        if isinstance(content, (dict, list, int, float, bool)):
            return {"result": content}

        text = str(content)
        try:
            parsed = json.loads(text)
            return {"result": parsed}
        except json.JSONDecodeError:
            return {"result": text}

    def _extract_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts") or []
        tokens = [part.get("text", "") for part in parts if part.get("text")]
        return "".join(tokens)
