"""Anthropic provider implementation."""

import json
from typing import Any, AsyncIterator
from anthropic import AsyncAnthropic, APIStatusError, APIConnectionError, APITimeoutError

from agentapi.errors import AgentProviderError
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall

class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the Anthropic provider with an API key and model."""
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    def _format_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Extract the system prompt and format remaining messages for Anthropic."""
        system_prompt = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] in ("user", "assistant"):
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            elif msg["role"] == "tool":
                # Basic mapping for tool results
                anthropic_messages.append({
                    "role": "user", 
                    "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id"), "content": msg["content"]}]
                })
        
        return system_prompt, anthropic_messages

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Map standard OpenAI-style function schemas to Anthropic's tool schema."""
        if not tools:
            return None
        
        anthropic_tools = []
        for t in tools:
            func = t.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {})
            })
        return anthropic_tools

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_calling: dict[str, Any] | None = None) -> ProviderResponse:
        """Execute a single chat completion request against the Anthropic API."""
        system, formatted_messages = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)
        
        kwargs = {
            "model": self.model,
            "system": system,
            "messages": formatted_messages,
            "max_tokens": 4096,
        }
        if formatted_tools:
            kwargs["tools"] = formatted_tools
            
        try:
            response = await self.client.messages.create(**kwargs)
        except APIStatusError as exc:
            raise AgentProviderError(
                f"Anthropic provider error for model '{self.model}': {type(exc).__name__}",
                status_code=exc.status_code
            ) from None
        except (APIConnectionError, APITimeoutError) as exc:
            raise AgentProviderError(
                f"Anthropic provider error for model '{self.model}': {type(exc).__name__}",
                status_code=502
            ) from None
        
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input)
                ))
                
        return ProviderResponse(content=content, tool_calls=tool_calls, raw_message=response.model_dump())

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_calling: dict[str, Any] | None = None) -> AsyncIterator[str]:
        """Stream a chat completion request from the Anthropic API."""
        system, formatted_messages = self._format_messages(messages)
        
        kwargs = {
            "model": self.model,
            "system": system,
            "messages": formatted_messages,
            "max_tokens": 4096,
        }
        
        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except APIStatusError as exc:
            raise AgentProviderError(
                f"Anthropic stream provider error for model '{self.model}': {type(exc).__name__}",
                status_code=exc.status_code
            ) from None
        except (APIConnectionError, APITimeoutError) as exc:
            raise AgentProviderError(
                f"Anthropic stream provider error for model '{self.model}': {type(exc).__name__}",
                status_code=502
            ) from None