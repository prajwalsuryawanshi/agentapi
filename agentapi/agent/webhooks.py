"""Webhook dispatching integration for Agent Hooks."""

import asyncio
import json
import logging
import hmac
import hashlib
from typing import Any
from pydantic import BaseModel
from agentapi.agent.hooks import AgentHook
from agentapi.providers.base import ProviderResponse, ToolCall

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


class WebhookHook(AgentHook):
    """
    An Agent Lifecycle Hook that dispatches events via HTTP POST to a remote endpoint.
    Features event filtering, exponential backoff retries, and HMAC payload signing.
    """

    def __init__(
        self,
        endpoint_url: str,
        events: list[str] | None = None,
        max_retries: int = 3,
        secret_token: str | None = None,
    ) -> None:
        if httpx is None:
            raise ImportError("httpx is required for WebhookHook. Run: pip install httpx")
        
        self.endpoint_url = endpoint_url
        self.events = set(events) if events else None
        self.max_retries = max_retries
        self.secret_token = secret_token.encode("utf-8") if secret_token else None

    async def _dispatch_webhook(self, event_name: str, payload: dict[str, Any]) -> None:
        """Sends the payload to the webhook endpoint with retries and exponential backoff."""
        if self.events and event_name not in self.events:
            return 
            
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Agent-Event": event_name}
        
        if self.secret_token:
            signature = hmac.new(self.secret_token, payload_bytes, hashlib.sha256).hexdigest()
            headers["X-Agent-Signature"] = f"sha256={signature}"

        async with httpx.AsyncClient() as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(self.endpoint_url, content=payload_bytes, headers=headers)
                    response.raise_for_status()
                    return
                except httpx.HTTPError as e:
                    if attempt == self.max_retries:
                        logger.error(f"Webhook {event_name} failed after {self.max_retries} retries: {e}")
                    else:
                        await asyncio.sleep(2 ** attempt)

    async def on_agent_start(self, run_id: str, message: str, **kwargs: Any) -> None:
        await self._dispatch_webhook("on_agent_start", {"run_id": run_id, "message": message})

    async def on_llm_start(self, run_id: str, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        await self._dispatch_webhook("on_llm_start", {"run_id": run_id, "messages": messages})

    async def on_llm_end(self, run_id: str, response: ProviderResponse, **kwargs: Any) -> None:
        tool_calls = [{"id": t.id, "name": t.name, "arguments": t.arguments} for t in response.tool_calls] if response.tool_calls else []
        await self._dispatch_webhook("on_llm_end", {
            "run_id": run_id, 
            "content": response.content,
            "tool_calls": tool_calls
        })

    async def on_tool_start(self, run_id: str, tool_call: ToolCall, **kwargs: Any) -> None:
        await self._dispatch_webhook("on_tool_start", {
            "run_id": run_id,
            "tool_call": {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}
        })

    async def on_tool_end(self, run_id: str, tool_call: ToolCall, result: str, **kwargs: Any) -> None:
        await self._dispatch_webhook("on_tool_end", {
            "run_id": run_id,
            "tool_call": {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments},
            "result": result
        })

    async def on_agent_end(self, run_id: str, final_response: str | BaseModel, **kwargs: Any) -> None:
        content = final_response.model_dump() if isinstance(final_response, BaseModel) else final_response
        await self._dispatch_webhook("on_agent_end", {"run_id": run_id, "final_response": content})

    async def on_agent_error(self, run_id: str, error: Exception, **kwargs: Any) -> None:
        await self._dispatch_webhook("on_agent_error", {"run_id": run_id, "error": str(error)})
