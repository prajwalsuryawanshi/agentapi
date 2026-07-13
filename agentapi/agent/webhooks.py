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

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
        return {"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}

    def __init__(
        self,
        endpoint_url: str,
        events: list[str] | None = None,
        max_retries: int = 3,
        secret_token: str | None = None,
    ) -> None:
        """
        Initialize the WebhookHook.
        
        WARNING: The `endpoint_url` should come from a trusted developer configuration
        or be heavily restricted. Accepting an endpoint_url from untrusted user input
        can lead to Server-Side Request Forgery (SSRF) vulnerabilities.
        """
        if httpx is None:
            raise ImportError("httpx is required for WebhookHook. Run: pip install httpx")
        
        self.endpoint_url = endpoint_url
        self.events = set(events) if events else None
        self.max_retries = max_retries
        self.secret_token = secret_token.encode("utf-8") if secret_token else None
        self._client = httpx.AsyncClient()
        self._background_tasks = set()

    def _dispatch_webhook(self, event_name: str, payload: dict[str, Any]) -> None:
        """Schedules the payload to be sent to the webhook endpoint in the background."""
        if self.events and event_name not in self.events:
            return 
            
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-Agent-Event": event_name}
        
        if self.secret_token:
            import time
            timestamp = str(int(time.time()))
            headers["X-Agent-Timestamp"] = timestamp
            # Combine timestamp and payload for signing to prevent replay attacks
            signed_content = f"{timestamp}:".encode("utf-8") + payload_bytes
            signature = hmac.new(self.secret_token, signed_content, hashlib.sha256).hexdigest()
            headers["X-Agent-Signature"] = f"sha256={signature}"

        async def _send_task():
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.post(self.endpoint_url, content=payload_bytes, headers=headers)
                    response.raise_for_status()
                    return
                except httpx.HTTPStatusError as e:
                    if 400 <= e.response.status_code < 500:
                        logger.error(f"Webhook {event_name} failed with client error {e.response.status_code}: {e}")
                        return
                    if attempt == self.max_retries:
                        logger.error(f"Webhook {event_name} failed after {self.max_retries} retries: {e}")
                    else:
                        await asyncio.sleep(2 ** attempt)
                except httpx.HTTPError as e:
                    if attempt == self.max_retries:
                        logger.error(f"Webhook {event_name} failed after {self.max_retries} retries: {e}")
                    else:
                        await asyncio.sleep(2 ** attempt)

        task = asyncio.create_task(_send_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def on_agent_start(self, run_id: str, message: str, **kwargs: Any) -> None:
        self._dispatch_webhook("on_agent_start", {"run_id": run_id, "message": message})

    async def on_llm_start(self, run_id: str, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        self._dispatch_webhook("on_llm_start", {"run_id": run_id, "messages": messages})

    async def on_llm_end(self, run_id: str, response: ProviderResponse, **kwargs: Any) -> None:
        tool_calls = [self._serialize_tool_call(t) for t in response.tool_calls] if response.tool_calls else []
        self._dispatch_webhook("on_llm_end", {
            "run_id": run_id, 
            "content": response.content,
            "tool_calls": tool_calls
        })

    async def on_tool_start(self, run_id: str, tool_call: ToolCall, **kwargs: Any) -> None:
        self._dispatch_webhook("on_tool_start", {
            "run_id": run_id,
            "tool_call": self._serialize_tool_call(tool_call)
        })

    async def on_tool_end(self, run_id: str, tool_call: ToolCall, result: str, **kwargs: Any) -> None:
        self._dispatch_webhook("on_tool_end", {
            "run_id": run_id,
            "tool_call": self._serialize_tool_call(tool_call),
            "result": result
        })

    async def on_agent_end(self, run_id: str, final_response: str | BaseModel, **kwargs: Any) -> None:
        content = final_response.model_dump() if isinstance(final_response, BaseModel) else final_response
        self._dispatch_webhook("on_agent_end", {"run_id": run_id, "final_response": content})

    async def on_agent_error(self, run_id: str, error: Exception, **kwargs: Any) -> None:
        error_name = type(error).__name__
        self._dispatch_webhook("on_agent_error", {"run_id": run_id, "error": error_name})
