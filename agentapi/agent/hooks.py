"""Base classes and interfaces for the Agent Lifecycle Hooks system."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel
    from agentapi.providers.base import ProviderResponse, ToolCall


class AgentHook:
    """
    Base class for Agent lifecycle hooks.
    Subclass this and override specific methods to inject custom logic
    (like logging, tracing, metrics) into the agent's execution loop.
    """

    async def on_agent_start(self, run_id: str, message: str, **kwargs: Any) -> None:
        """Called when Agent.run() is first invoked."""
        pass

    async def on_llm_start(self, run_id: str, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        """Called immediately before the LLM provider is queried."""
        pass

    async def on_llm_end(self, run_id: str, response: "ProviderResponse", **kwargs: Any) -> None:
        """Called immediately after the LLM provider returns a response."""
        pass

    async def on_tool_start(self, run_id: str, tool_call: "ToolCall", **kwargs: Any) -> None:
        """Called before a specific tool is executed."""
        pass

    async def on_tool_end(self, run_id: str, tool_call: "ToolCall", result: str, **kwargs: Any) -> None:
        """Called after a specific tool has finished executing."""
        pass

    async def on_agent_end(self, run_id: str, final_response: str | "BaseModel", **kwargs: Any) -> None:
        """Called when Agent.run() successfully returns its final answer."""
        pass

    async def on_agent_error(self, run_id: str, error: Exception, **kwargs: Any) -> None:
        """Called if an unhandled exception occurs inside Agent.run()."""
        pass
