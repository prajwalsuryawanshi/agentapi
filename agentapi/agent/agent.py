"""Agent implementation for AgentAPI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agentapi.config.settings import SUPPORTED_PROVIDERS

from agentapi.agent.memory import InMemoryMemory, MemoryBackend
from agentapi.agent.tools import ToolDefinition, parse_tool_args, to_tool_definition
from agentapi.config.settings import get_settings
from agentapi.errors import AgentConfigurationError
from agentapi.providers.base import BaseProvider, ToolCall
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.openai import OpenAIProvider
from agentapi.providers.openrouter import OpenRouterProvider
from agentapi.providers.huggingface import HuggingFaceProvider

logger = logging.getLogger(__name__)

ProviderFactory = Callable[["Agent", Any, str], BaseProvider]


class AgentAPIUsageError(Exception):
    """Raised when the developer uses the AgentAPI incorrectly."""
    pass


class AgentAPIProviderError(Exception):
    """Raised when an upstream LLM provider call fails."""

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class Agent:
    """
    Main Agent class for interacting with LLM providers.
    
    Supports both built-in providers and custom registered providers.
    """
    
    # Class-level registry for custom providers
    _provider_registry: dict[str, callable] = {}
    
    def __init__(
        self,
        system_prompt: str,
        provider: str | "BaseProvider" = "openai",
        tools: list | None = None,
        model: str | None = None,
    ):
        # ── Provider name validation ──────────────────────────────────────
        # Only validate string providers. If a BaseProvider instance is
        # passed directly, skip name validation (custom provider path).
        if isinstance(provider, str):
            normalized = provider.strip().lower()
            if normalized not in SUPPORTED_PROVIDERS:
                # Check if it's a registered custom provider first
                if normalized not in self._provider_registry:
                    raise ValueError(
                        f'\n\nAgent received unsupported provider: "{provider}"\n\n'
                        "Built-in supported providers:\n"
                        + "\n".join(f"  - {p}" for p in sorted(SUPPORTED_PROVIDERS))
                        + "\n\nTo use a custom provider, register it first:\n"
                        "  Agent.register_provider('myprovider', factory_fn)\n"
                        "  Agent(provider='myprovider', ...)\n"
                    )
        # ── End validation ───────────────────────────────────────────────

        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model
        self._memory = Memory()
        self._provider_instance = self._resolve_provider(provider)
    
    @classmethod
    def register_provider(cls, name: str, factory_fn: callable) -> None:
        """
        Register a custom provider with the Agent class.
        
        Args:
            name: The provider name to register
            factory_fn: A callable that returns a BaseProvider instance
        """
        cls._provider_registry[name.lower()] = factory_fn
    
    def _resolve_provider(self, provider: str | BaseProvider) -> BaseProvider:
        """
        Resolve the provider string to a BaseProvider instance.
        
        Args:
            provider: Either a provider name string or a BaseProvider instance
            
        Returns:
            A BaseProvider instance
            
        Raises:
            ValueError: If the provider is not found or supported
        """
        from agentapi.providers.openai import OpenAIProvider
        from agentapi.providers.gemini import GeminiProvider
        from agentapi.providers.openrouter import OpenRouterProvider
        
        # If provider is already a BaseProvider instance, return it directly
        if not isinstance(provider, str):
            # Check if it's a BaseProvider instance
            from agentapi.providers.base import BaseProvider
            if isinstance(provider, BaseProvider):
                return provider
        
        # Provider is a string - resolve it
        provider_name = provider.lower()
        
        # Check built-in providers first
        if provider_name == "openai":
            return OpenAIProvider(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.model or "gpt-4",
            )
        elif provider_name == "gemini":
            return GeminiProvider(
                api_key=os.getenv("GEMINI_API_KEY"),
                model=self.model or "gemini-pro",
            )
        elif provider_name == "openrouter":
            return OpenRouterProvider(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model=self.model or "openai/gpt-3.5-turbo",
            )
        if self.provider_name == "huggingface":
            return HuggingFaceProvider(
                api_key=self._require_api_key(settings.huggingface_api_key, "HUGGINGFACE_API_KEY"),
                model=self.model,
                base_url=settings.huggingface_base_url,
            )
        if self.provider_name == "anthropic":
            from agentapi.providers.anthropic import AnthropicProvider
            import os
            # Use getattr to safely check settings, fallback to os env if settings doesn't have it yet
            api_key = getattr(settings, "anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
            return AnthropicProvider(
                api_key=self._require_api_key(api_key, "ANTHROPIC_API_KEY"),
                model=self.model,
            )
        raise ValueError(
            "Unsupported provider. Use one of: openai, gemini, openrouter, huggingface, anthropic or register a custom provider"
        )

    @classmethod
    def register_provider(cls, name: str, factory: ProviderFactory) -> None:
        """Register a custom provider factory for Agent(provider=<name>)."""
        provider_name = name.strip().lower()
        if not provider_name:
            raise ValueError("Provider name cannot be empty")
        cls._custom_provider_factories[provider_name] = factory

    def _get_provider(self) -> BaseProvider:
        if self._provider is None:
            self._provider = self._create_provider(self._settings)
        return self._provider


    def _default_model_for(self, provider_name: str) -> str:
        if provider_name == "gemini":
            return "gemini-2.5-flash"
        if provider_name == "anthropic":
            return "claude-3-5-sonnet-20241022"
        if provider_name == "huggingface":
            return "Qwen/Qwen2.5-72B-Instruct"
        return "gpt-4o-mini"

    def _default_tool_calling_for(self, provider_name: str) -> dict[str, Any]:
        if provider_name == "gemini":
            return {"mode": "AUTO"}
        return {
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        if not self._tools:
            return None
        return [tool.schema for tool in self._tools.values()]

    async def _execute_tool_calls(self, calls: list[ToolCall], conversation_messages: list[dict[str, Any]]) -> None:
        for call in calls:
            tool_def = self._tools.get(call.name)
            if not tool_def:
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": f"Tool '{call.name}' is not registered",
                    }
                )
                continue

            try:
                args = parse_tool_args(call.arguments)
                result = tool_def.func(**args)
                if inspect.isawaitable(result):
                    result = await result
                output = str(result)
            except Exception as exc:  # noqa: BLE001
                output = f"Tool execution failed: {exc}"

            conversation_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": output,
                }
            )
