"""Agent implementation for AgentAPI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agentapi.config.settings import SUPPORTED_PROVIDERS

if TYPE_CHECKING:
    from agentapi.providers.base import BaseProvider


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
        # Check custom providers
        elif provider_name in self._provider_registry:
            return self._provider_registry[provider_name]()
        else:
            raise ValueError(
                f'\n\nProvider "{provider}" not found.\n\n'
                "Built-in supported providers:\n"
                + "\n".join(f"  - {p}" for p in sorted(SUPPORTED_PROVIDERS))
                + "\n\nTo use a custom provider, register it first:\n"
                "  Agent.register_provider('myprovider', factory_fn)\n"
                "  Agent(provider='myprovider', ...)\n"
            )


class Memory:
    """Simple memory class for storing conversation history."""
    
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> list:
        """Get all messages in the conversation history."""
        return self.messages