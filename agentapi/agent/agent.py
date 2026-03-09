"""Primary Agent class implementation."""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Callable

from agentapi.agent.memory import ConversationMemory
from agentapi.agent.tools import ToolDefinition, parse_tool_args, to_tool_definition
from agentapi.config.settings import get_settings
from agentapi.errors import AgentConfigurationError
from agentapi.providers.base import BaseProvider, ToolCall
from agentapi.providers.gemini import GeminiProvider
from agentapi.providers.openai import OpenAIProvider
from agentapi.providers.openrouter import OpenRouterProvider


ProviderFactory = Callable[["Agent", Any, str], BaseProvider]


class Agent:
    """Stateful agent with provider abstraction, tools, memory, and streaming."""

    _custom_provider_factories: dict[str, ProviderFactory] = {}

    def __init__(
        self,
        *,
        system_prompt: str,
        provider: str | BaseProvider | None = None,
        model: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
    ) -> None:
        settings = get_settings()

        self.system_prompt = system_prompt
        if isinstance(provider, BaseProvider):
            self.provider_name = provider.__class__.__name__.lower()
        else:
            self.provider_name = (provider or settings.default_provider).lower()

        self.model = model or self._default_model_for(self.provider_name)
        self.tool_calling = self._default_tool_calling_for(self.provider_name)
        if tool_calling:
            self.tool_calling.update(tool_calling)
        self.memory = ConversationMemory(system_prompt=system_prompt)

        self._settings = settings
        self._provider: BaseProvider | None = provider if isinstance(provider, BaseProvider) else None
        self._tools: dict[str, ToolDefinition] = {}

        for func in tools or []:
            self.add_tool(func)

    def add_tool(self, func: Callable[..., Any]) -> None:
        """Register a callable tool on the agent."""

        definition = to_tool_definition(func)
        self._tools[definition.name] = definition

    def reset_memory(self) -> None:
        """Clear conversation state but preserve system prompt."""

        self.memory.reset(system_prompt=self.system_prompt)

    async def run(self, message: str, *, max_tool_rounds: int = 3) -> str:
        """Execute a chat completion with optional tool-calling loop."""

        self.memory.add({"role": "user", "content": message})
        provider = self._get_provider()

        for _ in range(max_tool_rounds + 1):
            response = await provider.chat(
                self.memory.messages,
                tools=self._tool_schemas(),
                tool_calling=self.tool_calling,
            )

            if response.tool_calls:
                self.memory.add(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                            for call in response.tool_calls
                        ],
                    }
                )

                await self._execute_tool_calls(response.tool_calls)
                continue

            self.memory.add({"role": "assistant", "content": response.content})
            return response.content

        fallback = "Tool loop reached max rounds without final response"
        self.memory.add({"role": "assistant", "content": fallback})
        return fallback

    async def stream(self, message: str) -> AsyncIterator[str]:
        """Stream model tokens and persist final assistant message."""

        self.memory.add({"role": "user", "content": message})
        provider = self._get_provider()

        collected: list[str] = []
        async for token in provider.stream(
            self.memory.messages,
            tools=self._tool_schemas(),
            tool_calling=self.tool_calling,
        ):
            collected.append(token)
            yield token

        full_text = "".join(collected)
        self.memory.add({"role": "assistant", "content": full_text})

    def _create_provider(self, settings: Any) -> BaseProvider:
        custom_factory = self._custom_provider_factories.get(self.provider_name)
        if custom_factory:
            provider = custom_factory(self, settings, self.model)
            if not isinstance(provider, BaseProvider):
                raise TypeError("Custom provider factory must return a BaseProvider instance")
            return provider

        if self.provider_name == "openai":
            return OpenAIProvider(
                api_key=self._require_api_key(settings.openai_api_key, "OPENAI_API_KEY"),
                model=self.model,
            )
        if self.provider_name == "gemini":
            return GeminiProvider(
                api_key=self._require_api_key(settings.gemini_api_key, "GEMINI_API_KEY"),
                model=self.model,
            )
        if self.provider_name == "openrouter":
            return OpenRouterProvider(
                api_key=self._require_api_key(settings.openrouter_api_key, "OPENROUTER_API_KEY"),
                model=self.model,
            )

        raise ValueError(
            "Unsupported provider. Use one of: openai, gemini, openrouter, or register a custom provider"
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

    def _require_api_key(self, value: str | None, env_name: str) -> str:
        if value and value.strip():
            return value

        raise AgentConfigurationError(
            f"Missing API key for provider '{self.provider_name}'. "
            f"Set {env_name} in your environment or .env file."
        )

    def _default_model_for(self, provider_name: str) -> str:
        if provider_name == "gemini":
            return "gemini-2.5-flash"
        return "gpt-4o-mini"

    def _default_tool_calling_for(self, provider_name: str) -> dict[str, Any]:
        if provider_name == "gemini":
            return {
                "mode": "AUTO",  # Gemini functionCallingConfig mode
            }

        # OpenAI-compatible defaults.
        return {
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        if not self._tools:
            return None
        return [tool.schema for tool in self._tools.values()]

    async def _execute_tool_calls(self, calls: list[ToolCall]) -> None:
        for call in calls:
            tool_def = self._tools.get(call.name)
            if not tool_def:
                self.memory.add(
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

            self.memory.add(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": output,
                }
            )
