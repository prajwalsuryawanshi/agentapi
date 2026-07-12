"""Primary Agent class implementation."""

from __future__ import annotations

import inspect
import logging
import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Callable

from fastapi.responses import StreamingResponse

from agentapi.agent.hooks import AgentHook
from agentapi.agent.memory import InMemoryMemory, MemoryBackend
from agentapi.agent.tools import ToolDefinition, parse_tool_args, to_tool_definition
from agentapi.config.settings import get_settings
from agentapi.errors import AgentConfigurationError, AgentSchemaValidationError
from agentapi.providers.base import BaseProvider, ToolCall
from pydantic import BaseModel, ValidationError
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
    """Stateful agent with provider abstraction, tools, memory, and streaming."""

    _custom_provider_factories: dict[str, ProviderFactory] = {}

    def __init__(
        self,
        *,
        system_prompt: str,
        memory: MemoryBackend | None = None,
        provider: str | BaseProvider | None = None,
        model: str | None = None,
        tools: list[Callable[..., Any]] | None = None,
        tool_calling: dict[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
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
        self.memory = memory or InMemoryMemory()
        self.hooks = hooks or []

        self._settings = settings
        self._provider: BaseProvider | None = provider if isinstance(provider, BaseProvider) else None
        self._tools: dict[str, ToolDefinition] = {}

        for func in tools or []:
            self.add_tool(func)

    def add_tool(self, func: Callable[..., Any]) -> None:
        """Register a callable tool on the agent."""
        definition = to_tool_definition(func)
        self._tools[definition.name] = definition

    async def reset_memory(self) -> None:
        """Clear conversation state but preserve the agent system prompt."""
        await asyncio.to_thread(self.memory.reset)

    async def _conversation_messages(self, extra_messages: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        backend_msgs = await asyncio.to_thread(lambda: self.memory.messages)
        messages.extend(backend_msgs)
        if extra_messages:
            messages.extend(extra_messages)
        return messages

    async def _dispatch_hook(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        if not self.hooks:
            return
        tasks = []
        active_hooks = []
        for hook in self.hooks:
            method = getattr(hook, event_name, None)
            if method:
                tasks.append(method(*args, **kwargs))
                active_hooks.append(hook)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for hook, result in zip(active_hooks, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "[AgentAPI] Hook %s raised: %s",
                        type(hook).__name__,
                        result,
                    )

    async def run(
        self, 
        message: str, 
        *, 
        max_tool_rounds: int = 3,
        output_schema: type[BaseModel] | None = None,
        max_validation_retries: int = 1,
    ) -> str | BaseModel:
        """Execute a chat completion with optional tool-calling loop and Pydantic schema validation."""
        run_id = str(uuid.uuid4())
        
        try:
            await self._dispatch_hook("on_agent_start", run_id=run_id, message=message)

            system_instructions = self.system_prompt
            if output_schema:
                schema_json = output_schema.model_json_schema()
                system_instructions += f"\n\nYou must output a valid JSON object matching this schema:\n{json.dumps(schema_json, indent=2)}"

            conversation_messages = await self._conversation_messages([{"role": "user", "content": message}])
            if output_schema:
                conversation_messages[0]["content"] = system_instructions
            
            provider = self._get_provider()

            validation_attempts = 0
            while validation_attempts <= max_validation_retries:
                for _ in range(max_tool_rounds + 1):
                    # Narrow the provider error wrapper to only the actual provider call.
                    try:
                        await self._dispatch_hook("on_llm_start", run_id=run_id, messages=conversation_messages)
                        response = await provider.chat(
                            conversation_messages,
                            tools=self._tool_schemas(),
                            tool_calling=self.tool_calling,
                        )
                        await self._dispatch_hook("on_llm_end", run_id=run_id, response=response)
                    except AgentAPIProviderError:
                        raise
                    except Exception as exc:
                        logger.exception(
                            "[AgentAPI] Provider error in run(). Check your API key and provider configuration."
                        )
                        raise AgentAPIProviderError(
                            f"Provider call failed: {exc}. Check your API key and provider configuration.",
                            original=exc,
                        ) from exc

                    if response.tool_calls:
                        conversation_messages.append(
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
                        await self._execute_tool_calls(response.tool_calls, conversation_messages, run_id=run_id)
                        continue

                    if output_schema:
                        try:
                            parsed_response = output_schema.model_validate_json(response.content)
                            await asyncio.to_thread(self.memory.add, {"role": "user", "content": message})
                            await asyncio.to_thread(self.memory.add, {"role": "assistant", "content": response.content})
                            await self._dispatch_hook("on_agent_end", run_id=run_id, final_response=parsed_response)
                            return parsed_response
                        except ValidationError as e:
                            validation_attempts += 1
                            if validation_attempts > max_validation_retries:
                                raise AgentSchemaValidationError(
                                    f"Failed to produce valid JSON matching schema after {max_validation_retries} retries. Error: {e}"
                                ) from e
                            conversation_messages.append({"role": "assistant", "content": response.content})
                            conversation_messages.append({
                                "role": "user", 
                                "content": f"Your response failed schema validation: {e}. Please fix the JSON output."
                            })
                            break # Break the tool loop, retry the outer validation loop

                    await asyncio.to_thread(self.memory.add, {"role": "user", "content": message})
                    await asyncio.to_thread(self.memory.add, {"role": "assistant", "content": response.content})
                    await self._dispatch_hook("on_agent_end", run_id=run_id, final_response=response.content)
                    return response.content

                if response.tool_calls or not output_schema:
                    break

            fallback = "Tool loop reached max rounds without final response"
            await asyncio.to_thread(self.memory.add, {"role": "user", "content": message})
            await asyncio.to_thread(self.memory.add, {"role": "assistant", "content": fallback})
            await self._dispatch_hook("on_agent_end", run_id=run_id, final_response=fallback)
            return fallback

        except Exception as exc:
            await self._dispatch_hook("on_agent_error", run_id=run_id, error=exc)
            raise

    def stream(self, message: str) -> StreamingResponse:
        """Stream model tokens as FastAPI StreamingResponse using SSE format.

        Each token is split on newline boundaries and emitted as valid SSE
        frames so multi-line payloads remain a single logical SSE event.
        Provider errors are logged server-side and surfaced as a sanitized
        ``data: [ERROR] ...\\n\\n`` event so the client never receives raw
        exception text.

        Examples::

            # FastAPI endpoint
            @app.post("/chat/stream")
            async def chat_stream(message: str):
                return agent.stream(message)

            # Direct async iteration
            async for chunk in agent.stream(message):
                print(chunk, end="", flush=True)
        """
        async def _sse_generator() -> AsyncIterator[str]:
            try:
                async for token in self._stream_generator(message):
                    for line in str(token).replace("\r", "").split("\n"):
                        yield f"data: {line}\n"
                    yield "\n"
            except AgentAPIProviderError:
                logger.exception("[AgentAPI] Streaming error surfaced in SSE generator.")
                yield "data: [ERROR] Streaming failed. Check server logs for details.\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    async def _stream_generator(self, message: str) -> AsyncIterator[str]:
        """Internal async generator that streams tokens from the provider."""

        conversation_messages = await self._conversation_messages([{"role": "user", "content": message}])
        provider = self._get_provider()

        collected: list[str] = []

        try:
            async for token in provider.stream(
                conversation_messages,
                tools=self._tool_schemas(),
                tool_calling=self.tool_calling,
            ):
                collected.append(token)
                yield token
        except AgentAPIProviderError:
            raise
        except Exception as exc:
            logger.exception(
                "[AgentAPI] Streaming error. Check your provider configuration and API key."
            )
            raise AgentAPIProviderError(
                f"Streaming failed: {exc}",
                original=exc,
            ) from exc

        full_text = "".join(collected)
        await asyncio.to_thread(self.memory.add, {"role": "user", "content": message})
        await asyncio.to_thread(self.memory.add, {"role": "assistant", "content": full_text})

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

    async def _execute_tool_calls(self, calls: list[ToolCall], conversation_messages: list[dict[str, Any]], run_id: str | None = None) -> None:
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
                if run_id:
                    await self._dispatch_hook("on_tool_start", run_id=run_id, tool_call=call)
                args = parse_tool_args(call.arguments)
                result = tool_def.func(**args)
                if inspect.isawaitable(result):
                    result = await result
                output = str(result)
                if run_id:
                    await self._dispatch_hook("on_tool_end", run_id=run_id, tool_call=call, result=output)
            except Exception as exc:  # noqa: BLE001
                output = f"Tool execution failed: {exc}"
                if run_id:
                    await self._dispatch_hook("on_tool_end", run_id=run_id, tool_call=call, result=output)

            conversation_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": output,
                }
            )

