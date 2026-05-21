# API Reference

## Exports

AgentAPI public interface:

```python
from agentapi import Agent, AgentAPI, tool, AgentConfigurationError, BaseProvider
```

## `Agent`

Constructor:

```python
Agent(
    *,
    system_prompt: str,
    provider: str | BaseProvider | None = None,
    model: str | None = None,
    tools: list[Callable[..., Any]] | None = None,
    tool_calling: dict[str, Any] | None = None,
    event_handler: Callable[[dict[str, Any]], Any] | None = None,
)
```

Methods:

- `await run(message: str, *, max_tool_rounds: int = 3) -> str`
- `stream(message: str) -> AsyncIterator[str]`
- `add_tool(func)`
- `reset_memory()`
- `register_provider(name, factory)` (class method)

Notes:

- `run` is the best default for tool-centric workflows.
- `stream` is for incremental output over SSE.
- `event_handler` is optional and receives structured metadata-only lifecycle events.

### Structured events

Pass an `event_handler` when you want to connect AgentAPI to your own logger, metrics client, or tracing adapter:

```python
import logging
from agentapi import Agent

logger = logging.getLogger("agentapi.events")


def log_agent_event(event: dict) -> None:
    logger.info("agent event", extra={"agentapi_event": event})


agent = Agent(
    system_prompt="You are a helpful assistant",
    provider="openai",
    event_handler=log_agent_event,
)
```

The callback can be synchronous or async. AgentAPI emits events such as:

- `provider_call_start`
- `provider_call_end`
- `tool_execution_start`
- `tool_execution_end`
- `error`

Streaming metrics use the same `provider_call_start` and `provider_call_end` event names with `mode="stream"`. The final streaming `provider_call_end` event includes stream-specific metadata such as `token_count` and `content_length`.

Events include safe metadata like provider name, model name, tool name, duration, token count, and output length. They intentionally do not include prompts, tool arguments, provider payloads, API keys, or raw response content.

## `AgentAPI`

Subclass of FastAPI with chat decorators and branded docs pages.

Methods:

- `chat(path: str, **kwargs)`
- `stream(path: str, **kwargs)`

Behavior:

- `@app.chat` returns JSON unless handler returns async iterator.
- Async iterator return values are automatically converted to SSE.

## `tool`

Decorator for registering LLM-callable functions.

```python
@tool
def my_tool(arg: str) -> str:
    """Description used by model."""
    ...
```

Decorator supports metadata overrides:

```python
@tool(
    name="my_tool",
    description="Short action-oriented tool description",
    context="Extra model-facing guidance for when to use this tool",
)
def my_tool(arg: str) -> str:
    ...
```

## Errors

### `AgentConfigurationError`

Raised when required runtime configuration (such as API keys) is missing.

### `AgentProviderError`

Raised when upstream provider calls fail. Includes status-code context.
