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

## Architecture Reference

For internal and extension APIs, see the complete
[object model](architecture/object-model.md). For control flow and error
boundaries, see [runtime and data flows](architecture/runtime.md).
