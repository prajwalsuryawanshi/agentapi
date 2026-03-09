# API Reference

## Exports

AgentAPI public interface:

```python
from agentapi import Agent, AgentApp, tool, AgentConfigurationError, BaseProvider
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

## `AgentApp`

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

## Errors

### `AgentConfigurationError`

Raised when required runtime configuration (such as API keys) is missing.

### `AgentProviderError`

Raised when upstream provider calls fail. Includes status-code context.
