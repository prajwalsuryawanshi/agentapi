# Architecture overview

This guide documents AgentAPI as implemented in this repository. It is aimed at
contributors who need both a mental model and enough code-level detail to make a
safe change.

Start with the root
[`ARCHITECTURE.md`](https://github.com/prajwalsuryawanshi/agentapi/blob/main/ARCHITECTURE.md)
for the one-page summary. Continue with:

- [Runtime and data flows](runtime.md)
- [Object model reference](object-model.md)
- [Feature and deployment scalability](scalability.md)
- [Contributor change guide](contributing.md)

## Architectural goals

AgentAPI optimizes for a small public surface and replaceable infrastructure:

1. Application code talks to `Agent` rather than a vendor SDK.
2. Providers translate vendor payloads at the package edge.
3. Agents depend on a three-operation memory contract, not a database client.
4. A scoped session separates per-conversation identity from a shared store.
5. Tools remain ordinary Python callables with attached metadata.
6. FastAPI remains available underneath `AgentAPI`, so the framework does not
   need to duplicate routing, dependency injection, or ASGI lifecycle features.

## Repository map

```text
agentapi/
  __init__.py                 Public imports and FastAPI convenience re-exports
  __main__.py                 `python -m agentapi` entry point
  cli.py                      `agentapi new` and `agentapi run`
  errors.py                   Shared configuration/provider exceptions
  core/
    app.py                    AgentAPI FastAPI subclass and SSE route adapters
  agent/
    agent.py                  Provider-independent orchestration and tool loop
    tools.py                  Callable metadata, JSON schema, argument parsing
    memory.py                 Scope/session/store memory model and backends
  config/
    settings.py               Environment loading and provider defaults
  providers/
    base.py                   Provider protocol and normalized response models
    openai_compatible.py      Shared Chat Completions implementation
    openai.py                 OpenAI endpoint configuration
    openrouter.py             OpenRouter endpoint/header configuration
    huggingface.py            Hugging Face endpoint configuration
    gemini.py                 Gemini payload and response translation
    anthropic.py              Anthropic SDK translation
examples/
  example/                    Minimal app/agent/tool split
  kafka/                      Out-of-process multi-agent scaling patterns
tests/
  test_scoped_memory_backends.py  Current scoped-memory regression coverage
docs/
  architecture/              This contributor-focused architecture guide
```

## Layer boundaries

### 1. Transport layer

`agentapi.core.app.AgentAPI` subclasses `fastapi.FastAPI`. It registers branded
documentation routes and provides two POST decorators:

- `chat()` accepts ordinary return values, existing `Response` objects, or async
  iterators. Async iterators are converted to SSE.
- `stream()` requires an async iterator and converts it to SSE.

Because this is a FastAPI subclass, authentication, dependency injection,
middleware, lifespan hooks, request models, and ordinary routes remain FastAPI
concerns.

### 2. Orchestration layer

`agentapi.agent.agent.Agent` builds the canonical message list, lazily resolves a
provider, runs non-stream tool rounds, and records completed user/assistant
turns. It knows the normalized provider protocol but does not know HTTP request
objects or database clients.

### 3. Integration layer

Provider adapters implement `BaseProvider.chat()` and `BaseProvider.stream()`.
They convert the canonical message/tool representation into vendor payloads and
return `ProviderResponse`/`ToolCall` objects or text tokens.

Memory stores implement `MemoryStore.read()`, `append()`, and `clear()` against a
`MemoryScope`. A `MemorySession` adapts those store operations to the smaller
`MemoryBackend` contract consumed by `Agent`.

### 4. Application layer

Route handlers, tool implementations, agent factories, authorization, tenant
resolution, queues, and business workflows belong to the consuming application.
The Kafka examples intentionally live under `examples/`, not `agentapi/`, because
message-bus policy is an application architecture choice.

## Canonical internal data shapes

The current implementation uses dictionaries at integration boundaries rather
than a formal message class.

| Shape | Required/important fields | Producer and consumer |
| --- | --- | --- |
| Message | `role`, `content`; optionally `tool_calls`, `tool_call_id`, `name` | `Agent`, memory backends, providers |
| OpenAI-style tool schema | `type=function`, nested `function.name`, `description`, `parameters` | `tools.py`, provider adapters |
| `ToolCall` | `id`, `name`, JSON string `arguments` | Providers -> `Agent` |
| `ProviderResponse` | `content`, `tool_calls`, `raw_message` | Providers -> `Agent.run()` |
| Scope metadata | `conversation_id`, `scope_key`, optional actor/source fields | `MemoryScope` -> stores |

Treat these shapes as compatibility boundaries. If one changes, update all
producers, all consumers, provider tests, memory tests, examples, and the object
model reference together.

## Public API boundary

`agentapi/__init__.py` defines the import experience. It exports the core AgentAPI
types, memory abstractions/backends, error classes, and several FastAPI/Pydantic
conveniences. A contributor adding a public type must decide deliberately whether
it belongs in `__all__`; an implementation module alone does not make a stable
public API.

## Architectural invariants

Contributions should preserve these invariants unless the change explicitly
redesigns and documents the affected contract:

- The system prompt is prepended at runtime and is not persisted as conversation history.
- Provider-facing messages use one normalized role/content representation.
- `Agent` can accept either a provider name or an already-created `BaseProvider`.
- Provider creation is lazy; constructing an `Agent` does not make a network request.
- A `MemorySession` can only access the `MemoryScope` to which it is bound.
- Unscoped storage keys remain the raw conversation UUID for backward compatibility.
- Scoped keys are deterministic and opaque.
- `InMemoryStore` returns a new list container, while nested message dictionaries
  remain shared and must be treated as read-only.
- Non-stream `Agent.run()` performs tool rounds; the streaming path currently streams text only.
- Transport concerns stay in `core/app.py`; provider payload details stay in provider modules.

## Where to go next

The [runtime guide](runtime.md) traces concrete call paths. The [object model](object-model.md)
lists every framework class and its state. The [scalability guide](scalability.md)
explains safe reuse, extension seams, and current limits. The [contributor guide](contributing.md)
turns those boundaries into change checklists.
