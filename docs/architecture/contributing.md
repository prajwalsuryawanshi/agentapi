# Contributor change guide

Use this page to translate an issue into the smallest correct architectural
change. Read the [runtime flows](runtime.md) and relevant objects in the
[object model](object-model.md) before editing framework code.

## Find the owning boundary

| If the change is about... | Start here | Also inspect |
| --- | --- | --- |
| Routes, OpenAPI, JSON/SSE framing, heartbeat | `agentapi/core/app.py` | `docs/streaming.md`, handler tests |
| Agent turns, tool rounds, provider choice | `agentapi/agent/agent.py` | provider base models, memory timing, errors |
| Tool decorator, type mapping, arguments | `agentapi/agent/tools.py` | every provider tool translator |
| Conversation identity or storage | `agentapi/agent/memory.py` | exports, optional dependencies, memory tests |
| Provider-neutral contract | `agentapi/providers/base.py` | all provider modules and Agent callers |
| OpenAI-compatible HTTP behavior | `agentapi/providers/openai_compatible.py` | OpenAI, OpenRouter, Hugging Face subclasses |
| Gemini translation | `agentapi/providers/gemini.py` | canonical messages/tool schema |
| Anthropic translation | `agentapi/providers/anthropic.py` | dependency/settings/CLI integration status |
| Environment values/defaults | `agentapi/config/settings.py` | `.env.example`, CLI templates, docs |
| Generated projects or Uvicorn launch | `agentapi/cli.py` | getting-started docs and smoke tests |
| Public imports | `agentapi/__init__.py` | API reference and compatibility |
| Cross-agent workflow | consuming app or `examples/` | memory scopes, event contracts, durability |

Keep provider SDK payloads out of `Agent`, persistence details out of route
handlers, and application workflow policy out of the core package.

## Before editing

1. Check `git status --short` and preserve unrelated working-tree changes.
2. Identify the public or internal contract being changed.
3. Trace every producer and consumer of the affected data shape.
4. Decide whether backward compatibility is required for imports, constructor
   parameters, stored data, generated projects, or HTTP behavior.
5. Add tests at the narrowest boundary plus one integration path when behavior
   crosses layers.

## Adding a provider

### Prefer registration for external integrations

An application or extension package can implement `BaseProvider` and register a
factory without changing AgentAPI core:

```python
from agentapi import Agent, BaseProvider
from agentapi.providers.base import ProviderResponse


class ExampleProvider(BaseProvider):
    def __init__(self, *, model: str) -> None:
        self.model = model

    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(
            content="example response",
            tool_calls=[],
            raw_message={},
        )

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield "example response"


Agent.register_provider(
    "vendor.example",
    lambda agent, settings, model: ExampleProvider(model=model),
)
```

Test normalization to `ProviderResponse`, tool call IDs/names/JSON arguments,
empty content, malformed responses, HTTP failures, streaming termination, and
cancellation.

### Built-in provider checklist

A fully built-in provider usually touches all of these surfaces:

- adapter module implementing `BaseProvider`;
- `Agent._create_provider()` and appropriate lazy import;
- default model and tool-calling settings;
- `Provider` enum, `Settings` fields, validation, and `get_settings()`;
- `.env.example` and `agentapi.cli.ENV_TEMPLATE`;
- CLI provider choices if scaffolding supports it;
- `providers/__init__.py` and possibly top-level `__all__`;
- required or optional dependency declaration in `pyproject.toml`;
- provider docs, API docs, architecture object model, and examples; and
- mocked non-stream, stream, tool, configuration, and error tests.

If the vendor is OpenAI Chat Completions compatible, subclass
`OpenAICompatibleProvider` and configure only the differences. Change the shared
base only when the behavior is valid for every subclass.

## Adding a memory backend

Implement `MemoryStore` when one long-lived backend should support many scoped
conversations. The required operations are synchronous today:

```python
class ExampleStore(MemoryStore):
    def read(self, scope: MemoryScope) -> list[dict[str, Any]]:
        ...

    def append(self, scope: MemoryScope, message: dict[str, Any]) -> None:
        ...

    def clear(self, scope: MemoryScope) -> None:
        ...
```

Preserve these behaviors:

- every operation uses `scope.key` as the isolation identity;
- human-readable scope metadata is retained when enumeration is supported;
- a read of an unknown scope returns an empty list;
- list-level mutation of returned collections cannot alter stored list structure;
  nested message dictionaries are currently shared by `InMemoryStore` and should
  be treated as read-only;
- clear affects only the exact scope;
- connection ownership and `close()` behavior are explicit; and
- optional administrative methods either work with filters or clearly raise
  `NotImplementedError`.

Test the same conversation UUID across different tenant, user, agent, and source
values; same-scope sharing; reset isolation; malformed persisted data; TTL
behavior; injected fake clients; and resource cleanup.

If the dependency is optional, add a named extra, import it lazily, and provide an
actionable installation error. Export public backends through `agentapi/__init__.py`
and update `docs/memory.md` plus the architecture capability matrix.

## Changing memory scope or persisted shape

Scope-key changes are data migrations. The unscoped raw-UUID key is a documented
backward-compatibility rule, while scoped keys depend on stable field names,
sorted compact JSON, and SHA-256. Changing any of those makes existing records
unreachable.

For a new scope dimension:

1. Add and normalize the `MemoryScope` field.
2. Include it in key construction, metadata, and `matches()`.
3. Expose it from `MemorySession`.
4. Thread it through all compatibility wrapper constructors.
5. Add store indexes/filter support where appropriate.
6. Test collision isolation and administrative filtering for every backend.
7. Document storage compatibility and migration behavior.

## Changing tools or the tool loop

Schema changes must be checked against each provider adapter. OpenAI-style
strict schema is canonical internally, but Gemini and Anthropic translate it.

For annotation/schema work, cover bare annotations, unannotated parameters,
containers, nullable unions, defaults, required lists, nested schemas, and
unsupported types. For orchestration work, cover sync/async tools, invalid JSON,
unknown names, exceptions, maximum rounds, multiple calls, and memory contents.

Parallel execution is not a local loop optimization: it changes side-effect
ordering, result ordering, cancellation, and failure semantics. Specify those
semantics before implementing it.

## Changing streaming or decorators

Test both streaming levels described in [runtime and data flows](runtime.md):

- a handler returning `Agent.stream()` (an existing `StreamingResponse`); and
- a handler returning an async iterator for `AgentAPI` to adapt.

Verify exact SSE framing, multi-line content, chunk boundaries, `[DONE]`, error
events, heartbeat timing, response headers, cancellation, and empty streams.
Also verify that sync and async handler signatures remain visible to FastAPI and
that normal response objects pass through untouched.

Do not make `@app.stream` silently accept non-iterators; its stricter behavior is
the reason it remains separate from `@app.chat`.

## Changing errors

Map the full path from provider/tool/configuration source to HTTP or SSE output.
The current code has shared and orchestration-local provider error classes; see
the [error-boundary section](runtime.md#error-boundaries). Consolidating them is
a compatibility-sensitive change because callers may import or catch the shared
class and tests must cover exceptions raised before and during response iteration.

Never put API keys, authorization headers, unbounded provider bodies, or sensitive
tool arguments in client-visible errors or logs.

## Changing the public API

A new public symbol requires:

1. a stable name and documented import path;
2. export from `agentapi/__init__.py` and inclusion in `__all__` when appropriate;
3. docstrings and user-facing API documentation;
4. compatibility tests importing it from `agentapi`; and
5. architecture object-model updates when it adds or changes a boundary.

Avoid re-exporting dependencies merely to shorten imports unless that convenience
is an intentional long-term compatibility promise.

## Tests and local verification

The package supports Python 3.10+. The current CI workflow installs the package
and runs compilation on Python 3.11. The repository also contains scoped-memory
pytest coverage, although CI does not currently invoke it.

Baseline commands:

```bash
python -m pip install -e .
python -m compileall agentapi
python -m pytest
```

Install relevant optional extras before backend tests:

```bash
python -m pip install -e ".[redis,mongodb]"
```

Provider tests should mock HTTP/SDK boundaries and must not require live API keys.
CLI tests should use a temporary directory and patch subprocess execution.
Streaming tests should consume the response iterator rather than only asserting
the response class.

When a change introduces a new test dependency or command, update CI in the same
pull request so repository checks and contributor instructions remain aligned.

## Documentation update map

| Changed behavior | Update at minimum |
| --- | --- |
| Public import/signature | `docs/api-reference.md`, relevant guide, object model |
| Provider support/config | `README.md`, `.env.example`, `docs/providers.md`, object model |
| Memory scope/backend | `docs/memory.md`, scalability matrix, object model |
| Tool schema/execution | `docs/tools.md`, runtime flow, object model |
| SSE/decorator behavior | `docs/streaming.md`, runtime flow |
| CLI scaffold/arguments | `docs/cli.md`, getting started, templates |
| Deployment topology | `docs/deployment.md`, scalability guide, examples |
| Package/module ownership | root `ARCHITECTURE.md` and architecture overview |

## Pull-request architecture checklist

- The change lives in the layer that owns the behavior.
- Provider, memory, tool, transport, and application concerns remain separated.
- Normalized message/tool/response shapes stay compatible or include migration notes.
- State ownership and safe sharing are explicit.
- Multi-user scope values come from trusted application identity.
- Sync/async behavior and cancellation are tested where applicable.
- Errors are normalized and do not disclose secrets.
- New dependencies are justified, declared, and lazy when optional.
- Public exports, CLI templates, examples, and docs agree.
- Focused tests and the baseline verification commands pass.
