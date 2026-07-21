# Object model reference

This is a contributor map of framework objects, methods, and owned attributes.
It is intentionally more detailed than the user-facing API reference. Leading
underscores identify implementation details, not extension promises.

## Public import surface

`agentapi/__init__.py` exports:

- Core: `Agent`, `AgentAPI`, `tool`, `BaseProvider`
- Errors: `AgentConfigurationError`, `AgentProviderError`
- Memory contracts: `MemoryBackend`, `MemoryScope`, `MemorySession`, `MemoryStore`
- Memory implementations: `InMemoryMemory`, `InMemoryStore`, `RedisMemory`,
  `RedisStore`, `MongoDBMemory`, `MongoDBStore`, `create_conversation_id`
- Convenience re-exports from FastAPI, FastAPI responses, and Pydantic

Types described below but absent from `__all__` are internal or require an
implementation-module import.

## `AgentAPI`

Defined in `agentapi/core/app.py`; subclasses `fastapi.FastAPI`.

### Constructor-owned attributes

| Attribute | Meaning |
| --- | --- |
| `_sse_chunk_size` | Maximum characters emitted per adapted SSE data frame |
| `_sse_heartbeat_seconds` | Quiet interval for keepalive comments, or `None` |
| `_agentapi_docs_url` / `_agentapi_redoc_url` | Branded documentation route paths |
| `_agentapi_swagger_ui_oauth2_redirect_url` | Swagger OAuth redirect path |
| `_agentapi_swagger_ui_init_oauth` | Swagger OAuth initialization configuration |
| `_agentapi_swagger_ui_parameters` | Swagger UI runtime parameters |
| `_agentapi_logo_file` / `_agentapi_favicon_file` | Packaged asset filesystem paths |
| `_agentapi_logo_path` / `_agentapi_favicon_path` | Public asset route paths |
| `openapi` | Rebound to `_custom_openapi` |

Inherited FastAPI attributes such as `routes`, `state`, `openapi_schema`,
`title`, `version`, and `description` retain their normal FastAPI semantics.

### Methods

| Method | Role |
| --- | --- |
| `chat(path, **kwargs)` | Register a POST handler that may return JSON, a Response, or an async iterator |
| `stream(path, **kwargs)` | Register a POST handler that must return an async iterator |
| `_invoke_handler(func, *args, **kwargs)` | Call sync or async application handlers uniformly |
| `_to_sse_response(source)` | Adapt an async string iterator into framed SSE with optional heartbeat |
| `_iter_token_chunks(token, chunk_size=None)` | Split a text fragment into bounded async chunks |
| `_custom_openapi()` | Build/cache OpenAPI and inject the AgentAPI logo extension |
| `_logo()` / `_favicon()` | Serve packaged image assets |
| `_swagger_ui_html()` / `_redoc_html()` | Generate branded documentation HTML |
| `_swagger_ui_redirect()` | Serve the Swagger OAuth redirect page |

## `Agent`

Defined in `agentapi/agent/agent.py`. It is stateful and binds one provider
configuration, one tool registry, and one memory view.

### Attributes

| Attribute | Visibility | Meaning |
| --- | --- | --- |
| `_custom_provider_factories` | class-private | Process-wide mapping of normalized names to provider factories |
| `system_prompt` | public | Prompt prepended to every provider request |
| `provider_name` | public | Normalized configured name or injected provider class name |
| `model` | public | Explicit model or provider-family default |
| `tool_calling` | public | Provider request controls after defaults and overrides are merged |
| `memory` | public | Bound `MemoryBackend` used for history |
| `_settings` | private | Immutable environment snapshot captured at construction |
| `_provider` | private | Injected or lazily created/cached `BaseProvider` |
| `_tools` | private | Tool definitions keyed by model-facing name |

### Methods

| Method | Role |
| --- | --- |
| `run(message, max_tool_rounds=3)` | Execute provider calls and sequential tool rounds; persist the final turn |
| `stream(message)` | Return an SSE `StreamingResponse` over provider text tokens |
| `add_tool(func)` | Convert and insert/replace a registered tool |
| `reset_memory()` | Clear only the bound memory scope |
| `register_provider(name, factory)` | Add or replace a class-wide custom provider factory |
| `_conversation_messages(extra_messages=None)` | Build system + stored + extra canonical messages |
| `_stream_generator(message)` | Stream raw provider tokens and persist concatenated output |
| `_create_provider(settings)` | Resolve a custom factory or construct a built-in adapter |
| `_get_provider()` | Lazily create and cache the provider |
| `_require_api_key(value, env_name)` | Validate a key at provider-creation time |
| `_default_model_for(provider_name)` | Select built-in model defaults |
| `_default_tool_calling_for(provider_name)` | Select Gemini or OpenAI-style tool defaults |
| `_tool_schemas()` | Return registered schemas or `None` |
| `_execute_tool_calls(calls, messages)` | Invoke tools and append canonical tool results |

`ProviderFactory` is the callable contract
`(agent: Agent, settings: Any, model: str) -> BaseProvider`.

`AgentAPIUsageError` and `AgentAPIProviderError` are module-local orchestration
exceptions. The latter stores an `original` exception attribute. They are not
part of the package public exports.

## Tool objects and functions

Defined in `agentapi/agent/tools.py`.

### `ToolDefinition`

A mutable dataclass with:

| Field | Meaning |
| --- | --- |
| `name` | Model-facing lookup name |
| `description` | Model-facing behavior description |
| `context` | Additional model-facing usage guidance |
| `func` | Original executable Python callable |
| `schema` | OpenAI-style strict function schema |

### Module functions

| Function | Role |
| --- | --- |
| `tool(...)` | Decorate a callable with AgentAPI metadata; supports bare and configured forms |
| `to_tool_definition(func)` | Reuse attached metadata/schema or derive them on registration |
| `parse_tool_args(args_json)` | Parse the provider JSON argument string into a dictionary |
| `_json_type(annotation)` | Map supported Python/union annotations to JSON Schema types |
| `_compose_tool_description(...)` | Combine docstring/description with optional context |
| `_build_openai_tool_schema(...)` | Generate the canonical strict function schema |

## Provider protocol and models

Defined in `agentapi/providers/base.py`.

### `ToolCall`

Dataclass fields: `id: str`, `name: str`, and `arguments: str`. Arguments remain
JSON text until orchestration parses them.

### `ProviderResponse`

Dataclass fields: `content: str`, `tool_calls: list[ToolCall]`, and
`raw_message: dict[str, Any]`.

### `BaseProvider`

Abstract methods:

- `chat(messages, *, tools=None, tool_calling=None) -> ProviderResponse`
- `stream(messages, *, tools=None, tool_calling=None) -> AsyncIterator[str]`

The annotations and keyword-only optional arguments are part of the adapter
contract. New providers should return empty content/tool collections rather
than vendor-specific `None` values.

## Provider implementations

| Class | Owned attributes | Important methods/behavior |
| --- | --- | --- |
| `OpenAICompatibleProvider` | `api_key`, `model`, normalized `base_url`, `extra_headers` | `_headers`, `_extract_message`, `chat`, `stream`, `_safe_error_detail`; shared `/chat/completions` HTTP adapter |
| `OpenAIProvider` | inherited | Configures `https://api.openai.com/v1` |
| `OpenRouterProvider` | inherited | Configures OpenRouter URL plus `HTTP-Referer` and `X-Title` headers |
| `HuggingFaceProvider` | inherited | Configures supplied base URL or Hugging Face's OpenAI-compatible endpoint |
| `GeminiProvider` | `api_key`, `model`, `base_url` | Builds Gemini contents/system/tool payloads, maps function calls/results, extracts text, and handles cumulative SSE chunks |
| `AnthropicProvider` | `client`, `model` | Translates system/messages/tools and SDK content blocks; uses the Anthropic async SDK |

Gemini helpers are `_map_http_status_error`, `_safe_error_detail`,
`_safe_error_detail_sync`, `_build_payload`, `_extract_tool_calls`,
`_to_function_declarations`, `_to_function_response_payload`, and `_extract_text`.

Anthropic helpers are `_format_messages` and `_format_tools`. Its current stream
path translates text only and does not include formatted tools in the streaming
request.

### Current Anthropic integration boundary

`Agent._create_provider()` recognizes the explicit name `anthropic` and imports
the adapter lazily. However, `Provider` settings validation, CLI choices,
`providers/__init__.py`, environment templates, and declared project dependencies
do not currently expose a complete built-in Anthropic installation path. Treat
it as a partial integration until those surfaces and tests are aligned.

## Memory identity and contracts

Defined in `agentapi/agent/memory.py`.

### `MemoryScope`

A frozen dataclass with optional constructor fields that are normalized after
construction:

| Field/property | Meaning |
| --- | --- |
| `conversation_id` | Canonical UUID string; generated when omitted |
| `user_id` | Optional user isolation dimension |
| `tenant_id` | Optional tenant isolation dimension |
| `agent_id` | Optional agent/persona isolation dimension |
| `source_id` | Optional channel or memory-source dimension |
| `key` | Raw UUID when unscoped; otherwise deterministic SHA-256 scope key |
| `metadata` | Persistable plain identifiers plus `scope_key` |

`matches()` compares only filters that are not `None`.

### `MemoryBackend`

The smallest contract consumed by `Agent`:

- read-only property `messages -> list[dict[str, Any]]`
- `add(message) -> None`
- `reset() -> None`

### `MemorySession`

Attributes: private `_store: MemoryStore` and public `scope: MemoryScope`.
Properties expose `scope_key`, `conversation_id`, `user_id`, `tenant_id`,
`agent_id`, `source_id`, and delegated `messages`. `add()` delegates to
`store.append()` and `reset()` delegates to `store.clear()`.

### `MemoryStore`

`session(scope=None, **scope_fields)` creates a bound `MemorySession` and rejects
passing both representations. Subclasses must implement `read(scope)`,
`append(scope, message)`, and `clear(scope)`. `list_conversations(...)` and
`purge(...)` are optional administrative capabilities whose base implementations
raise `NotImplementedError`.

## Memory stores

| Class | Owned state | Storage behavior | Administrative support |
| --- | --- | --- | --- |
| `InMemoryStore` | `_conversations`, `_scopes`, `_lock` | Per-scope lists guarded by a thread lock | `list_conversations`, `purge` |
| `RedisStore` | `_redis`, `_ttl_seconds` | `conv:{scope.key}:messages` list and `:meta` hash with TTL | Inherits unsupported base methods; has `close()` |
| `MongoDBStore` | `_collection`, `_client`, `_owns_client`, `_ttl_seconds` | One document per scope; atomic `$push`, metadata, timestamps, optional TTL | `list_conversations`, `purge`, ownership-aware `close()` |

Redis private helpers `_messages_key()`, `_meta_key()`, and `_ensure_meta()` own
key construction and metadata initialization. MongoDB helpers `_query()`,
`_ensure_indexes()`, `_base_metadata()`, `_expires_at()`, and `_filter()` own
persistence policy.

## Compatibility memory wrappers

`InMemoryMemory`, `RedisMemory`, and `MongoDBMemory` subclass `MemorySession`.
Each constructor creates a store, creates a `MemoryScope`, and binds them. They
preserve the simpler single-scope API but do not share a client/store with other
wrapper instances.

- `InMemoryMemory` accepts an optional conversation ID and all optional scope fields.
- `RedisMemory` requires `redis_url` and `conversation_id`; it exposes legacy
  `_messages_key`/`_meta_key` properties and `close()`.
- `MongoDBMemory` requires `conversation_id`, accepts URL/client/collection
  injection and all scope/TTL options, and provides `close()`.

`create_conversation_id()` returns a UUIDv4 string.

## Configuration objects

Defined in `agentapi/config/settings.py`.

`Provider` is a string enum containing `OPENAI`, `GEMINI`, `OPENROUTER`, and
`HUGGINGFACE`. `Settings` is a frozen dataclass containing the corresponding API
keys, `huggingface_base_url`, and `default_provider`. `__post_init__()` normalizes
and validates the default against the enum.

`get_settings()` reads the environment on every call. `.env` loading occurs when
the module is imported. `_validate_api_key()` trims configured keys and rejects
empty or very short values; missing values remain `None` until a selected
provider requires one.

## Shared errors

Defined in `agentapi/errors.py`:

- `AgentConfigurationError(ValueError)` represents missing/invalid runtime configuration.
- `AgentProviderError(RuntimeError)` represents upstream request/response failures.

The shared provider error currently stores only the exception message.

## CLI functions and templates

`agentapi/cli.py` owns four scaffold strings: `MAIN_TEMPLATE`, `TOOLS_TEMPLATE`,
`AGENTS_TEMPLATE`, and `ENV_TEMPLATE`.

| Function | Responsibility |
| --- | --- |
| `_write_file` | UTF-8 template write |
| `_prompt_with_default` | Interactive input with fallback |
| `_collect_new_project_config` | Resolve and validate project/provider selection |
| `cmd_new` | Refuse an existing path, create a project, and write four files |
| `cmd_run` | Spawn Uvicorn with host/port/reload/worker arguments |
| `build_parser` | Define the `new` and `run` argparse trees |
| `main` | Parse arguments and dispatch to the chosen command |
