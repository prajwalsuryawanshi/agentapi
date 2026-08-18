# Feature and deployment scalability

AgentAPI scales in two different senses:

- **Feature scalability** is the ability to add providers, storage engines,
  tools, and transport behavior without rewriting the agent loop.
- **Runtime scalability** is the ability to serve more users, workers, and
  agent workflows while keeping state isolated and external services healthy.

The current boundaries support both, but they do not make every object safe to
share automatically.

## Feature scalability by boundary

| Feature | Stable seam | Core code that should remain unchanged | Cross-cutting surfaces to check |
| --- | --- | --- | --- |
| Third-party model provider | `BaseProvider` + `Agent.register_provider()` | Built-in selection and other adapters | Provider tests and user docs |
| Built-in model provider | `BaseProvider` + `_create_provider()` branch | Tool loop, memory backends, HTTP decorators | Settings, keys, defaults, CLI, exports, dependencies, docs, tests |
| OpenAI-compatible endpoint | `OpenAICompatibleProvider` subclass | Message/tool orchestration | URL, headers, model default, error cases |
| Storage engine | `MemoryStore` + inherited `session()` | `Agent`, providers, route decorators | Optional dependency, exports, lifecycle, scope tests, docs |
| Custom single-session memory | `MemoryBackend` | Provider implementations | Contract tests for messages/add/reset |
| Tool | `@tool` callable | Provider and route code | Type annotations, schema, error/side-effect policy |
| Tool schema feature | `tools.py` canonical schema + provider translators | Memory and transport | Every provider's accepted schema subset and tests |
| HTTP route behavior | FastAPI or `AgentAPI` decorator | Provider payload translators | Signature preservation, JSON/SSE/error tests, docs |
| Multi-agent workflow | Application service, queue, or workflow engine | Core package | Idempotency, job state, correlation IDs, memory scope |

The seam is only the starting point. A feature is complete when all public
surfaces that enumerate or scaffold it agree.

## State-sharing model

### `AgentAPI` application object

One `AgentAPI` instance is intended to live for the ASGI application lifetime.
Its route table, OpenAPI cache, branding paths, and SSE settings are
application-wide. Runtime dependencies such as database clients or producers can
live in FastAPI lifespan state.

### `Agent` object

An `Agent` is stateful. It owns a provider cache, tool registry, and one bound
memory backend. Sharing is safe only when those three things are meant to be
shared.

Do not use one default `Agent()` as a multi-user conversation singleton. The
default `InMemoryMemory` is one private conversation. Prefer an application-wide
store plus a per-request or per-conversation session:

```python
store = RedisStore(redis_url="redis://localhost:6379")


def build_agent(*, tenant_id: str, user_id: str, conversation_id: str) -> Agent:
    memory = store.session(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        agent_id="support",
        source_id="chat",
    )
    return Agent(
        system_prompt="You are a support assistant.",
        provider="openai",
        memory=memory,
    )
```

Even when a backend makes individual reads/appends thread-safe, a complete agent
turn is not transactional. Concurrent `run()` calls against the same scope can
read the same prior history and then interleave final writes. Applications that
allow concurrent turns in one conversation need a per-scope lock, queue, or
optimistic concurrency policy outside the current contract.

### Provider objects

Named providers are cached per `Agent`. The OpenAI-compatible and Gemini
implementations create an `httpx.AsyncClient` per request/stream, so they do not
currently share a connection pool across calls. This keeps ownership simple but
adds connection setup overhead at high throughput.

Custom providers may own reusable clients, but their `chat()` and `stream()`
implementations must be safe for the way the containing Agent is shared.

### Provider registry

`Agent._custom_provider_factories` is class-level mutable process state.
Registration is convenient for application startup and plugins, but names can
collide and registrations persist between tests. Register deterministically at
startup, use namespaced provider names, and restore registry state in isolated
tests.

## Memory backend capability matrix

| Property | `InMemoryStore` | `RedisStore` | `MongoDBStore` |
| --- | --- | --- | --- |
| Scope isolation | Yes | Yes | Yes |
| Shared across threads | Yes, per operation | Client-dependent; standard Redis client is thread-safe | Client-dependent; standard Mongo client is thread-safe |
| Shared across processes/hosts | No | Yes | Yes |
| Persistence | Process lifetime | Redis durability policy | MongoDB durability policy |
| Default retention | Until purged/process exit | Seven-day TTL | Seven-day TTL |
| Atomic append | Lock-protected list append | Redis `RPUSH` | MongoDB `$push` |
| Enumerate by actor/source | Yes | No | Yes |
| Bulk purge by actor/source | Yes | No | Yes |
| Explicit close | Not needed | Yes | Only closes an internally owned client |

The current Redis and MongoDB implementations use synchronous clients from
async route flows. Their operations are small, but they still block the event
loop during network latency. An async-store redesign would require either a new
async memory contract or orchestration that can await both sync and async
backends; do not silently change `messages`, `add()`, or `reset()` into awaitables.

### Store reuse versus compatibility wrappers

The `InMemoryMemory`, `RedisMemory`, and `MongoDBMemory` classes are convenient
single-scope wrappers. Every wrapper instance constructs a wrapper-local store
object. `InMemoryMemory` owns its process data and `RedisMemory` creates its
Redis client; `MongoDBMemory` can instead use an injected client or collection,
which remains externally owned. At scale, construct one `*Store` during
application startup, reuse it, and create cheap `MemorySession` views with
`store.session(...)`.

### Scope is isolation, not authorization

`MemoryScope` creates deterministic keys but does not authorize access. Resolve
`tenant_id` and `user_id` from trusted authentication context, not unvalidated
request fields. Do not expose `scope.key` as a security token. Administrative
`list_conversations()` and `purge()` calls need application-level authorization.

## Streaming scalability

Async iteration provides natural backpressure in the normal SSE path: the next
chunk is requested as the response is consumed. With heartbeats enabled,
`AgentAPI._to_sse_response()` places upstream items in an unbounded
`asyncio.Queue`. A very fast producer and slow client can therefore accumulate
memory. Bounded queues and an explicit overflow policy would be required for a
strict memory ceiling.

Other operating characteristics:

- `_sse_chunk_size` limits frame size, not provider token size or total output.
- Heartbeats keep quiet proxy connections active but consume one background task
  per active adapted stream.
- `X-Accel-Buffering: no` helps nginx, but proxy and load-balancer timeouts remain
  deployment configuration.
- Disconnect/cancellation behavior should be tested whenever the producer loop
  changes.
- Streaming currently does not execute tool calls. Use `run()` for tool-centric
  correctness or design a tested streaming-tool state machine as a separate feature.

## Horizontal API scaling

A multi-worker deployment should use external memory:

```text
load balancer
    -> AgentAPI worker 1 ----\
    -> AgentAPI worker 2 -----+--> Redis or MongoDB scoped memory
    -> AgentAPI worker N ----/
```

Process-local stores, provider registries, OpenAPI caches, and application
dictionaries are not shared between workers. Sticky sessions can reduce visible
breakage but are not a substitute for shared durable state.

The CLI supports Uvicorn's `--workers`, but enabling it does not migrate memory
or job state automatically.

## Multi-agent and workload scaling

The core package deliberately does not prescribe an orchestrator. The examples
under `examples/kafka/` demonstrate three application-level patterns:

- stage-oriented event pipelines;
- an HTTP ingress that enqueues slow work for a consumer group; and
- a tool that publishes a task to a specialist.

Kafka adds partition-based parallelism, replay, backpressure, and loose coupling,
but production workflows also need:

- idempotency keys and safe retry semantics;
- durable result/job storage rather than process dictionaries;
- dead-letter handling and poison-message policy;
- correlation, tenant, user, conversation, and trace identifiers;
- per-job or deliberately shared memory scopes;
- timeouts, cancellation, quotas, and cost controls; and
- independent health/lag/latency monitoring per stage.

The examples are architectural demonstrations. For instance, an agent constructed
once inside a worker keeps memory across every job handled by that worker. A
production worker should create or select memory from the job's explicit scope
unless cross-job history is intentional.

## External-service resilience

The HTTPX-based OpenAI-compatible and Gemini adapters use fixed request timeouts
and normalize common HTTP and network failures. The Anthropic adapter relies on
its SDK's client defaults. The core does not yet provide retries, exponential
backoff, circuit breakers, concurrency limits, rate-limit coordination, or
fallback providers. Those policies should be introduced as explicit, testable
layers; hidden retries inside the agent loop can duplicate tool side effects and
inflate latency/cost.

Tool implementations are also external-service boundaries. AgentAPI catches tool
exceptions and sends their text back to the model. Applications should sanitize
sensitive details, make retried side effects idempotent, validate authorization,
and enforce their own timeouts.

## Observability and operability

The current framework uses Python logging around orchestration failures but has
no built-in metrics or tracing hooks. A scalable observability contribution
should preserve provider independence and define stable events around:

- agent run/stream start and completion;
- provider latency, model, outcome, and token/cost metadata where available;
- tool selection, latency, and outcome without logging sensitive arguments;
- memory read/write latency and scope identifiers in privacy-safe form; and
- active streams, cancellations, queue depth, and heartbeat counts.

Avoid embedding one monitoring vendor directly in provider or memory contracts.
A callback/protocol boundary or standard telemetry API keeps integrations
replaceable.

## Current scaling limits to consider before a feature

- Canonical messages and tool schemas are unvalidated dictionaries.
- The tool loop is sequential and stores only the final turn.
- Streaming and non-stream tool behavior are not equivalent.
- A conversation turn is not atomic across memory operations.
- Redis enumeration/purge is not implemented.
- Synchronous external memory calls can block the async event loop.
- HTTP clients are created per provider call rather than shared.
- Provider factories are process-global mutable state.
- Retry, rate-limit, tracing, and cost policies are application concerns today.
- CI compiles the package but does not currently run the repository tests.

These are design constraints, not reasons to bypass the existing abstractions.
When addressing one, propose the contract change explicitly and preserve backward
compatibility or document the migration.
