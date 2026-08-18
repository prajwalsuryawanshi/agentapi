# AgentAPI

AgentAPI is a lightweight framework for building multi-agent AI backends with a familiar Python and FastAPI workflow. It gives you an `Agent` runtime, pluggable memory backends, provider abstraction, streaming support, and a small web layer for shipping real APIs quickly.

## At A Glance

- Build one agent or many agents in the same application
- Switch providers without changing orchestration code
- Isolate conversations with per-user or per-session memory
- Stream responses with Server-Sent Events when needed
- Scaffold a new project with the CLI in a few seconds

## Core Primitives

### `Agent`

`Agent` owns the system prompt, provider selection, tool execution, and response flow.

Use it when you want to:

- run a single assistant endpoint
- chain multiple model calls together
- give each agent a specialized role such as planner, editor, or summarizer

### `AgentAPI`

`AgentAPI` extends FastAPI and provides chat-oriented route helpers:

- `@app.chat` for routes that return JSON or SSE depending on the handler output
- `@app.stream` for explicit streaming-only endpoints
- branded OpenAPI, Swagger UI, and ReDoc pages

### Memory Backends

AgentAPI keeps memory isolated from the agent logic so you can choose the right storage model:

- `InMemoryMemory` for development, tests, and local multi-agent work
- `RedisMemory` for production, shared sessions, and multi-worker deployments

### `@tool`

Decorate ordinary Python functions to make them available to the model as tools. AgentAPI converts them into provider-compatible schemas automatically.

## Quick Start

```python
from agentapi import Agent, AgentAPI

app = AgentAPI()

assistant = Agent(
    system_prompt="You are a concise and reliable assistant.",
    provider="openai",
)


@app.chat("/chat")
async def chat(message: str):
    return await assistant.run(message)


@app.chat("/stream")
async def stream_chat(message: str):
    return assistant.stream(message)
```

## Multi-Agent Orchestration

AgentAPI is built for composing multiple agents in one app. A common setup is to assign each agent a role and a memory boundary.

```python
import asyncio
from agentapi import Agent, AgentAPI, InMemoryMemory, RedisMemory, create_conversation_id

app = AgentAPI()

conversation_id = create_conversation_id()

research_agent = Agent(
    system_prompt="You are a research specialist.",
    provider="openai",
    memory=RedisMemory(redis_url="redis://localhost:6379", conversation_id=conversation_id),
)

editor_agent = Agent(
    system_prompt="You are a precise editor.",
    provider="gemini",
    memory=InMemoryMemory(conversation_id=create_conversation_id()),
)


@app.post("/orchestrate")
async def orchestrate(message: str):
    research, edit = await asyncio.gather(
        research_agent.run(message),
        editor_agent.run(message),
    )

    return {
        "research": research,
        "edit": edit,
        "summary": f"Research:\n{research}\n\nEdit:\n{edit}",
    }
```

Good orchestration patterns include:

- planner plus executor
- draft plus critique
- router plus specialist agents
- one agent calling another agent as a tool

## When To Use Which Memory

- Use `InMemoryMemory` when you want simple, isolated session state in one process.
- Use `RedisMemory` when multiple workers or servers must see the same conversation.
- Give each user, session, or tenant a unique `conversation_id`.

## Production Notes

- Keep the system prompt on the `Agent`, not in the memory backend.
- Use `RedisMemory` for multi-worker deployments.
- Use `conversation_id` as the primary isolation key for user sessions.
- Store agent state per user or per agent if you are orchestrating more than one model.

## Documentation Map

1. Start with [Installation](installation.md)
2. Read [Getting Started](getting-started.md)
3. For framework contributions, study the [Architecture Guide](architecture/index.md)
4. Review [Providers](providers.md)
5. Learn [Memory](memory.md)
6. Add [Tool Calling](tools.md)
7. Check [Streaming](streaming.md)
8. Review [CLI](cli.md)
9. Read the [FAQ](faq.md)
