# AgentAPI

AgentAPI is a lightweight framework for building agentic AI backends with a familiar Python/FastAPI style.

## What You Get

- A stateful `Agent` with memory and tool calling
- Provider abstraction for OpenAI, Gemini, and OpenRouter
- Simple web API layer using `AgentApp`
- Built-in streaming support through Server-Sent Events (SSE)
- CLI commands to scaffold and run projects

## Core Concepts

### `Agent`

`Agent` handles prompt orchestration, model calls, tool execution, and conversation memory.

### `AgentApp`

`AgentApp` extends FastAPI with chat-focused decorators:

- `@app.chat` for regular or streaming responses
- `@app.stream` as a streaming-only compatibility alias

### `@tool`

Mark plain Python functions as LLM-callable tools. AgentAPI builds provider-compatible tool schemas automatically.

## Quick Example

```python
from agentapi import Agent, AgentApp

app = AgentApp()

agent = Agent(
    system_prompt="You are a helpful assistant",
    provider="openai",
)


@app.chat("/chat")
async def chat(message: str):
    return await agent.run(message)


@app.chat("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
```

## Next Steps

1. Start with [Installation](installation.md)
2. Follow [Getting Started](getting-started.md)
3. Configure [Providers](providers.md)
4. Add [Tool Calling](tools.md)
