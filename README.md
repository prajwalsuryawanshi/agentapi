# AgentAPI

![AgentAPI](https://raw.githubusercontent.com/prajwalsuryawanshi/agentapi/main/agentapi/assets/agentapi-logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![Status](https://img.shields.io/badge/status-MVP-success.svg)](#project-status)
[![Docs](https://img.shields.io/badge/docs-site-blue)](https://agentapi.prajwalsuryawanshi.in)

AgentAPI is a lightweight Python framework for building agent backends with FastAPI-style ergonomics: provider abstraction, tool calling, conversation memory, and streaming-first APIs.

It targets teams who want minimal setup for agentic workflows while keeping the system flexible for production (Redis memory, custom providers, tool schemas).

Documentation site: https://agentapi.prajwalsuryawanshi.in

## Table of Contents

- [Why AgentAPI](#why-agentapi)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Provider Configuration](#provider-configuration)
- [Tool Calling](#tool-calling)
- [Streaming](#streaming)
- [CLI](#cli)
- [Custom Providers](#custom-providers)
- [Error Handling](#error-handling)
- [Project Structure](#project-structure)
- [Project Status](#project-status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why AgentAPI

- Keep agent backends simple and readable.
- Use one `Agent` interface across providers.
- Add tools as plain Python functions.
- Stream responses with minimal boilerplate.
- Start fast, then customize deeply when needed.

## Features

- `Agent` class with memory and tool execution loop.
- Provider abstraction for `openai`, `gemini`, and `openrouter`.
- AgentAPI app integration with `@app.chat`.
- Automatic SSE when a chat handler returns an async iterator.
- Built-in project scaffolding and run helper via CLI.
- Environment-based configuration using `.env`.
- Extensible provider system (custom instance or registered factory).

## Installation

Install from PyPI:

```bash
pip install agentapi-core
```

Install in editable mode while developing:

```bash
pip install -e .
```

## Quick Start

Create `main.py`:

```python
from agentapi import AgentAPI, Agent, InMemoryMemory, create_conversation_id

app = AgentAPI()

# Agent owns the system prompt; memory stores only user/assistant/tool turns
agent = Agent(
  system_prompt="You are a helpful assistant",
  provider="openai",
  memory=InMemoryMemory(),
)


@app.chat("/chat")
async def chat(message: str, conversation_id: str | None = None):
  # create a conversation id for anonymous users
  conversation_id = conversation_id or create_conversation_id()
  return await agent.run(message, conversation_id=conversation_id)


@app.chat("/stream")
async def stream_chat(message: str, conversation_id: str | None = None):
  conversation_id = conversation_id or create_conversation_id()
  return agent.stream(message, conversation_id=conversation_id)
```

Run the app:

```bash
uvicorn main:app --reload
```

Open docs locally:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Memory and Providers

AgentAPI separates the Agent (which owns `system_prompt`) from the memory backend. Memory backends store conversational turns and are conversation-scoped by a `conversation_id` (UUID). Use `InMemoryMemory` for local development and `RedisMemory` for multi-worker/production deployments.

Provider configuration remains the same — set API keys via `.env` and choose a provider name (or pass a custom `BaseProvider` instance).

### Provider Configuration

Create `.env`:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
DEFAULT_PROVIDER=openai
```

Supported provider names:

- `openai`
- `gemini`
- `openrouter`

## Tool Calling

Define tools with plain Python:

```python
from agentapi import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: sunny"
```

Attach tools to an agent:

```python
agent = Agent(
    system_prompt="You are a weather assistant",
    provider="openai",
    tools=[get_weather],
)
```

Tool schemas are generated from function signatures and mapped to provider-specific tool formats internally.

## Streaming

`@app.chat` auto-switches to SSE (`text/event-stream`) when your handler returns an async iterator.

Example:

```python
@app.chat("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
```
### Configuring SSE for proxies

If you put AgentAPI behind nginx, Cloudflare, or any other reverse proxy, long streams can get cut off — most proxies kill idle connections after 30–60 seconds. To handle this, you can tune two streaming options:

```python
app = AgentAPI(
    sse_chunk_size=1024,        # how much text per `data:` event (default: 64)
    sse_heartbeat_seconds=15,   # send a keepalive every N seconds (default: off)
)
```

Both default to the old behavior, so adding these won't break anything if you were already using AgentAPI.

When `sse_heartbeat_seconds` is set, AgentAPI sends `: keepalive\n\n` during quiet periods. It's a valid SSE comment — clients ignore it, but proxies see traffic on the wire and leave the connection alone.

A typical nginx config that plays well with SSE:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    proxy_buffering off;
    proxy_cache off;
    chunked_transfer_encoding off;
    proxy_read_timeout 1h;
}
```

The key bits are `proxy_buffering off` (so events aren't held back) and a long `proxy_read_timeout` paired with the heartbeat above.

## CLI

Create a new project scaffold:

```bash
agentapi new myproject
```

Interactive setup (asks for project name and provider):

```bash
agentapi new
```

Run app via helper:

```bash
agentapi run --app main:app --reload
```

## Custom Providers

`OpenAICompatibleProvider` is an internal helper for OpenAI-compatible APIs. AgentAPI is not locked to it.

You can customize providers in two ways:

1. Pass a provider instance directly: `provider=<BaseProvider instance>`
2. Register a provider factory and reference by name.

```python
from agentapi import Agent, BaseProvider
from agentapi.providers.base import ProviderResponse


class MyProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(content="hello", tool_calls=[], raw_message={})

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield "hello"


Agent.register_provider(
    "myprovider",
    lambda agent, settings, model: MyProvider(),
)

agent = Agent(system_prompt="You are helpful", provider="myprovider")
```

## Error Handling

AgentAPI converts common runtime issues into clear API-level errors:

- Missing API keys -> configuration error message.
- Upstream provider failures -> provider error message with status context.
- Streaming endpoints emit SSE error events instead of hard crashes.

## Examples & docs

See `examples/` for runnable examples and `docs/` for the memory abstraction guide and orchestration notes. Key example:

- `examples/multi_user_example.py` — conversation-aware FastAPI app using `create_conversation_id()` and `InMemoryMemory`.

## Project Structure

```text
agentapi/
  agent/
    agent.py
    memory.py
    tools.py
  assets/
    agentapi-logo.png
    agentapi-favicon.png
  config/
    settings.py
  core/
    app.py
  providers/
    base.py
    gemini.py
    openai_compatible.py
    openai.py
    openrouter.py
examples/
  main.py
```

## Project Status

Current phase: MVP

Implemented:

- Core agent runtime
- Provider abstraction (OpenAI, Gemini, OpenRouter)
- Tool calling and in-memory conversation memory
- Automatic SSE streaming on chat endpoints
- CLI scaffolding and run helper

## Roadmap

- Add Anthropic provider.
- Expand memory backends (Redis/Postgres).
- Add richer observability and tracing hooks.
- Improve generated project templates.
- Add test suite and CI workflows.
 - Automated RAG for code — `agent.add_knowledge('url-to-file')`.

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for setup and PR workflow.

## Publishing

This repository is configured to publish on GitHub Release.

1. Bump `version` in `pyproject.toml`.
2. Commit and tag a release version.
3. Create a GitHub Release.
4. GitHub Actions publishes to PyPI using trusted publishing.

Required one-time setup:

- In PyPI, create project `agentapi-core`.
- Configure trusted publisher for this GitHub repository.
- Keep release workflow enabled in `.github/workflows/publish.yml`.

### Trusted Publisher Values

Use these exact values when adding publishers.

TestPyPI pending publisher:

- Project name: `agentapi-core`
- Owner: `prajwalsuryawanshi`
- Repository: `agentapi`
- Workflow filename: `publish-testpypi.yml`
- Environment name: `testpypi`

PyPI trusted publisher (after project exists):

- Project name: `agentapi-core`
- Owner: `prajwalsuryawanshi`
- Repository: `agentapi`
- Workflow filename: `publish.yml`
- Environment name: `pypi`

Note: package names on PyPI are normalized, so use lowercase `agentapi-core`.

## License

MIT License. See [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
