# AgentAPI

![AgentAPI](https://raw.githubusercontent.com/prajwalsuryawanshi/agentapi/main/agentapi/assets/agentapi-logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![Status](https://img.shields.io/badge/status-MVP-success.svg)](#project-status)
[![Docs](https://img.shields.io/badge/docs-site-blue)](https://agentapi.prajwalsuryawanshi.in)

AgentAPI is an open-source Python framework for building agentic AI backends with a clean developer experience: minimal setup, provider abstraction, tool calling, memory, and streaming-first APIs.

It is designed for teams that want FastAPI-style simplicity for LLM agents, without heavy orchestration overhead.

Documentation site: `https://agentapi.prajwalsuryawanshi.in`

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
from agentapi import AgentApp, Agent

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

Run it:

```bash
uvicorn main:app --reload
```

Open docs:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## Provider Configuration

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

MIT License. See `LICENSE`.
