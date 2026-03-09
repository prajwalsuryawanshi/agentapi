# CLI

AgentAPI provides a small CLI for scaffolding and running apps.

## Commands

## `agentapi new`

Create a new project.

```bash
agentapi new myproject
```

Interactive mode:

```bash
agentapi new
```

Options:

- `--provider {openai,gemini,openrouter}`
- `--interactive`

Generated files:

- `main.py`
- `tools.py`
- `agents.py`
- `.env`

## `agentapi run`

Run an ASGI app via uvicorn.

```bash
agentapi run --app main:app --reload
```

Options:

- `--host` (default: `127.0.0.1`)
- `--port` (default: `8000`)
- `--reload`
- `--workers`

## Module Invocation

You can also run through Python:

```bash
python -m agentapi --help
```
