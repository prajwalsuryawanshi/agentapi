# CLI

AgentAPI provides a lightweight CLI for scaffolding and running AI agent applications with minimal setup.

---

# Commands

## `agentapi new`

Create a new AgentAPI project scaffold.

### Basic Usage

```bash
agentapi new myproject
```

This command generates a starter project with the required files and configuration.

### Interactive Mode

You can also create a project using interactive prompts:

```bash
agentapi new
```

### Options

| Option | Description |
|--------|-------------|
| `--provider` | Select the default provider (`openai`, `gemini`, `openrouter`) |
| `--interactive` | Enable interactive project setup |

### Example

```bash
agentapi new myproject --provider openai
```

### Generated Files

```text
myproject/
├── main.py
├── agents.py
├── tools.py
└── .env
```

### File Overview

| File | Purpose |
|------|---------|
| `main.py` | Main application entry point |
| `agents.py` | Agent definitions and configurations |
| `tools.py` | Custom tool functions |
| `.env` | Environment variable configuration |

---

## `agentapi run`

Run an ASGI application using `uvicorn`.

### Basic Usage

```bash
agentapi run --app main:app --reload
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host address |
| `--port` | `8000` | Server port |
| `--reload` | Disabled | Enable auto-reload during development |
| `--workers` | `1` | Number of worker processes |

### Example

Run the application on a custom port:

```bash
agentapi run --app main:app --port 8080
```

Run the application with multiple workers:

```bash
agentapi run --app main:app --workers 4
```

---

# Module Invocation

You can also run AgentAPI using Python module invocation:

```bash
python -m agentapi --help
```

This is useful in environments where the `agentapi` command is unavailable in the system path.

---

# Development Workflow Example

Create a new project:

```bash
agentapi new myproject
```

Move into the project directory:

```bash
cd myproject
```

Run the development server:

```bash
agentapi run --app main:app --reload
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

---

# Troubleshooting

## `agentapi: command not found`

Ensure AgentAPI is installed correctly:

```bash
pip install agentapi-core
```

If using a virtual environment, make sure it is activated.

---

## Port Already in Use

Run the server on a different port:

```bash
agentapi run --app main:app --port 8080
```

---

# Tips

- Use `--reload` during development for automatic server restarts.
- Keep tools modular inside `tools.py` for better maintainability.
- Use the interactive setup if you are new to AgentAPI.
