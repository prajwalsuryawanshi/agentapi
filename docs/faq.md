# FAQ

## Why is package name `agentapi-core` but imports are `agentapi`?

PyPI distribution name and Python import package name are separate. Install with:

```bash
pip install agentapi-core
```

Import with:

```python
import agentapi
```

## Why does PyPI reject a project name as too similar?

PyPI enforces anti-typosquatting checks. Pick a more unique project name if blocked.

## Why did README image not show on PyPI?

PyPI needs absolute image URLs. Repo-relative paths usually fail in rendered descriptions.

## Do I need API tokens for publishing?

Not with OIDC Trusted Publisher workflows. Tokens are optional for manual uploads only.

## How do I stream responses?

Return an async iterator from `@app.chat` and AgentAPI will emit SSE automatically.

## How do I clear chat context?

Call:

```python
agent.reset_memory()
```
