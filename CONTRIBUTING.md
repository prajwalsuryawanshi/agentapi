# Contributing to AgentAPI

Thanks for your interest in contributing to AgentAPI.

## Development Setup

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies in editable mode.

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

## Local Run

Run the example app:

```bash
uvicorn examples.main:app --reload
```

## Code Style

- Keep changes focused and minimal.
- Add docstrings for public APIs.
- Update docs for user-facing behavior changes.
- Keep compatibility with Python 3.10+.

## Pull Requests

1. Create a feature branch.
2. Commit with clear messages.
3. Open a pull request with:
   - Problem statement
   - What changed
   - How you tested it

## Commit Message Guidance

Use short, action-oriented commit messages.

Examples:

- `chore: add pypi publish workflow`
- `docs: improve provider configuration section`
- `fix: handle streaming http errors safely`

## Reporting Issues

Use the issue templates and include reproduction steps.
