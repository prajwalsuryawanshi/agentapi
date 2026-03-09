# Installation

## Requirements

- Python 3.10+
- A provider API key (OpenAI, Gemini, or OpenRouter)

## Install from PyPI

```bash
pip install agentapi-core
```

## Install for Local Development

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

## Verify Installation

```bash
python -c "import agentapi; print(agentapi.__all__)"
```

## Environment Variables

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
DEFAULT_PROVIDER=openai
```

## Notes

- `DEFAULT_PROVIDER` is used when you do not explicitly pass `provider=` to `Agent`.
- API key errors are surfaced as `AgentConfigurationError` with clear guidance.
