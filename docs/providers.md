# Providers

AgentAPI supports multiple providers behind one interface.

## Built-in Providers

- `openai`
- `gemini`
- `openrouter`

## Selecting a Provider

```python
agent = Agent(system_prompt="You are helpful", provider="gemini")
```

If omitted, `DEFAULT_PROVIDER` from environment is used.

## Default Models

- OpenAI/OpenRouter: `gpt-4o-mini`
- Gemini: `gemini-2.5-flash`

You can always override explicitly:

```python
agent = Agent(
    system_prompt="You are helpful",
    provider="openai",
    model="gpt-4o",
)
```

## Provider-specific Tool Calling Defaults

AgentAPI applies provider-aware defaults automatically:

- OpenAI-compatible:
  - `tool_choice = "auto"`
  - `parallel_tool_calls = true`
- Gemini:
  - `mode = "AUTO"` for function calling

You can override with `tool_calling=`.

## Custom Providers

Register your own provider factory:

```python
from agentapi import Agent, BaseProvider
from agentapi.providers.base import ProviderResponse


class MyProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(content="hello", tool_calls=[], raw_message={})

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield "hello"


Agent.register_provider("myprovider", lambda agent, settings, model: MyProvider())
```

Then use it:

```python
agent = Agent(system_prompt="You are helpful", provider="myprovider")
```
