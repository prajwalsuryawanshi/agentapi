# Providers

Providers are the intelligence layer behind AgentAPI. They handle communication with Large Language Models (LLMs), manage tool-calling behavior, and process conversational context through a unified interface.

AgentAPI is designed with a flexible provider architecture, allowing you to switch between built-in providers or implement custom providers depending on your workflow and infrastructure needs.

---

## Built-in Providers

AgentAPI includes support for multiple providers out of the box.

| Provider     | Description                                                     | Default Model      |
| ------------ | --------------------------------------------------------------- | ------------------ |
| `openai`     | Optimized for general-purpose reasoning, chat, and tool calling | `gpt-4o-mini`      |
| `gemini`     | Google's Gemini models with native function-calling support     | `gemini-2.5-flash` |
| `openrouter` | Access OpenAI-compatible models through OpenRouter              | `gpt-4o-mini`      |

---

## Selecting a Provider

You can select a provider directly when creating an `Agent`.

```python
from agentapi import Agent

agent = Agent(
    system_prompt="You are helpful",
    provider="gemini",
)
```

If no provider is specified, AgentAPI automatically falls back to the `DEFAULT_PROVIDER` environment variable.

---

## Provider Configuration

Configure provider API keys using environment variables.

### OpenAI

```bash
OPENAI_API_KEY=your_api_key_here
```

### Gemini

```bash
GEMINI_API_KEY=your_api_key_here
```

### OpenRouter

```bash
OPENROUTER_API_KEY=your_api_key_here
```

You can also explicitly override the default model:

```python
agent = Agent(
    system_prompt="You are helpful",
    provider="openai",
    model="gpt-4o",
)
```

---

## Provider-specific Tool Calling Defaults

AgentAPI automatically applies provider-aware defaults for tool-calling behavior.

### OpenAI-compatible Providers

```python
tool_choice = "auto"
parallel_tool_calls = True
```

### Gemini

```python
mode = "AUTO"
```

You can override these defaults using the `tool_calling=` parameter when needed.

---

## Custom Providers

AgentAPI supports fully custom providers through the `BaseProvider` interface.

This allows you to:

* integrate internal LLM infrastructure
* connect private APIs
* use self-hosted models
* customize orchestration behavior

---

## The BaseProvider Interface

All custom providers should inherit from the abstract `BaseProvider` class.

```python
async def generate_response(
    self,
    prompt: str,
    system_prompt: str,
) -> str:
    """
    Generate a response from the LLM.

    Args:
        prompt (str):
            The primary user input or query.

        system_prompt (str):
            Instructions guiding model behavior.

    Returns:
        str:
            The generated model response.
    """
    pass
```

---

## Example: Custom HTTP Provider

Below is a practical example of implementing a provider using a custom HTTP endpoint.

```python
import aiohttp

from agentapi.providers.base import BaseProvider


class MyCustomProvider(BaseProvider):
    """
    A custom provider communicating with a private LLM endpoint.
    """

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str,
    ) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
            ) as response:

                response.raise_for_status()

                data = await response.json()

                return data["choices"][0]["message"]["content"]
```

---

## Registering a Custom Provider

Once your provider is implemented, register it using `Agent.register_provider()`.

```python
from agentapi import Agent

Agent.register_provider(
    "myprovider",
    lambda agent, settings, model: MyCustomProvider(
        api_url="https://example.com/v1/chat",
        api_key="your_api_key",
    ),
)
```

Then use it like any built-in provider:

```python
agent = Agent(
    system_prompt="You are helpful",
    provider="myprovider",
)
```
