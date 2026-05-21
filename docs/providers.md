# Providers

Providers are the intelligence layer behind AgentAPI. They handle communication with Large Language Models (LLMs), manage tool-calling behavior, and process conversational context through a unified interface.

AgentAPI is designed with a flexible provider architecture, allowing you to switch between built-in providers or implement fully custom providers depending on your workflow and infrastructure requirements.

---

# Built-in Providers

AgentAPI includes support for multiple providers out of the box.

| Provider     | Description                                                               | Default Model      |
| ------------ | ------------------------------------------------------------------------- | ------------------ |
| `openai`     | Optimized for reasoning, tool calling, and general-purpose chat workflows | `gpt-4o-mini`      |
| `gemini`     | Google's Gemini models with native function-calling support               | `gemini-2.5-flash` |
| `openrouter` | Access OpenAI-compatible models through OpenRouter                        | `gpt-4o-mini`      |

---

# Selecting a Provider

You can configure a provider directly when creating an `Agent`.

```python id="21lmz5"
from agentapi import Agent

agent = Agent(
    system_prompt="You are helpful",
    provider="gemini",
)
```

If no provider is specified, AgentAPI automatically uses the `DEFAULT_PROVIDER` environment variable.

---

# Agent Provider Configuration

The `Agent` class accepts provider-related configuration parameters.

```python id="g2gw9w"
from agentapi import Agent

agent = Agent(
    system_prompt="You are helpful",
    provider="openai",
    model="gpt-4o",
)
```

## Parameters

| Parameter       | Type                 | Description                                                               |
| --------------- | -------------------- | ------------------------------------------------------------------------- |
| `system_prompt` | `str`                | Global instructions guiding model behavior                                |
| `provider`      | `str`                | Provider name (`openai`, `gemini`, `openrouter`, or custom provider name) |
| `model`         | `str \| None`        | Optional model override                                                   |
| `tools`         | `list \| None`       | List of registered tools available to the agent                           |
| `memory`        | `BaseMemory \| None` | Conversation memory backend                                               |

---

# Provider Configuration

Configure provider API keys using environment variables.

## OpenAI

```bash id="h5p74r"
OPENAI_API_KEY=your_api_key_here
```

## Gemini

```bash id="i5itx2"
GEMINI_API_KEY=your_api_key_here
```

## OpenRouter

```bash id="4y4my6"
OPENROUTER_API_KEY=your_api_key_here
```

You can also configure a default provider globally:

```bash id="4mrh0r"
DEFAULT_PROVIDER=openai
```

---

# Default Models

AgentAPI automatically assigns default models for built-in providers.

| Provider     | Default Model      |
| ------------ | ------------------ |
| `openai`     | `gpt-4o-mini`      |
| `gemini`     | `gemini-2.5-flash` |
| `openrouter` | `gpt-4o-mini`      |

You can override the default model explicitly:

```python id="ypgxq1"
agent = Agent(
    system_prompt="You are helpful",
    provider="openai",
    model="gpt-4o",
)
```

---

# Provider-specific Tool Calling Defaults

AgentAPI automatically applies provider-aware defaults internally.

## OpenAI-compatible Providers

```python id="fr8l6n"
tool_choice = "auto"
parallel_tool_calls = True
```

## Gemini

```python id="k4zd04"
mode = "AUTO"
```

You can override these defaults using the `tool_calling=` parameter.

---

# BaseProvider

Custom providers must inherit from the abstract `BaseProvider` class.

```python id="4s6qcu"
from agentapi import BaseProvider
```

The `BaseProvider` interface defines the methods required for all providers.

---

# BaseProvider Methods

## chat()

Used for standard non-streaming responses.

```python id="s8fgin"
async def chat(
    self,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_calling: dict[str, Any] | None = None,
) -> ProviderResponse:
```

### Parameters

| Parameter      | Type                           | Description                                  |
| -------------- | ------------------------------ | -------------------------------------------- |
| `messages`     | `list[dict[str, Any]]`         | Conversation message history                 |
| `tools`        | `list[dict[str, Any]] \| None` | Tool definitions available to the provider   |
| `tool_calling` | `dict[str, Any] \| None`       | Provider-specific tool-calling configuration |

### Returns

Returns a `ProviderResponse` object containing:

* generated content
* tool calls
* raw provider response data

---

## stream()

Used for streaming incremental responses.

```python id="iqmxql"
async def stream(
    self,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_calling: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
```

### Parameters

| Parameter      | Type                           | Description                                  |
| -------------- | ------------------------------ | -------------------------------------------- |
| `messages`     | `list[dict[str, Any]]`         | Conversation message history                 |
| `tools`        | `list[dict[str, Any]] \| None` | Tool definitions available to the provider   |
| `tool_calling` | `dict[str, Any] \| None`       | Provider-specific tool-calling configuration |

### Yields

Streaming text chunks incrementally as they are generated by the provider.

---

# ProviderResponse

Providers return a `ProviderResponse` object from the `chat()` method.

```python id="d27m5t"
ProviderResponse(
    content="hello",
    tool_calls=[],
    raw_message={},
)
```

## Fields

| Field         | Type   | Description                            |
| ------------- | ------ | -------------------------------------- |
| `content`     | `str`  | Generated response text                |
| `tool_calls`  | `list` | Tool calls returned by the provider    |
| `raw_message` | `dict` | Raw provider-specific response payload |

---

# Custom Providers

AgentAPI supports fully custom providers for:

* private APIs
* self-hosted models
* internal infrastructure
* experimental orchestration layers

---

# Example: Custom HTTP Provider

Below is a practical example of implementing a custom provider using a custom HTTP endpoint.

```python id="om5tx3"
import aiohttp

from agentapi.providers.base import BaseProvider, ProviderResponse


class MyCustomProvider(BaseProvider):

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    async def chat(
        self,
        messages,
        *,
        tools=None,
        tool_calling=None,
    ) -> ProviderResponse:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": messages,
            "tools": tools,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
            ) as response:

                response.raise_for_status()

                data = await response.json()

                return ProviderResponse(
                    content=data["choices"][0]["message"]["content"],
                    tool_calls=[],
                    raw_message=data,
                )

    async def stream(
        self,
        messages,
        *,
        tools=None,
        tool_calling=None,
    ):

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json={"messages": messages},
            ) as response:

                async for chunk in response.content:
                    yield chunk.decode()
```

---

# Registering a Custom Provider

Once your provider is implemented, register it using `Agent.register_provider()`.

```python id="jlwmrg"
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

```python id="tjlwmq"
agent = Agent(
    system_prompt="You are helpful",
    provider="myprovider",
)
```

---

# Developer Notes

* Provider implementations can internally manage retries, headers, authentication, and request formatting.
* Tool-calling behavior differs slightly across providers. AgentAPI applies sensible defaults automatically.
* Streaming providers should yield incremental text chunks for SSE compatibility.
* Custom providers can integrate with any REST API, local model runtime, or orchestration system.
