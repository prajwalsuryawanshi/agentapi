# Tool Calling

Tool calling lets the model invoke your Python functions during a conversation.

## Define a Tool

```python
from agentapi import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city in plain text."""
    return f"Weather in {city}: sunny"
```

## Attach Tools to an Agent

```python
from agentapi import Agent
from tools import get_weather

agent = Agent(
    system_prompt="You are a weather assistant",
    provider="openai",
    tools=[get_weather],
)
```

## How Tool Execution Works

1. User message is sent to provider with tool schemas.
2. Provider may return tool calls.
3. Agent executes matching Python functions.
4. Tool outputs are appended to conversation memory.
5. Agent asks provider again for final response.

## Recommended Tool Authoring

- Add concise docstrings that state what the tool does and returns.
- Use clear argument names.
- Keep outputs structured and predictable.
- Handle exceptions inside tools for graceful failure messages.

## Parsing and Validation

AgentAPI safely parses tool arguments from model JSON payloads before execution.
