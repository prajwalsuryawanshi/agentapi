# Memory

AgentAPI separates the idea of memory from the agent itself. The agent asks a memory backend for the current conversation state, and the backend decides where that state lives.

## Why Memory Is Abstracted

The abstraction lets you choose the right storage for the job without changing your agent code:

- `InMemoryMemory` for local development, demos, and tests
- `RedisMemory` for production, multi-worker setups, and shared session state

This keeps the `Agent` API stable while letting you swap storage later.

## Memory Backends

### `InMemoryMemory`

Use this when you want the simplest setup:

- no external service required
- fast local iteration
- session isolation through `conversation_id`

Example:

```python
from agentapi import Agent, InMemoryMemory, create_conversation_id

conversation_id = create_conversation_id()

memory = InMemoryMemory(
    conversation_id=conversation_id,
)

agent = Agent(
    system_prompt="You are a helpful assistant.",
    provider="openai",
    memory=memory,
)
```

### `RedisMemory`

Use this when you need persistence across workers or servers:

- shared state across multiple processes
- better for production deployments
- optional metadata like `user_id` and `tenant_id`

Example:

```python
from agentapi import Agent, RedisMemory, create_conversation_id

memory = RedisMemory(
    redis_url="redis://localhost:6379",
    conversation_id=create_conversation_id(),
    user_id="123",
    tenant_id="acme",
)

agent = Agent(
    system_prompt="You are a helpful assistant.",
    provider="gemini",
    memory=memory,
)
```

## How Conversation IDs Work

`conversation_id` is the isolation key.

- For unauthenticated users, generate a UUID with `create_conversation_id()`.
- For authenticated users, derive it from your auth/session layer.
- The memory backend does not care how the ID is created, only that it is consistent for the same conversation.

Common examples:

```python
conversation_id = create_conversation_id()
```
Store mappings between users/sessions and conversation IDs separately in your application layer if needed.
## Using Memory in an App

Create one memory object per session or conversation and pass it into the agent:

```python
from agentapi import Agent, InMemoryMemory

memory = InMemoryMemory(
    conversation_id="550e8400-e29b-41d4-a716-446655440000",
)

agent = Agent(
    system_prompt="You are concise.",
    provider="openai",
    memory=memory,
)
```

For multi-user APIs, keep a mapping from `conversation_id` to memory objects in your web layer. In production, prefer `RedisMemory` so the state is shared across workers.

## Multiple Agents with Independent Memory

You can create as many memory instances as you need—each one is completely isolated with zero conflicts.

```python
from agentapi import Agent, InMemoryMemory, create_conversation_id
import asyncio

# Create 5 separate memory instances for 5 agents
memories = [
    InMemoryMemory(conversation_id=create_conversation_id())
    for _ in range(5)
]

# Create 5 agents, each with its own memory
agents = [
    Agent(
        system_prompt=f"You are Agent {i+1} specializing in {['research', 'editing', 'summarization', 'analysis', 'planning'][i]}",
        provider="openai",
        memory=memories[i],
    )
    for i in range(5)
]

async def main():
    # Each agent processes independently — no cross-talk
    results = await asyncio.gather(
        agents[0].run("Research the top 3 AI trends"),
        agents[1].run("Edit this paragraph for clarity"),
        agents[2].run("Summarize the meeting notes"),
        agents[3].run("Analyze the sales data"),
        agents[4].run("Plan the product roadmap"),
    )
    
    return results

# Each agent maintains its own conversation history
# agents[0].memory.messages contains only Agent 0's conversation
# agents[1].memory.messages contains only Agent 1's conversation
# ...etc
```

**Key points:**

- Each `InMemoryMemory()` instance is independent and process-local
- 5 agents + 5 memory instances = 5 completely isolated conversation stores
- Perfect for local development, testing, and multi-agent scenarios
- In production with multiple workers, use `RedisMemory` with shared `conversation_id` for inter-process memory

## Resetting Memory

Use `reset_memory()` on the agent when you want to start a new conversation while keeping the same system prompt.

```python
agent.reset_memory()
```

If you need a different prompt for the next turn, create a new `Agent` with a different `system_prompt`.
