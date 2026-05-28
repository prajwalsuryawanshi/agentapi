"""Conversation-aware AgentAPI example."""

from __future__ import annotations

from agentapi import Agent, AgentAPI, InMemoryMemory, create_conversation_id


app = AgentAPI()
_agents: dict[str, Agent] = {}


def get_agent(conversation_id: str) -> Agent:
    agent = _agents.get(conversation_id)
    if agent is None:
        agent = Agent(
            system_prompt="You are a helpful assistant.",
            provider="openai",
            memory=InMemoryMemory(conversation_id=conversation_id),
        )
        _agents[conversation_id] = agent

    return agent


@app.chat("/chat")
async def chat(message: str, conversation_id: str | None = None):
    conversation_id = conversation_id or create_conversation_id()
    agent = get_agent(conversation_id)
    reply = await agent.run(message)
    return {"conversation_id": conversation_id, "reply": reply}


@app.chat("/stream")
async def stream_chat(message: str, conversation_id: str | None = None):
    conversation_id = conversation_id or create_conversation_id()
    agent = get_agent(conversation_id)
    return agent.stream(message)