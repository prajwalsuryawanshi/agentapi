"""Minimal AgentAPI example."""

from agentapi import Agent, AgentApp

app = AgentApp()

agent = Agent(
    system_prompt="You are a helpful assistant",
    provider="openai",
)

@app.chat("/chat")
async def chat(message: str):
    return await agent.run(message)


@app.chat("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
