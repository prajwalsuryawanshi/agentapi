"""Minimal AgentAPI example."""

from agentapi import Agent, AgentAPI, tool

app = AgentAPI()


@tool
def get_date() -> str:
    from datetime import datetime

    return datetime.now().isoformat()


agent = Agent(
    system_prompt="You are a helpful assistant",
    provider="openai",
    tools=[get_date],
)

agent2 = Agent(
    system_prompt="You are a chess master who loves to teach chess", 
    provider="openai",
)

@app.chat("/chat")
async def chat(message: str):
    return await agent.run(message)


@app.chat("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
