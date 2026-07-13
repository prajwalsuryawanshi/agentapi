import asyncio
from agentapi.agent.agent import Agent
from agentapi.agent.hooks import AgentHook
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall

class LoggingHook(AgentHook):
    async def on_agent_start(self, run_id: str, message: str, **kwargs):
        print(f"[HOOK] Agent Start! ID: {run_id} | Message: '{message}'")

    async def on_llm_start(self, run_id: str, messages: list, **kwargs):
        print(f"[HOOK] LLM Queried with {len(messages)} messages.")

    async def on_llm_end(self, run_id: str, response: ProviderResponse, **kwargs):
        print(f"[HOOK] LLM Responded! Tool calls: {len(response.tool_calls)}")

    async def on_tool_start(self, run_id: str, tool_call: ToolCall, **kwargs):
        print(f"[HOOK] Tool '{tool_call.name}' started with args: {tool_call.arguments}")

    async def on_tool_end(self, run_id: str, tool_call: ToolCall, result: str, **kwargs):
        print(f"[HOOK] Tool '{tool_call.name}' finished. Result: {result}")

    async def on_agent_end(self, run_id: str, final_response, **kwargs):
        print(f"[HOOK] Agent Finished! Output length: {len(str(final_response))}")

class MockProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        if len(messages) == 2:
            return ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="call_1", name="get_weather", arguments='{"location": "Tokyo"}')],
                raw_message={}
            )
        return ProviderResponse(
            content="The weather in Tokyo is sunny.",
            tool_calls=[],
            raw_message={}
        )

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield ""

def get_weather(location: str) -> str:
    return f"Sunny in {location}"

async def main():
    hook = LoggingHook()
    agent = Agent(
        system_prompt="You are a helpful assistant.",
        provider=MockProvider(),
        tools=[get_weather],
        hooks=[hook]
    )
    
    print("Running with LoggingHook...")
    result = await agent.run("What is the weather in Tokyo?")
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
