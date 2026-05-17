import asyncio

from agentapi.agent.agent import Agent
from agentapi.agent.tools import tool
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall


@tool
def weather(city: str) -> str:
    return f"Weather in {city}: sunny"


class FakeToolProvider(BaseProvider):
    def __init__(self) -> None:
        self.chat_calls = 0
        self.stream_called = False

    async def chat(
        self,
        messages,
        *,
        tools=None,
        tool_calling=None,
    ) -> ProviderResponse:
        self.chat_calls += 1
        if self.chat_calls == 1:
            return ProviderResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="weather", arguments='{"city": "pune"}')],
                raw_message={},
            )
        return ProviderResponse(content="Sunny in Pune", tool_calls=[], raw_message={})

    async def stream(
        self,
        messages,
        *,
        tools=None,
        tool_calling=None,
    ):
        self.stream_called = True
        yield "Sunny in Pune"


class NoToolProvider(BaseProvider):
    async def chat(
        self,
        messages,
        *,
        tools=None,
        tool_calling=None,
    ) -> ProviderResponse:
        return ProviderResponse(content="", tool_calls=[], raw_message={})

    async def stream(
        self,
        messages,
        *,
        tools=None,
        tool_calling=None,
    ):
        yield "hello"


def test_stream_executes_tool_calls_before_final_stream() -> None:
    async def inner() -> None:
        provider = FakeToolProvider()
        agent = Agent(system_prompt="You are helpful", provider=provider, tools=[weather])

        tokens = [token async for token in agent.stream("whats the weather in pune")]

        assert "".join(tokens) == "Sunny in Pune"
        assert provider.chat_calls == 2
        assert provider.stream_called is True
        assert agent.memory.messages[-1]["content"] == "Sunny in Pune"

    asyncio.run(inner())


def test_stream_uses_provider_stream_when_no_tools_registered() -> None:
    async def inner() -> None:
        provider = NoToolProvider()
        agent = Agent(system_prompt="You are helpful", provider=provider)

        tokens = [token async for token in agent.stream("hello")]

        assert tokens == ["hello"]
        assert agent.memory.messages[-1]["content"] == "hello"

    asyncio.run(inner())


if __name__ == "__main__":
    test_stream_executes_tool_calls_before_final_stream()
    test_stream_uses_provider_stream_when_no_tools_registered()
