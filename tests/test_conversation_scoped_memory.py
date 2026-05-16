"""Tests for conversation-scoped memory isolation on shared Agent instances."""

from __future__ import annotations

import asyncio

from agentapi import Agent, InMemoryMemory, RedisMemory
from agentapi.providers.base import BaseProvider, ProviderResponse


class EchoProvider(BaseProvider):
    """Provider that echoes the visible non-system history for assertions."""

    async def chat(self, messages, *, tools=None, tool_calling=None):
        user_message = messages[-1]["content"]
        history = [message["content"] for message in messages if message["role"] != "system"]

        if user_message == "slow-a":
            await asyncio.sleep(0.05)
        elif user_message == "fast-b":
            await asyncio.sleep(0.0)

        return ProviderResponse(
            content=f"reply({user_message}) seen={history}",
            tool_calls=[],
            raw_message={},
        )

    async def stream(self, messages, *, tools=None, tool_calling=None):
        user_message = messages[-1]["content"]
        history = [message["content"] for message in messages if message["role"] != "system"]
        payload = f"stream({user_message}) seen={history}"
        midpoint = len(payload) // 2
        yield payload[:midpoint]
        await asyncio.sleep(0)
        yield payload[midpoint:]


class FakeRedisClient:
    def __init__(self):
        self.closed = 0
        self._lists: dict[str, list[str]] = {}
        self._hashes: dict[str, dict[str, str]] = {}

    def exists(self, key: str) -> bool:
        return key in self._hashes

    def hset(self, key: str, mapping):
        self._hashes[key] = dict(mapping)

    def expire(self, key: str, ttl: int):
        return True

    def lrange(self, key: str, start: int, end: int):
        values = self._lists.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    def rpush(self, key: str, value: str):
        self._lists.setdefault(key, []).append(value)

    def delete(self, key: str):
        self._lists.pop(key, None)

    def close(self):
        self.closed += 1


def test_run_preserves_backward_compatible_behavior_without_conversation_id():
    agent = Agent(system_prompt="sys", provider=EchoProvider())

    first = asyncio.run(agent.run("hello"))
    second = asyncio.run(agent.run("again"))

    assert first == "reply(hello) seen=['hello']"
    assert second == "reply(again) seen=['hello', \"reply(hello) seen=['hello']\", 'again']"
    assert [message["content"] for message in agent.memory.messages] == [
        "hello",
        "reply(hello) seen=['hello']",
        "again",
        "reply(again) seen=['hello', \"reply(hello) seen=['hello']\", 'again']",
    ]


def test_run_isolates_different_conversation_ids_on_shared_agent():
    agent = Agent(system_prompt="sys", provider=EchoProvider(), memory=InMemoryMemory())

    first = asyncio.run(agent.run("hello-a", conversation_id="550e8400-e29b-41d4-a716-446655440000"))
    second = asyncio.run(agent.run("hello-b", conversation_id="550e8400-e29b-41d4-a716-446655440001"))

    assert first == "reply(hello-a) seen=['hello-a']"
    assert second == "reply(hello-b) seen=['hello-b']"

    memory_a = agent.memory.for_conversation("550e8400-e29b-41d4-a716-446655440000")
    memory_b = agent.memory.for_conversation("550e8400-e29b-41d4-a716-446655440001")

    assert [message["content"] for message in memory_a.messages] == [
        "hello-a",
        "reply(hello-a) seen=['hello-a']",
    ]
    assert [message["content"] for message in memory_b.messages] == [
        "hello-b",
        "reply(hello-b) seen=['hello-b']",
    ]


def test_run_preserves_continuity_with_same_conversation_id():
    agent = Agent(system_prompt="sys", provider=EchoProvider(), memory=InMemoryMemory())
    conversation_id = "550e8400-e29b-41d4-a716-446655440010"

    first = asyncio.run(agent.run("first", conversation_id=conversation_id))
    second = asyncio.run(agent.run("second", conversation_id=conversation_id))

    assert first == "reply(first) seen=['first']"
    assert second == "reply(second) seen=['first', \"reply(first) seen=['first']\", 'second']"


def test_concurrent_run_calls_with_different_conversation_ids_stay_isolated():
    async def scenario():
        agent = Agent(system_prompt="sys", provider=EchoProvider(), memory=InMemoryMemory())

        first_id = "550e8400-e29b-41d4-a716-446655440020"
        second_id = "550e8400-e29b-41d4-a716-446655440021"

        results = await asyncio.gather(
            agent.run("slow-a", conversation_id=first_id),
            agent.run("fast-b", conversation_id=second_id),
        )

        memory_a = agent.memory.for_conversation(first_id)
        memory_b = agent.memory.for_conversation(second_id)

        return results, memory_a.messages, memory_b.messages

    results, messages_a, messages_b = asyncio.run(scenario())

    assert results == [
        "reply(slow-a) seen=['slow-a']",
        "reply(fast-b) seen=['fast-b']",
    ]
    assert [message["content"] for message in messages_a] == [
        "slow-a",
        "reply(slow-a) seen=['slow-a']",
    ]
    assert [message["content"] for message in messages_b] == [
        "fast-b",
        "reply(fast-b) seen=['fast-b']",
    ]


def test_stream_writes_only_to_requested_conversation():
    async def scenario():
        agent = Agent(system_prompt="sys", provider=EchoProvider(), memory=InMemoryMemory())
        conversation_id = "550e8400-e29b-41d4-a716-446655440030"

        collected: list[str] = []
        async for token in agent.stream("stream-me", conversation_id=conversation_id):
            collected.append(token)

        resolved_memory = agent.memory.for_conversation(conversation_id)
        return "".join(collected), resolved_memory.messages

    streamed_text, messages = asyncio.run(scenario())

    assert streamed_text == "stream(stream-me) seen=['stream-me']"
    assert [message["content"] for message in messages] == [
        "stream-me",
        "stream(stream-me) seen=['stream-me']",
    ]


def test_redis_sibling_views_do_not_close_shared_client():
    client = FakeRedisClient()
    base = RedisMemory(
        redis_url="redis://localhost:6379",
        conversation_id="550e8400-e29b-41d4-a716-446655440100",
        _redis_client=client,
    )
    sibling = base.for_conversation("550e8400-e29b-41d4-a716-446655440101")

    base.close()
    sibling.close()

    assert client.closed == 0


def test_redis_owner_closes_client_once():
    client = FakeRedisClient()
    memory = RedisMemory(
        redis_url="redis://localhost:6379",
        conversation_id="550e8400-e29b-41d4-a716-446655440102",
        _redis_client=client,
    )
    memory._owns_redis_client = True

    memory.close()

    assert client.closed == 1
