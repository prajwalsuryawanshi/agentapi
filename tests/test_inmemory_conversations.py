"""Test UUID-based conversation isolation in InMemoryMemory."""

import pytest
from agentapi import InMemoryMemory, create_conversation_id


def test_auto_generated_conversation_id():
    """InMemoryMemory auto-generates conversation_id if not provided."""
    mem = InMemoryMemory()
    assert mem.conversation_id is not None
    assert len(mem.conversation_id) == 36  # UUID hex string length


def test_explicit_conversation_id():
    """InMemoryMemory accepts explicit conversation_id."""
    conv_id = create_conversation_id()
    mem = InMemoryMemory(conversation_id=conv_id)
    assert mem.conversation_id == conv_id


def test_conversation_isolation():
    """Different conversation_ids maintain separate message histories."""
    conv_id_1 = create_conversation_id()
    conv_id_2 = create_conversation_id()

    # Create two separate memory instances with different conversation IDs
    mem1 = InMemoryMemory(conversation_id=conv_id_1)
    mem2 = InMemoryMemory(conversation_id=conv_id_2)

    # Add messages to each conversation
    mem1.add({"role": "user", "content": "Hello from conversation 1"})
    mem2.add({"role": "user", "content": "Hello from conversation 2"})

    # Verify isolation: each conversation only sees its own messages
    assert len(mem1.messages) == 1
    assert len(mem2.messages) == 1
    assert mem1.messages[0]["content"] == "Hello from conversation 1"
    assert mem2.messages[0]["content"] == "Hello from conversation 2"


def test_multiple_agents_same_conversation():
    """Separate InMemoryMemory instances do not share state, even with the same conversation_id."""
    shared_conv_id = create_conversation_id()

    # Two memory instances using the SAME conversation_id remain isolated.
    mem_a = InMemoryMemory(conversation_id=shared_conv_id)
    mem_b = InMemoryMemory(conversation_id=shared_conv_id)

    # Add message via mem_a
    mem_a.add({"role": "user", "content": "Message from A"})

    assert len(mem_a.messages) == 1
    assert len(mem_b.messages) == 0
    assert mem_a.conversation_id == mem_b.conversation_id
    assert mem_a.conversation_id == shared_conv_id


def test_reset_preserves_isolation():
    """Resetting one conversation doesn't affect others."""
    conv_id_1 = create_conversation_id()
    conv_id_2 = create_conversation_id()

    mem1 = InMemoryMemory(conversation_id=conv_id_1)
    mem2 = InMemoryMemory(conversation_id=conv_id_2)

    mem1.add({"role": "user", "content": "Msg 1"})
    mem2.add({"role": "user", "content": "Msg 2"})

    # Reset conversation 1
    mem1.reset()

    # Conversation 1 should be cleared, 2 unaffected
    assert len(mem1.messages) == 0
    assert len(mem2.messages) == 1
    assert mem2.messages[0]["content"] == "Msg 2"


def test_system_prompt_per_conversation():
    """Messages remain isolated per conversation."""
    conv_id_1 = create_conversation_id()
    conv_id_2 = create_conversation_id()

    mem1 = InMemoryMemory(conversation_id=conv_id_1)
    mem2 = InMemoryMemory(conversation_id=conv_id_2)

    mem1.add({"role": "user", "content": "You are helpful"})
    mem2.add({"role": "user", "content": "You are strict"})

    assert mem1.messages[0]["content"] == "You are helpful"
    assert mem2.messages[0]["content"] == "You are strict"


def test_invalid_uuid_raises_error():
    """Invalid conversation_id format raises ValueError."""
    with pytest.raises(ValueError):
        InMemoryMemory(conversation_id="not-a-valid-uuid")


def test_ttl_disabled_keeps_messages():
    """Messages remain available indefinitely when no TTL is configured."""
    now = [100.0]
    mem = InMemoryMemory(conversation_ttl_seconds=None, time_fn=lambda: now[0])

    mem.add({"role": "user", "content": "keep me"})
    now[0] += 10_000

    assert mem.messages == [{"role": "user", "content": "keep me"}]


def test_expired_conversation_is_cleared_on_access():
    """Expired in-memory conversations are cleaned up when retrieved."""
    now = [100.0]
    mem = InMemoryMemory(conversation_ttl_seconds=30, time_fn=lambda: now[0])

    mem.add({"role": "user", "content": "temporary"})
    now[0] += 30

    assert mem.messages == []


def test_add_after_expiry_starts_fresh_history():
    """Adding a message after expiry discards stale messages first."""
    now = [100.0]
    mem = InMemoryMemory(conversation_ttl_seconds=10, time_fn=lambda: now[0])

    mem.add({"role": "user", "content": "old"})
    now[0] += 11
    mem.add({"role": "user", "content": "new"})

    assert mem.messages == [{"role": "user", "content": "new"}]


def test_ttl_refreshes_on_activity_before_expiry():
    """Active conversations stay alive until they are idle past the TTL."""
    now = [100.0]
    mem = InMemoryMemory(conversation_ttl_seconds=10, time_fn=lambda: now[0])

    mem.add({"role": "user", "content": "still active"})
    now[0] += 9
    assert len(mem.messages) == 1

    now[0] += 9
    assert mem.messages == [{"role": "user", "content": "still active"}]

    now[0] += 10
    assert mem.messages == []


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), True, "60"])
def test_invalid_ttl_raises_error(ttl):
    """Invalid TTL values fail fast."""
    with pytest.raises(ValueError, match="conversation_ttl_seconds"):
        InMemoryMemory(conversation_ttl_seconds=ttl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
