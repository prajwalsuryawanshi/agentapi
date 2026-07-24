"""Test file-based conversation memory backend (FileMemory)."""

import os
import json
import pytest
import shutil
from agentapi import FileMemory, create_conversation_id


@pytest.fixture
def temp_storage():
    """Fixture to create and clean up a temporary storage directory."""
    dir_name = "./temp_test_memory_store"
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    yield dir_name
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def test_auto_generated_conversation_id(temp_storage):
    """FileMemory auto-generates conversation_id if not provided."""
    mem = FileMemory(storage_dir=temp_storage)
    assert mem.conversation_id is not None
    assert len(mem.conversation_id) == 36  # UUID hex string length
    assert os.path.exists(temp_storage)


def test_explicit_conversation_id(temp_storage):
    """FileMemory accepts explicit conversation_id."""
    conv_id = create_conversation_id()
    mem = FileMemory(conversation_id=conv_id, storage_dir=temp_storage)
    assert mem.conversation_id == conv_id
    assert mem.file_path.endswith(f"{conv_id}.json")


def test_conversation_persistence_and_isolation(temp_storage):
    """Different conversation_ids maintain separate persistent message histories on disk."""
    conv_id_1 = create_conversation_id()
    conv_id_2 = create_conversation_id()

    # Create separate FileMemory instances
    mem1 = FileMemory(conversation_id=conv_id_1, storage_dir=temp_storage)
    mem2 = FileMemory(conversation_id=conv_id_2, storage_dir=temp_storage)

    # Initially histories should be empty
    assert len(mem1.messages) == 0
    assert len(mem2.messages) == 0

    # Add messages
    msg1 = {"role": "user", "content": "Hello from conversation 1"}
    msg2 = {"role": "user", "content": "Hello from conversation 2"}

    mem1.add(msg1)
    mem2.add(msg2)

    # Verify that files are created on disk
    assert os.path.exists(mem1.file_path)
    assert os.path.exists(mem2.file_path)

    # Verify message retention and isolation
    assert len(mem1.messages) == 1
    assert len(mem2.messages) == 1
    assert mem1.messages[0]["content"] == "Hello from conversation 1"
    assert mem2.messages[0]["content"] == "Hello from conversation 2"


def test_reset_deletes_file(temp_storage):
    """Resetting the memory clears the messages and deletes the JSON file from disk."""
    conv_id = create_conversation_id()
    mem = FileMemory(conversation_id=conv_id, storage_dir=temp_storage)

    mem.add({"role": "user", "content": "Keep this secure"})
    assert os.path.exists(mem.file_path)
    assert len(mem.messages) == 1

    # Reset memory
    mem.reset()

    # File should be removed, messages empty
    assert not os.path.exists(mem.file_path)
    assert len(mem.messages) == 0


def test_invalid_json_handling(temp_storage):
    """FileMemory handles corrupted or invalid JSON files gracefully."""
    conv_id = create_conversation_id()
    mem = FileMemory(conversation_id=conv_id, storage_dir=temp_storage)

    # Write invalid content directly to the file path
    with open(mem.file_path, "w", encoding="utf-8") as f:
        f.write("{invalid_json:")

    # Reading should not crash, should return empty list
    assert mem.messages == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
