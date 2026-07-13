import asyncio
import uuid
from pathlib import Path
from agentapi.agent.agent import Agent
from agentapi.agent.memory import FileMemory
from agentapi.providers.base import BaseProvider, ProviderResponse

class MockProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(
            content=f"I see {len(messages)} messages in memory.",
            tool_calls=[],
            raw_message={}
        )

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield ""

async def main():
    memory_dir = Path(".test_memory")
    try:
        # 1. Generate a fixed conversation ID
        conversation_id = str(uuid.uuid4())
        
        print(f"--- Session 1: Starting new conversation {conversation_id} ---")
        memory1 = FileMemory(conversation_id=conversation_id, storage_dir=memory_dir)
        
        agent1 = Agent(
            system_prompt="You are a helpful assistant.",
            provider=MockProvider(),
            memory=memory1
        )
        
        result1 = await agent1.run("Hello there!")
        print(f"Agent 1 response: {result1}")
        print(f"Messages in file: {len(memory1.messages)}")
        
        # 2. Re-initialize a brand new Agent with the SAME conversation ID
        print(f"\n--- Session 2: Resuming conversation {conversation_id} ---")
        memory2 = FileMemory(conversation_id=conversation_id, storage_dir=memory_dir)
        
        agent2 = Agent(
            system_prompt="You are a helpful assistant.",
            provider=MockProvider(),
            memory=memory2
        )
        
        # The agent should instantly remember the previous messages loaded from the file
        result2 = await agent2.run("Do you remember me?")
        print(f"Agent 2 response: {result2}")
        print(f"Messages in file: {len(memory2.messages)}")
        
        # 3. Test Reset
        print("\n--- Resetting Memory ---")
        memory2.reset()
        print(f"Messages in file after reset: {len(memory2.messages)}")
        
    finally:
        # Cleanup
        if memory_dir.exists():
            for file in memory_dir.glob("*.json"):
                file.unlink()
            try:
                memory_dir.rmdir()
            except OSError:
                pass

if __name__ == "__main__":
    asyncio.run(main())
