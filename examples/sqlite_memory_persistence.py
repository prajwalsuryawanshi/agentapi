import asyncio
import uuid
from pathlib import Path
from agentapi.agent.agent import Agent
from agentapi.agent.memory import SqliteMemory
from agentapi.providers.base import BaseProvider, ProviderResponse

class MockProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(
            content=f"I see {len(messages)} messages in the SQLite DB.",
            tool_calls=[],
            raw_message={}
        )

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield ""

async def main():
    db_path = Path(".test_memory/test.db")
    memory1 = None
    memory2 = None
    try:
        # 1. Generate a fixed conversation ID
        conversation_id = str(uuid.uuid4())
        
        print(f"--- Session 1: Starting new conversation {conversation_id} ---")
        memory1 = SqliteMemory(conversation_id=conversation_id, db_path=db_path)
        
        agent1 = Agent(
            system_prompt="You are a helpful assistant.",
            provider=MockProvider(),
            memory=memory1
        )
        
        result1 = await agent1.run("Hello SQLite!")
        print(f"Agent 1 response: {result1}")
        print(f"Messages in DB: {len(memory1.messages)}")
        memory1.close()
        
        # 2. Re-initialize a brand new Agent with the SAME conversation ID
        print(f"\n--- Session 2: Resuming conversation {conversation_id} ---")
        memory2 = SqliteMemory(conversation_id=conversation_id, db_path=db_path)
        
        agent2 = Agent(
            system_prompt="You are a helpful assistant.",
            provider=MockProvider(),
            memory=memory2
        )
        
        result2 = await agent2.run("Do you remember me from the DB?")
        print(f"Agent 2 response: {result2}")
        print(f"Messages in DB: {len(memory2.messages)}")
        
        # 3. Test Reset
        print("\n--- Resetting Memory ---")
        memory2.reset()
        print(f"Messages in DB after reset: {len(memory2.messages)}")
    finally:
        if memory1:
            memory1.close()
        if memory2:
            memory2.close()
            
        # Cleanup
        if db_path.exists():
            db_path.unlink()
        if db_path.parent.exists():
            try:
                db_path.parent.rmdir()
            except OSError:
                pass

if __name__ == "__main__":
    asyncio.run(main())
