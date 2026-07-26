"""
examples/mongo_memory_example.py

Demonstrates MongoMemory — MongoDB-backed conversation memory for AgentAPI.

Prerequisites
─────────────
Development (MongoDB only):
    pip install "agentapi-core[mongo]"
    # Start MongoDB: docker run -d -p 27017:27017 mongo

Production (MongoDB + Redis cache):
    pip install "agentapi-core[mongo,redis]"
    # Start MongoDB: docker run -d -p 27017:27017 mongo
    # Start Redis:   docker run -d -p 6379:6379 redis

Environment variables (.env):
    OPENAI_API_KEY=your_key_here
    MONGO_URI=mongodb://localhost:27017
    REDIS_URL=redis://localhost:6379   # only needed for production example
"""

import asyncio
import os

from dotenv import load_dotenv

from agentapi import Agent, AgentApp
from agentapi.agent.memory import MongoMemory

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Development — MongoDB only (no Redis)
# ─────────────────────────────────────────────────────────────────────────────

def create_dev_memory(session_id: str) -> MongoMemory:
    """
    Development-mode memory.
    MongoDB is the only store — simple, no Redis needed.
    """
    return MongoMemory(
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        db_name=os.getenv("MONGO_DB_NAME", "agentapi"),
        collection_name=os.getenv("MONGO_COLLECTION", "conversations"),
        session_id=session_id,
        max_history=100,         # Keep last 100 messages
        use_redis_cache=False,   # ← development mode flag
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Production — MongoDB + Redis write-through cache
# ─────────────────────────────────────────────────────────────────────────────

def create_production_memory(session_id: str) -> MongoMemory:
    """
    Production-mode memory.
    MongoDB = durable source of truth.
    Redis   = fast write-through cache (TTL = 1 hour).

    Use_redis_cache=True is the production flag the maintainer asked for.
    All reads hit Redis first (microsecond latency).
    On Redis miss, falls back to MongoDB and repopulates cache.
    Redis errors fail open — agent keeps running using MongoDB.
    """
    return MongoMemory(
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        db_name=os.getenv("MONGO_DB_NAME", "agentapi"),
        collection_name=os.getenv("MONGO_COLLECTION", "conversations"),
        session_id=session_id,
        max_history=200,
        use_redis_cache=True,    # ← production mode flag
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        redis_ttl=int(os.getenv("REDIS_TTL", "3600")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# AgentApp integration — per-session memory via env flag
# ─────────────────────────────────────────────────────────────────────────────

app = AgentApp()

def get_memory(session_id: str) -> MongoMemory:
    """
    Factory that selects dev or production memory based on env flag.
    Set MONGO_USE_REDIS_CACHE=true in your .env for production.
    """
    use_redis = os.getenv("MONGO_USE_REDIS_CACHE", "false").lower() == "true"
    if use_redis:
        return create_production_memory(session_id)
    return create_dev_memory(session_id)


@app.chat("/chat")
async def chat(message: str) -> str:
    """
    Chat endpoint with per-session MongoMemory.
    Each session_id gets its own isolated conversation history in MongoDB.
    """
    # In a real app, session_id comes from the request (auth token, cookie, etc.)
    session_id = "demo_session"

    memory = get_memory(session_id)
    agent = Agent(
        system_prompt="You are a helpful assistant with persistent memory.",
        provider="openai",
        memory=memory,
    )
    return await agent.run(message)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — verifies memory works without starting AgentApp
# ─────────────────────────────────────────────────────────────────────────────

async def _test_mongo_memory() -> None:
    """Quick smoke test for MongoMemory without requiring LLM API keys."""
    print("\n── MongoMemory smoke test (dev mode) ──")
    memory = create_dev_memory("test_session")

    # Clear any leftover state from previous runs
    await memory.clear()

    # Add messages
    await memory.add({"role": "user", "content": "Hello!"})
    await memory.add({"role": "assistant", "content": "Hi there!"})
    await memory.add({"role": "user", "content": "What is AgentAPI?"})

    # Retrieve and display
    history = await memory.get()
    print(f"  Stored {len(history)} messages:")
    for msg in history:
        print(f"    [{msg['role']}] {msg['content']}")

    # Clean up
    await memory.clear()
    assert await memory.get() == [], "clear() should empty the history"
    print("  clear() ✓")

    await memory.close()
    print("── Test passed ──\n")


if __name__ == "__main__":
    asyncio.run(_test_mongo_memory())