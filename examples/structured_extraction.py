import asyncio
from pydantic import BaseModel
from agentapi.agent.agent import Agent
from agentapi.providers.base import BaseProvider, ProviderResponse

class UserSchema(BaseModel):
    name: str
    age: int
    is_active: bool

class MockProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        # On the first call, we simulate a bad JSON response
        if len(messages) == 2: # system + user
            return ProviderResponse(
                content='{"name": "Alice", "age": "twenty"}', # Age is a string instead of int
                tool_calls=[],
                raw_message={}
            )
        # On the retry, we simulate a corrected JSON response
        return ProviderResponse(
            content='{"name": "Alice", "age": 20, "is_active": true}',
            tool_calls=[],
            raw_message={}
        )

    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield ""

async def main():
    agent = Agent(system_prompt="You are a helpful assistant.", provider=MockProvider())
    
    print("Running with Pydantic output_schema validation...")
    result = await agent.run(
        "Extract the user details", 
        output_schema=UserSchema,
        max_validation_retries=1
    )
    
    print(f"Result type: {type(result)}")
    print(f"Result data: {result}")

if __name__ == "__main__":
    asyncio.run(main())
