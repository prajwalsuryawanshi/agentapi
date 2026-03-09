# Getting Started

This guide takes you from zero to a running API in a few minutes.

## 1. Create `main.py`

```python
from agentapi import Agent, AgentApp

app = AgentApp()

agent = Agent(
    system_prompt="You are a helpful assistant",
    provider="openai",
)


@app.chat("/chat")
async def chat(message: str):
    return await agent.run(message)


@app.chat("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
```

## 2. Run the Server

```bash
uvicorn main:app --reload
```

## 3. Test Endpoints

### Chat

```bash
curl -X POST "http://127.0.0.1:8000/chat?message=Hello"
```

### Stream

```bash
curl -N -X POST "http://127.0.0.1:8000/stream?message=Write%20a%20short%20poem"
```

## 4. Explore Interactive Docs

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 5. Use the CLI (Optional)

```bash
agentapi new myproject
cd myproject
uvicorn main:app --reload
```

The scaffold includes:

- `main.py`
- `tools.py`
- `agents.py`
- `.env`
