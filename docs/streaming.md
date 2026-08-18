# Streaming

AgentAPI supports token streaming using SSE.

## Choose the Right Path

- Use `await agent.run(...)` when you need final answers with tool rounds.
- Use `agent.stream(...)` when you need chunked token delivery.

## Automatic Streaming with `@app.chat`

If a handler returns an async iterator, AgentAPI automatically returns `text/event-stream`.

```python
@app.chat("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
```

## SSE Behavior

- Emits incremental `data:` chunks
- Sends `data: [DONE]` at completion
- Includes anti-buffering headers for reverse proxies

## Tool Calling and Streaming

For tool-heavy flows, prefer `run` when correctness is more important than token latency.
For live typing UX, use `stream`.

## Error Behavior in Streams

During streaming, runtime errors are emitted as SSE error events:

- config errors (missing keys)
- upstream provider failures

## Explicit Streaming Route

`@app.stream` still exists as a compatibility alias when you want strict streaming-only semantics.

```python
@app.stream("/stream")
async def stream_chat(message: str):
    return agent.stream(message)
```

## Quick Test

```bash
curl -N -X POST "http://127.0.0.1:8000/stream?message=hello"
```

## Architecture Reference

AgentAPI has separate Agent-owned and application-adapted SSE paths. Contributors
should read [Streaming has two adaptation levels](architecture/runtime.md#streaming-has-two-adaptation-levels),
[Streaming scalability](architecture/scalability.md#streaming-scalability), and
the [streaming change checklist](architecture/contributing.md#changing-streaming-or-decorators).
