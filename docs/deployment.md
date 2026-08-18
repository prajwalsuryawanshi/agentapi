# Deployment

This page explains how to deploy applications built with AgentAPI.

## Pre-Deployment Checklist

Before shipping, verify the following:

- Production API keys are set as environment variables.
- `provider` and model settings are explicit in your app config.
- Health checks and logs are enabled.
- CORS is configured only for allowed origins.
- You use `agent.run(...)` for tool-critical correctness paths.
- You use `agent.stream(...)` only for low-latency streaming endpoints.

## Environment Variables

Set only the provider keys you need for your runtime:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
DEFAULT_PROVIDER=openai
PORT=8000
```

For production, store secrets in your platform secret manager, not in `.env` files committed to source control.

## Run with Uvicorn (VM or Container)

If your entry file is `main.py` and app object is `app`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Recommended for production:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

Choose worker count based on CPU and request patterns. Start small and measure latency.

## Docker Deployment

Example `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -U pip && \
	pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t my-agentapi-app .
docker run -p 8000:8000 --env-file .env my-agentapi-app
```

## Reverse Proxy (Nginx / Ingress)

If you serve streaming endpoints, keep SSE-friendly proxy settings:

- Disable response buffering on stream routes.
- Keep HTTP connection alive.
- Increase read timeout for long responses.

Without these, streamed responses may appear delayed or in one large chunk.

## Platform Deployments

AgentAPI apps work on common Python platforms:

- Render, Railway, Fly.io, DigitalOcean App Platform
- AWS ECS/Fargate, Azure Container Apps, GCP Cloud Run
- Kubernetes clusters with an Ingress controller

Use 

```python
uvicorn main:app --host 0.0.0.0 --port $PORT
```
as your startup command.

## Health and Observability

Recommended additions for production apps:

- A lightweight health endpoint, for example `/health`.
- Structured request/error logs.
- Metrics for request count, latency, and provider failures.

This helps isolate issues such as provider key errors, model errors, and stream interruptions.

## Reliability Tips

- Set request timeouts for upstream provider calls.
- Guard tool functions with exception handling.
- Limit tool rounds with `max_tool_rounds` in `agent.run(...)`.
- Keep tool output stable and parseable.

## Multi-User Apps

Do not share one `Agent` memory across all users.

- Use one `Agent` per user/session.
- Or reset memory between unrelated requests.

This avoids context leakage across users.

## Architecture Reference

See [Feature and deployment scalability](architecture/scalability.md) for state
ownership, store reuse, multi-worker deployments, conversation concurrency,
streaming backpressure, and external multi-agent orchestration.

