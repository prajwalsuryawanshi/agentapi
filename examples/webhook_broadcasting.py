import asyncio
import sys
from agentapi.agent.agent import Agent
from agentapi.agent.webhooks import WebhookHook
from agentapi.providers.base import BaseProvider, ProviderResponse, ToolCall

import os
import hmac
import hashlib

async def handle_webhook(request):
    event = request.headers.get("X-Agent-Event")
    signature = request.headers.get("X-Agent-Signature")
    timestamp = request.headers.get("X-Agent-Timestamp")
    
    payload_bytes = await request.read()
    payload = await request.json()
    
    secret_token = os.environ.get("WEBHOOK_SECRET_TOKEN", "fallback-secret-key").encode("utf-8")
    
    if not signature or not timestamp:
        return web.Response(text="Missing signature or timestamp", status=401)
        
    signed_content = f"{timestamp}:".encode("utf-8") + payload_bytes
    expected_sig = "sha256=" + hmac.new(secret_token, signed_content, hashlib.sha256).hexdigest()
    
    if not hmac.compare_digest(signature, expected_sig):
        return web.Response(text="Invalid signature", status=401)

    print(f"\n[SERVER RECEIVER] Received Webhook Event: {event}")
    print(f"[SERVER RECEIVER] Signature Verified: {signature}")
    print(f"[SERVER RECEIVER] Payload: {payload}")
    return web.Response(text="OK")

class MockProvider(BaseProvider):
    async def chat(self, messages, *, tools=None, tool_calling=None):
        return ProviderResponse(
            content="Webhook dispatched!",
            tool_calls=[ToolCall(id="call_1", name="dummy_tool", arguments='{}')],
            raw_message={}
        )
    async def stream(self, messages, *, tools=None, tool_calling=None):
        yield ""

def dummy_tool():
    return "Tool success"

async def main():
    # 1. Start the mock webhook receiver server
    app = web.Application()
    app.router.add_post("/webhook", handle_webhook)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    print("Mock webhook server running on http://localhost:8080/webhook")

    # 2. Configure the Agent with the WebhookHook
    secret_token = os.environ.get("WEBHOOK_SECRET_TOKEN", "fallback-secret-key")
    webhook_hook = WebhookHook(
        endpoint_url="http://localhost:8080/webhook",
        secret_token=secret_token
    )
    
    agent = Agent(
        system_prompt="You are an agent.",
        provider=MockProvider(),
        tools=[dummy_tool],
        hooks=[webhook_hook]
    )

    # 3. Run the agent and trigger the webhooks
    print("\n--- Running Agent ---")
    await agent.run("Trigger webhooks!")
    
    # 4. Cleanup
    await asyncio.sleep(1) # Give server time to process last webhook
    await runner.cleanup()

if __name__ == "__main__":
    try:
        import httpx
        from aiohttp import web
    except ImportError:
        print("Please install httpx and aiohttp: pip install httpx aiohttp")
        sys.exit(1)
        
    asyncio.run(main())
