from agentapi.core.app import AgentAPI


def test_stream_endpoint_openapi_media_type():
    app = AgentAPI()

    @app.stream("/stream")
    async def stream(message: str):
        async def gen():
            yield "hello"

        return gen()

    schema = app.openapi()

    content = schema["paths"]["/stream"]["post"]["responses"]["200"]["content"]

    assert "text/event-stream" in content