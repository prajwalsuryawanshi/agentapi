"""FastAPI wrapper with chat and stream decorators."""

from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar

from fastapi import FastAPI, Response
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import JSONResponse, StreamingResponse

from agentapi.errors import AgentConfigurationError
from agentapi.errors import AgentProviderError


F = TypeVar("F", bound=Callable[..., Any])


class AgentApp(FastAPI):
    """A small FastAPI extension with AgentAPI-focused decorators."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("title", "AgentAPI")
        kwargs.setdefault("description", "AgentAPI application")
        kwargs.setdefault("version", "0.1.0")

        docs_url = kwargs.pop("docs_url", "/docs")
        redoc_url = kwargs.pop("redoc_url", "/redoc")
        openapi_url = kwargs.pop("openapi_url", "/openapi.json")
        swagger_ui_oauth2_redirect_url = kwargs.pop(
            "swagger_ui_oauth2_redirect_url",
            "/docs/oauth2-redirect",
        )
        swagger_ui_init_oauth = kwargs.pop("swagger_ui_init_oauth", None)
        swagger_ui_parameters = kwargs.pop("swagger_ui_parameters", None)

        # Disable framework-default docs pages so we can serve AgentAPI-branded ones.
        super().__init__(
            *args,
            docs_url=None,
            redoc_url=None,
            openapi_url=openapi_url,
            **kwargs,
        )

        self._agentapi_docs_url = docs_url
        self._agentapi_redoc_url = redoc_url
        self._agentapi_swagger_ui_oauth2_redirect_url = swagger_ui_oauth2_redirect_url
        self._agentapi_swagger_ui_init_oauth = swagger_ui_init_oauth
        self._agentapi_swagger_ui_parameters = swagger_ui_parameters
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
        self._agentapi_logo_file = assets_dir / "agentapi-logo.png"
        self._agentapi_favicon_file = assets_dir / "agentapi-favicon.png"
        self._agentapi_logo_path = "/agentapi-logo.png"
        self._agentapi_favicon_path = "/agentapi-favicon.png"

        self.openapi = self._custom_openapi

        self.add_api_route(
            self._agentapi_logo_path,
            self._logo,
            methods=["GET"],
            include_in_schema=False,
        )

        self.add_api_route(
            self._agentapi_favicon_path,
            self._favicon,
            methods=["GET"],
            include_in_schema=False,
        )

        if self._agentapi_docs_url:
            self.add_api_route(
                self._agentapi_docs_url,
                self._swagger_ui_html,
                methods=["GET"],
                include_in_schema=False,
            )

            if self._agentapi_swagger_ui_oauth2_redirect_url:
                self.add_api_route(
                    self._agentapi_swagger_ui_oauth2_redirect_url,
                    self._swagger_ui_redirect,
                    methods=["GET"],
                    include_in_schema=False,
                )

        if self._agentapi_redoc_url:
            self.add_api_route(
                self._agentapi_redoc_url,
                self._redoc_html,
                methods=["GET"],
                include_in_schema=False,
            )

    def _custom_openapi(self) -> dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema

        schema = get_openapi(
            title=self.title,
            version=self.version,
            description=self.description,
            routes=self.routes,
        )
        schema.setdefault("info", {})["x-logo"] = {
            "url": self._agentapi_logo_path,
            "altText": "AgentAPI",
        }
        self.openapi_schema = schema
        return schema

    async def _logo(self) -> Response:
        return FileResponse(self._agentapi_logo_file)

    async def _favicon(self) -> Response:
        return FileResponse(self._agentapi_favicon_file)

    async def _swagger_ui_html(self) -> Response:
        base = get_swagger_ui_html(
            openapi_url=self.openapi_url or "/openapi.json",
            title=f"{self.title} - Docs",
            oauth2_redirect_url=self._agentapi_swagger_ui_oauth2_redirect_url,
            init_oauth=self._agentapi_swagger_ui_init_oauth,
            swagger_ui_parameters=self._agentapi_swagger_ui_parameters,
            swagger_favicon_url=self._agentapi_favicon_path,
        )

        html = base.body.decode("utf-8")
        inject = f"""
<script>
window.addEventListener('load', function () {{
  var topbarLogo = document.querySelector('.topbar-wrapper .link img');
  if (topbarLogo) {{
    topbarLogo.src = '{self._agentapi_logo_path}';
    topbarLogo.alt = 'AgentAPI';
    topbarLogo.style.height = '36px';
    topbarLogo.style.width = 'auto';
  }}
}});
</script>
"""
        return HTMLResponse(html.replace("</body>", f"{inject}</body>"))

    async def _swagger_ui_redirect(self) -> Response:
        return get_swagger_ui_oauth2_redirect_html()

    async def _redoc_html(self) -> Response:
        return get_redoc_html(
            openapi_url=self.openapi_url or "/openapi.json",
            title=f"{self.title} - ReDoc",
            redoc_favicon_url=self._agentapi_favicon_path,
        )

    async def _invoke_handler(self, func: F, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _iter_token_chunks(self, token: str, *, chunk_size: int = 64) -> AsyncIterator[str]:
        # Providers may emit large text fragments; split them to keep downstream
        # streaming UX incremental.
        async def _gen() -> AsyncIterator[str]:
            if not token:
                return
            for index in range(0, len(token), chunk_size):
                yield token[index : index + chunk_size]

        return _gen()

    def _to_sse_response(self, source: AsyncIterator[str]) -> StreamingResponse:
        async def sse_encoder(stream: AsyncIterator[str]) -> AsyncIterator[str]:
            try:
                async for token in stream:
                    async for chunk in self._iter_token_chunks(str(token)):
                        yield f"data: {chunk}\\n\\n"
            except AgentConfigurationError as exc:
                # Surface runtime config issues as an SSE error event.
                yield f"event: error\\ndata: {exc}\\n\\n"
            except AgentProviderError as exc:
                yield f"event: error\\ndata: {exc}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(
            sse_encoder(source),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def chat(self, path: str, **kwargs: Any) -> Callable[[F], F]:
        """Register a chat route.

        If the handler returns an async iterator, AgentApp automatically responds
        as SSE (`text/event-stream`). Otherwise, it returns regular JSON.
        """

        def decorator(func: F) -> F:
            signature = inspect.signature(func)

            @wraps(func)
            async def endpoint(*args: Any, **inner_kwargs: Any) -> Any:
                try:
                    result = await self._invoke_handler(func, *args, **inner_kwargs)
                    if hasattr(result, "__aiter__"):
                        return self._to_sse_response(result)
                    return result
                except AgentConfigurationError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=500)
                except AgentProviderError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

            setattr(endpoint, "__signature__", signature)

            self.post(path, **kwargs)(endpoint)
            return func

        return decorator

    def stream(self, path: str, **kwargs: Any) -> Callable[[F], F]:
        """Register an SSE streaming route.

        Backward-compatible alias for explicit streaming-only endpoints.
        New code can use `@app.chat` and simply return an async iterator.
        """

        def decorator(func: F) -> F:
            signature = inspect.signature(func)

            @wraps(func)
            async def endpoint(*args: Any, **inner_kwargs: Any) -> Any:
                try:
                    result = await self._invoke_handler(func, *args, **inner_kwargs)
                except AgentConfigurationError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=500)
                except AgentProviderError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

                if not hasattr(result, "__aiter__"):
                    raise TypeError("@app.stream handlers must return an async iterator")

                return self._to_sse_response(result)

            setattr(endpoint, "__signature__", signature)

            self.post(path, **kwargs)(endpoint)
            return func

        return decorator
