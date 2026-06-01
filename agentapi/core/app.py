"""FastAPI wrapper with chat and stream decorators."""

from __future__ import annotations

import inspect
import asyncio
import math
import contextlib
import logging
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar
from uuid import uuid4

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
logger = logging.getLogger("agentapi")


class AgentAPI(FastAPI):
    """A small FastAPI extension with AgentAPI-focused decorators."""

    def __init__(
        self,
        *args: Any,
        sse_chunk_size: int = 64,
        sse_heartbeat_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(sse_chunk_size, bool)
            or not isinstance(sse_chunk_size, int)
            or sse_chunk_size <= 0
        ):
            raise ValueError("sse_chunk_size must be a positive integer")
        if sse_heartbeat_seconds is not None and (
            isinstance(sse_heartbeat_seconds, bool)
            or not isinstance(sse_heartbeat_seconds, (int, float))
            or not math.isfinite(sse_heartbeat_seconds)
            or sse_heartbeat_seconds <= 0
        ):
            raise ValueError(
                "sse_heartbeat_seconds must be a positive finite number when set"
            )

        self._sse_chunk_size = sse_chunk_size
        self._sse_heartbeat_seconds = sse_heartbeat_seconds

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

        html = bytes(base.body).decode("utf-8")
        inject = """
<style>
    body { background: #0b1220; }
</style>
"""
        return HTMLResponse(html.replace("</body>", f"{inject}</body>"))

    async def _swagger_ui_redirect(self) -> Response:
        return get_swagger_ui_oauth2_redirect_html()

    async def _redoc_html(self) -> Response:
        base = get_redoc_html(
            openapi_url=self.openapi_url or "/openapi.json",
            title=f"{self.title} - ReDoc",
            redoc_favicon_url=self._agentapi_favicon_path,
        )
        return HTMLResponse(bytes(base.body).decode("utf-8"))

    async def _invoke_handler(self, func: F, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _iter_token_chunks(
        self, token: str, *, chunk_size: int | None = None
    ) -> AsyncIterator[str]:
        size = chunk_size if chunk_size is not None else self._sse_chunk_size

        async def _gen() -> AsyncIterator[str]:
            if not token:
                return
            for index in range(0, len(token), size):
                yield token[index : index + size]

        return _gen()

    def _to_sse_response(self, source: AsyncIterator[str]) -> StreamingResponse:
        heartbeat_seconds = self._sse_heartbeat_seconds

        async def sse_encoder(stream: AsyncIterator[str]) -> AsyncIterator[str]:
            if not heartbeat_seconds:
                try:
                    async for token in stream:
                        async for chunk in self._iter_token_chunks(str(token)):
                            yield f"data: {chunk}\n\n"
                except AgentConfigurationError as exc:
                    yield f"event: error\ndata: {exc}\n\n"
                except AgentProviderError as exc:
                    yield f"event: error\ndata: {exc}\n\n"
                yield "data: [DONE]\n\n"
                return

            queue: asyncio.Queue = asyncio.Queue()

            async def producer() -> None:
                try:
                    async for token in stream:
                        await queue.put(("data", token))
                except (AgentConfigurationError, AgentProviderError) as exc:
                    await queue.put(("error", str(exc)))
                except Exception as exc:  # noqa: BLE001
                    await queue.put(("error", f"Internal error: {exc}"))
                finally:
                    await queue.put(("done", None))

            task = asyncio.create_task(producer())
            try:
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(
                            queue.get(), timeout=heartbeat_seconds
                        )
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue

                    if kind == "done":
                        yield "data: [DONE]\n\n"
                        return
                    if kind == "error":
                        yield f"event: error\ndata: {payload}\n\n"
                        continue

                    async for chunk in self._iter_token_chunks(str(payload)):
                        yield f"data: {chunk}\n\n"
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

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
        """Register a chat route."""

        def decorator(func: F) -> F:
            signature = inspect.signature(func)

            @wraps(func)
            async def endpoint(*args: Any, **inner_kwargs: Any) -> Any:
                request_id = str(uuid4())
                logger.info(
                    "agentapi.request.start request_id=%s path=%s handler=%s",
                    request_id,
                    path,
                    func.__name__,
                )

                try:
                    result = await self._invoke_handler(func, *args, **inner_kwargs)
                    if hasattr(result, "__aiter__"):
                        logger.info(
                            "agentapi.request.stream request_id=%s path=%s handler=%s",
                            request_id,
                            path,
                            func.__name__,
                        )
                        return self._to_sse_response(result)

                    logger.info(
                        "agentapi.request.success request_id=%s path=%s handler=%s",
                        request_id,
                        path,
                        func.__name__,
                    )
                    return result
                except AgentConfigurationError as exc:
                    logger.exception(
                        "agentapi.request.error request_id=%s path=%s handler=%s",
                        request_id,
                        path,
                        func.__name__,
                    )
                    return JSONResponse({"error": str(exc)}, status_code=500)
                except AgentProviderError as exc:
                    logger.exception(
                        "agentapi.request.error request_id=%s path=%s handler=%s",
                        request_id,
                        path,
                        func.__name__,
                    )
                    return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

            setattr(endpoint, "__signature__", signature)
            self.post(path, **kwargs)(endpoint)
            return func

        return decorator

    def stream(self, path: str, **kwargs: Any) -> Callable[[F], F]:
        """Register an SSE streaming route."""

        def decorator(func: F) -> F:
            signature = inspect.signature(func)

            @wraps(func)
            async def endpoint(*args: Any, **inner_kwargs: Any) -> Any:
                request_id = str(uuid4())
                logger.info(
                    "agentapi.request.start request_id=%s path=%s handler=%s",
                    request_id,
                    path,
                    func.__name__,
                )

                try:
                    result = await self._invoke_handler(func, *args, **inner_kwargs)
                except AgentConfigurationError as exc:
                    logger.exception(
                        "agentapi.request.error request_id=%s path=%s handler=%s",
                        request_id,
                        path,
                        func.__name__,
                    )
                    return JSONResponse({"error": str(exc)}, status_code=500)
                except AgentProviderError as exc:
                    logger.exception(
                        "agentapi.request.error request_id=%s path=%s handler=%s",
                        request_id,
                        path,
                        func.__name__,
                    )
                    return JSONResponse({"error": str(exc)}, status_code=exc.status_code)

                if not hasattr(result, "__aiter__"):
                    logger.error(
                        "agentapi.request.invalid_stream request_id=%s path=%s handler=%s",
                        request_id,
                        path,
                        func.__name__,
                    )
                    raise TypeError("@app.stream handlers must return an async iterator")

                logger.info(
                    "agentapi.request.stream request_id=%s path=%s handler=%s",
                    request_id,
                    path,
                    func.__name__,
                )
                return self._to_sse_response(result)

            setattr(endpoint, "__signature__", signature)
            self.post(path, **kwargs)(endpoint)
            return func

        return decorator