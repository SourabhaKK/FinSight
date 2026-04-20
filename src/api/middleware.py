from __future__ import annotations

import logging
import time
import traceback

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """Pure ASGI middleware — logs method, path, status_code, duration_ms."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s %d %.1fms",
                scope.get("method", ""),
                scope.get("path", ""),
                status_code,
                duration_ms,
            )


class GlobalExceptionMiddleware:
    """Pure ASGI middleware — catches unhandled exceptions and returns a JSON 500."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            logger.error(
                "Unhandled exception on %s %s:\n%s",
                scope.get("method", ""),
                scope.get("path", ""),
                traceback.format_exc(),
            )
            response = JSONResponse(
                {"error": str(exc), "status_code": 500},
                status_code=500,
            )
            await response(scope, receive, send)
