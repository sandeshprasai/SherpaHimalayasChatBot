# src/api/middleware.py
"""
Custom middleware:
  1. APIKeyMiddleware   — enforces X-API-Key header on /chat routes
  2. RequestLogMiddleware — structured request/response logging
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.config.settings import settings
from src.services.logger import logger


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Checks the X-API-Key header for all routes under /chat.
    Whitelists /health and / (root) so monitoring tools work without a key.
    """
    PROTECTED_PREFIXES = ("/chat",)
    WHITELIST = ("/", "/health", "/health/", "/docs", "/openapi.json")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for whitelisted paths
        if any(path.startswith(w) for w in self.WHITELIST) or \
           not any(path.startswith(p) for p in self.PROTECTED_PREFIXES):
            return await call_next(request)

        # Validate API key
        api_key = request.headers.get("X-API-Key", "")
        if api_key != settings.API_KEY:
            logger.warning(f"Unauthorized request to {path} from {request.client.host}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. "
                                   "Provide it in the X-API-Key header."}
            )

        return await call_next(request)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing info. Non-blocking.
    """
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        latency_ms = round((time.time() - start) * 1000)
        logger.info(
            f"{request.method} {request.url.path} "
            f"| status={response.status_code} | {latency_ms}ms "
            f"| ip={request.client.host if request.client else 'unknown'}"
        )
        return response