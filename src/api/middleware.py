import hmac
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.config.settings import settings
from src.services.logger import logger


class APIKeyMiddleware(BaseHTTPMiddleware):
    
    PROTECTED_PREFIXES = ("/chat", "/v1/chat", "/session", "/v1/session")
    
    WHITELIST_EXACT = {"/", "/health", "/health/", "/v1/health", "/v1/health/"}
    WHITELIST_PREFIX = ("/docs", "/openapi.json")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self.WHITELIST_EXACT:
            return await call_next(request)

        if any(path.startswith(p) for p in self.WHITELIST_PREFIX):
            return await call_next(request)


        is_protected = any(path.startswith(p) for p in self.PROTECTED_PREFIXES)

        if not is_protected:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(api_key, settings.API_KEY):
            logger.warning(f"Unauthorized request to {path} from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Provide it in the X-API-Key header."}
            )

        return await call_next(request)


class RequestLogMiddleware(BaseHTTPMiddleware):
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