import uuid

from fastapi import APIRouter, Request, Response, status

from src.api.dependencies import limiter
from src.config.settings import settings
from src.services.logger import logger

router = APIRouter(prefix="/session", tags=["Session"])

# Keep this in sync with the TTL used in src/services/memory.py (setex(..., 3600, ...))
# so the cookie never outlives the Redis-backed conversation history it points at.
SESSION_COOKIE_MAX_AGE = 3600


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Issue a new chat session and set the session_id cookie",
)
@limiter.limit("10/minute")
async def create_session(request: Request, response: Response):
    session_id = str(uuid.uuid4())

    is_prod = settings.ENVIRONMENT == "production"
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        secure=is_prod,
        samesite="none" if is_prod else "lax",
        path="/",
    )

    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Issued new session | session={session_id} | ip={client_ip}")

    return {"session_id": session_id, "expires_in": SESSION_COOKIE_MAX_AGE}