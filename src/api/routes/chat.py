import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Request, Cookie
from src.api.schemas.chat import ChatRequest, ChatResponse, SourceDocument
from src.api.dependencies import get_pipeline, limiter
from src.core.pipeline import Pipeline
from src.services.logger import logger, fallback_logger

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send a message to the Sherpa Chatbot"
)
@limiter.limit("20/minute")   
async def chat(      
    request: Request,
    body: ChatRequest,
    # FastAPI automatically looks for a cookie named "session_id"
    session_id: str = Cookie(default=None),
    pipeline: Pipeline = Depends(get_pipeline)
):
    start = time.time()

    # --- STRICT COOKIE VALIDATION ---
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session cookie is missing. Please ensure your browser sends the session_id cookie."
        )

    try:
        val_uuid = uuid.UUID(session_id, version=4)
        safe_session_id = str(val_uuid)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid session_id cookie format. Must be a valid UUIDv4 string."
        )

    try:
        result = await pipeline.run(body.message, session_id=safe_session_id)
    except Exception as e:
        logger.error("pipeline_error", extra={
            "session_id": safe_session_id,
            "query": body.message[:80],
            "error": str(e),
        }, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request."
        )

    latency_ms = round((time.time() - start) * 1000)

    sources = []
    if result.sources:
        for s in result.sources:
            sources.append(SourceDocument(
                title=s.title or "Unknown",
                source=s.source_id or ""
            ))

    intent_str = result.intent.value if hasattr(result.intent, 'value') else str(result.intent)
    strategy_str = result.strategy.value if hasattr(result.strategy, 'value') else str(result.strategy)

    log_fields = {
        "session_id": safe_session_id,
        "query": body.message.strip(),
        "intent": intent_str,
        "strategy": strategy_str,
        "tokens": result.tokens_used,
        "latency_ms": latency_ms,
        "fallback": result.is_fallback,
        "source_count": len(sources),
    }

    # Log exactly once per request, to whichever logger fits.
    if result.is_fallback:
        fallback_logger.warning("chat_completed", extra=log_fields)
    else:
        logger.info("chat_completed", extra=log_fields)

    return ChatResponse(
        success=True,
        session_id=safe_session_id,
        answer=result.answer,
        sections=result.sections,
        suggestions=result.suggestions,
        sources=sources,
        latency_ms=latency_ms,
        is_fallback=result.is_fallback
    )