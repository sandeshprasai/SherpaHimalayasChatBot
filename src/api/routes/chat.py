import time
from fastapi import APIRouter, Depends, HTTPException, status ,Request
from fastapi.concurrency import run_in_threadpool
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
    summary="Send a message to the Sherpa Chatbot",
    description=(
        "Accepts a natural language query about trekking, peaks, expeditions, "
        "pricing, or general travel to Nepal/Tibet/Bhutan. "
        "Returns an AI-generated answer with optional source references."
    )
)
@limiter.limit("20/minute")   
async def chat(      
    request: Request,
    body: ChatRequest,
    pipeline: Pipeline = Depends(get_pipeline)
):
    
    start = time.time()

    try:
        result = await pipeline.run(body.message)
    except Exception as e:
        logger.error(f"Pipeline error for query='{body.message[:80]}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request."
        )

    latency_ms = round((time.time() - start) * 1000)

    if result.is_fallback:
        fallback_logger.warning(
            f"FALLBACK | query='{body.message}' | latency={latency_ms}ms"
        )

    # ── Build response ─────────────────────────────────────────────────────
    sources = []
    if result.sources:
        for s in result.sources:
            sources.append(SourceDocument(
                title=s.title or "Unknown",
                source=s.source_id or ""
            ))

    logger.info(
        f"Chat OK | latency={latency_ms}ms | fallback={result.is_fallback} "
        f"| intent={result.intent} | strategy={result.strategy} "
        f"| tokens={result.tokens_used}"
    )

    return ChatResponse(
        success=True,
        answer=result.answer,
        sections=result.sections,
        suggestions=result.suggestions,
        sources=sources,
        latency_ms=latency_ms,
        is_fallback=result.is_fallback
    )