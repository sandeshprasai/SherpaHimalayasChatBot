# src/api/main.py
"""
FastAPI application factory.
All configuration, middleware, and routing is wired here.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.middleware import APIKeyMiddleware, RequestLogMiddleware
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.dependencies import initialize_pipeline, limiter
from src.config.settings import settings
from src.services.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup / shutdown hooks
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup before the first request is served."""
    logger.info(f"Starting {settings.company_name} Chatbot API | env={settings.ENVIRONMENT}")
    initialize_pipeline()          # Load ChromaDB + embedding model into RAM
    yield
    logger.info("Shutting down API server.")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"{settings.company_name} Chatbot API",
    version="1.0.0",
    description=(
        "Production AI backend for the Sherpa Himalaya chatbot. "
        "Hybrid RAG system with metadata and vector retrieval, "
        "powered by Groq Llama 3.3 70B."
    ),
    lifespan=lifespan,
    # Hide docs in production to avoid exposing the schema publicly
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# Attach rate limiter state to app
# ─────────────────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler       # Returns 429 with Retry-After header
)

# ─────────────────────────────────────────────────────────────────────────────
# Middleware (order matters — outermost executes first)
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(RequestLogMiddleware)   # 1. Log every request
app.add_middleware(APIKeyMiddleware)       # 2. Enforce API key on /chat

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # e.g. ["https://sherpahimalaya.com"]
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global exception handlers
# ─────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": f"{settings.company_name} Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }

app.include_router(health_router)
app.include_router(chat_router)