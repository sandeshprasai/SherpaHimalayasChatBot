# =============================================================================
# Stage 1 — Builder
# Install all Python dependencies into a clean prefix so we can copy them
# into the final image without build tools.
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed only during build (gcc for some C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# =============================================================================
# Stage 2 — Runtime
# Lean final image: no compilers, no build artefacts.
# =============================================================================
FROM python:3.11-slim AS runtime

# ── System runtime libraries ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy installed packages from builder ─────────────────────────────────────
COPY --from=builder /install /usr/local

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Copy application source code ──────────────────────────────────────────────
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY run.py     ./

# ── Mount point for external data volume (indexes + datasets) ─────────────────
# This directory will be bind-mounted or filled by a Docker volume at runtime.
# It is intentionally empty in the image — data lives OUTSIDE the image.
RUN mkdir -p /app/data/datasets/rag_ready \
             /app/data/indexes/chroma/vector_db \
             /app/data/indexes/metadata \
             /app/data/logs \
             /app/data/cache \
 && chown -R appuser:appuser /app

USER appuser

# ── Runtime environment ───────────────────────────────────────────────────────
# Sensible defaults; all of these are overridden by your .env / docker-compose.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=error \
    ENVIRONMENT=production \
    DEBUG=false \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data \
    RAG_READY_DIR=/app/data/datasets/rag_ready \
    CHROMA_DB_PATH=/app/data/indexes/chroma/vector_db \
    METADATA_INDEX_PATH=/app/data/indexes/metadata/metadata_index.json

EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# run.py detects ENVIRONMENT=production and launches Gunicorn automatically.
CMD ["python", "run.py"]