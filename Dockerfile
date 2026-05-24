
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY src/       ./src/
COPY scripts/   ./scripts/
COPY run.py     ./

RUN mkdir -p /app/data/datasets/rag_ready \
             /app/data/indexes/chroma/vector_db \
             /app/data/indexes/metadata \
             /app/data/logs \
 && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["python", "run.py"]