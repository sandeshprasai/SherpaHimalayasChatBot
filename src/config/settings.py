"""
src/config/settings.py
──────────────────────
Centralized application configuration for:
- FastAPI
- RAG pipeline
- ChromaDB
- LLMs
- Production deployment
"""

import importlib
import os

from typing import List
from pydantic_settings import BaseSettings


# =============================================================================
# Disable telemetry / analytics noise
# =============================================================================

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("LLAMA_INDEX_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")

try:
    analytics = importlib.import_module("analytics")
    analytics.capture = lambda *args, **kwargs: None
except Exception:
    pass

try:
    posthog = importlib.import_module("posthog")
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass


# =============================================================================
# Root Resolution
# =============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(HERE, "../..")
)


# =============================================================================
# Settings
# =============================================================================

class Settings(BaseSettings):

    # =========================================================================
    # APP
    # =========================================================================

    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # =========================================================================
    # SERVER
    # =========================================================================

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # =========================================================================
    # SECURITY
    # =========================================================================

    API_KEY: str = "supersecretapikey"

    # =========================================================================
    # GROQ
    # =========================================================================

    GROQ_API_KEY: str

    # =========================================================================
    # LOGGING
    # =========================================================================

    LOG_LEVEL: str = "INFO"

    # =========================================================================
    # CORS
    # =========================================================================

    ALLOWED_ORIGINS: str = ""

    # =========================================================================
    # PROJECT ROOT
    # =========================================================================

    project_root: str = PROJECT_ROOT

    # =========================================================================
    # DATA DIRECTORIES
    # =========================================================================

    data_dir: str = os.path.join(PROJECT_ROOT, "data")

    rag_ready_dir: str = os.path.join(
        PROJECT_ROOT,
        "data/datasets/rag_ready"
    )

    # =========================================================================
    # INDEXES
    # =========================================================================

    vector_db_dir: str = os.path.join(
        PROJECT_ROOT,
        "data/indexes/chroma/vector_db"
    )

    metadata_index: str = os.path.join(
        PROJECT_ROOT,
        "data/indexes/metadata/metadata_index.json"
    )

    cleaned_metadata_index: str = os.path.join(
        PROJECT_ROOT,
        "data/indexes/metadata/metadata_index.cleaned.json"
    )

    # =========================================================================
    # MODEL CONFIG
    # =========================================================================

    llm_model: str = "llama-3.3-70b-versatile"

    llm_temperature: float = 0.1

    llm_max_tokens: int = 1024

    embed_model: str = "BAAI/bge-small-en-v1.5"

    # =========================================================================
    # RETRIEVAL CONFIG
    # =========================================================================

    similarity_top_k: int = 5

    chroma_collection: str = "sherpa_knowledge"

    # =========================================================================
    # BRANDING
    # =========================================================================

    company_email: str = "info@sherpahimalaya.com"

    company_name: str = "Sherpa Himalaya"

    system_name: str = "Travel Information System"

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    daily_token_limit: int = 100_000

    # =========================================================================
    # ENV FILE
    # =========================================================================

    class Config:
        env_file = os.path.join(PROJECT_ROOT, ".env")
        extra = "ignore"

    # =========================================================================
    # Parsed CORS Origins
    # =========================================================================

    @property
    def cors_origins(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


# =============================================================================
# Singleton Settings Instance
# =============================================================================

settings = Settings()