"""
config/settings.py
──────────────────
Single source of truth for every constant, path, and model name
in the Sherpa Himalaya chatbot.

Import anywhere with:
    from config.settings import settings
"""

import importlib
import os
# Disable common telemetry/analytics from deps to reduce noisy console errors
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("LLAMA_INDEX_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")
# Try to monkeypatch common analytics libraries to avoid noisy errors
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
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# ROOT RESOLUTION  (works regardless of working directory)
# ──────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))

@dataclass(frozen=True)
class Settings:
    project_root: str = _PROJECT_ROOT
    data_dir: str = os.path.join(_PROJECT_ROOT, "data")
    
    # Dataset Path
    rag_ready_dir: str = os.path.join(_PROJECT_ROOT, "data/datasets/rag_ready")
    
    # Index Paths (Updated for refactored structure)
    vector_db_dir: str = os.path.join(_PROJECT_ROOT, "data/indexes/chroma/vector_db")
    metadata_index: str = os.path.join(_PROJECT_ROOT, "data/indexes/metadata/metadata_index.json")
    cleaned_metadata_index: str = os.path.join(_PROJECT_ROOT, "data/indexes/metadata/metadata_index.cleaned.json")

    # LLM & Embedding Config
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval Config
    similarity_top_k: int = 5
    chroma_collection: str = "sherpa_knowledge"

    # Company Branding
    company_email: str = "info@sherpahimalaya.com"
    company_name: str = "Sherpa Himalaya"
    system_name: str = "Travel Information System"

    daily_token_limit: int = 100_000

settings = Settings()