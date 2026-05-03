"""
config/settings.py
──────────────────
Single source of truth for every constant, path, and model name
in the Sherpa Himalaya chatbot.

Import anywhere with:
    from config.settings import settings
"""

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
    import analytics
    analytics.capture = lambda *args, **kwargs: None
except Exception:
    pass
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# ROOT RESOLUTION  (works regardless of working directory)
# ──────────────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))   # src/config/
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../..")) # project root


@dataclass(frozen=True)
class Settings:

    # ── Project layout ─────────────────────────────────────────────────────
    project_root:     str = _PROJECT_ROOT
    data_dir:         str = os.path.join(_PROJECT_ROOT, "data")
    rag_ready_dir:    str = os.path.join(_PROJECT_ROOT, "data/rag_ready")
    vector_db_dir:    str = os.path.join(_PROJECT_ROOT, "data/vector_db")
    metadata_index:   str = os.path.join(_PROJECT_ROOT, "data/rag_ready/metadata_index.json")
    # Optional cleaned metadata produced by src/utils/metadata_cleaner.py
    cleaned_metadata_index: str = os.path.join(_PROJECT_ROOT, "data/rag_ready/metadata_index.cleaned.json")

    # ── LLM ────────────────────────────────────────────────────────────────
    llm_model:        str = "llama-3.3-70b-versatile"   # groq
    llm_temperature:  float = 0.1                        # low = factual
    llm_max_tokens:   int = 1024

    # ── Embeddings ─────────────────────────────────────────────────────────
    embed_model:      str = "BAAI/bge-small-en-v1.5"    # local HuggingFace

    # ── Retrieval ──────────────────────────────────────────────────────────
    similarity_top_k: int = 5        # chunks fetched per vector query
    chroma_collection:str = "sherpa_knowledge"

    # ── Token budget (Groq free tier) ──────────────────────────────────────
    daily_token_limit:      int = 100_000
    estimated_resp_tokens:  int = 800

    # ── Company contact (used in fallback messages) ─────────────────────────
    company_email:    str = "info@sherpahimalaya.com"
    company_name:     str = "Sherpa Himalaya"

    # ── Difficulty ranking (shared across retrieval + classifier) ───────────
    difficulty_rank: dict = field(default_factory=lambda: {
        "easy":                   1,
        "moderate":               2,
        "easy/moderate":          2,
        "moderately challenging": 3,
        "challenging":            4,
        "strenuous":              5,
        "extreme":                6,
    })


# ── Singleton ──────────────────────────────────────────────────────────────
settings = Settings()