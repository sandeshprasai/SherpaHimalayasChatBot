import importlib
import os

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        extra="ignore",
        case_sensitive=False
    )

    # =========================================================================
    # APP
    # =========================================================================

    ENVIRONMENT: str = Field(..., env="ENVIRONMENT")
    DEBUG: bool = Field(..., env="DEBUG")

    # =========================================================================
    # SERVER
    # =========================================================================

    HOST: str = Field(..., env="HOST")
    PORT: int = Field(..., env="PORT")

    # =========================================================================
    # SECURITY
    # =========================================================================

    API_KEY: str = Field(..., env="API_KEY")

    # =========================================================================
    # GROQ
    # =========================================================================

    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")

    # =========================================================================
    # LOGGING
    # =========================================================================

    LOG_LEVEL: str = Field(..., env="LOG_LEVEL")

    # =========================================================================
    # CORS
    # =========================================================================

    ALLOWED_ORIGINS: str = Field(..., env="ALLOWED_ORIGINS")

    # =========================================================================
    # PROJECT ROOT
    # =========================================================================

    project_root: str = PROJECT_ROOT

    # =========================================================================
    # DATA DIRECTORIES
    # =========================================================================

    data_dir: str = Field(
        default=os.path.join(PROJECT_ROOT, "data"),
        env="DATA_DIR"
    )

    rag_ready_dir: str = Field(
        default=os.path.join(
            PROJECT_ROOT,
            "data/datasets/rag_ready"
        ),
        env="RAG_READY_DIR"
    )

    # =========================================================================
    # INDEXES
    # =========================================================================

    vector_db_dir: str = Field(
        ...,
        env="CHROMA_DB_PATH"
    )

    metadata_index: str = Field(
        ...,
        env="METADATA_INDEX_PATH"
    )

    cleaned_metadata_index: str = Field(
        default=os.path.join(
            PROJECT_ROOT,
            "data/indexes/metadata/metadata_index.cleaned.json"
        ),
        env="CLEANED_METADATA_INDEX_PATH"
    )

    # =========================================================================
    # MODEL CONFIG
    # =========================================================================

    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        env="LLM_MODEL"
    )

    llm_temperature: float = Field(
        default=0.1,
        env="LLM_TEMPERATURE"
    )

    llm_max_tokens: int = Field(
        default=1024,
        env="LLM_MAX_TOKENS"
    )

    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        env="EMBED_MODEL"
    )

    # =========================================================================
    # RETRIEVAL CONFIG
    # =========================================================================

    similarity_top_k: int = Field(
        default=5,
        env="SIMILARITY_TOP_K"
    )

    chroma_collection: str = Field(
        ...,
        env="CHROMA_COLLECTION"
    )

    # =========================================================================
    # BRANDING
    # =========================================================================

    company_email: str = Field(
        ...,
        env="COMPANY_EMAIL"
    )

    company_name: str = Field(
        ...,
        env="COMPANY_NAME"
    )

    system_name: str = Field(
        ...,
        env="SYSTEM_NAME"
    )

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    daily_token_limit: int = Field(
        default=100000,
        env="DAILY_TOKEN_LIMIT"
    )

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