import importlib
import os
from typing import List
from pydantic import Field, field_validator
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

# =============================================================================
# Root Resolution
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))

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
    # APP & SERVER
    # =========================================================================
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    API_KEY: str = Field(..., env="API_KEY")
    
    @field_validator("API_KEY")
    @classmethod
    def api_key_must_be_strong(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("API_KEY must be at least 32 characters long.")
        return v

    # =========================================================================
    # OPENAI CONFIG (Replaces Groq)
    # =========================================================================
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    # High-reasoning model for final synthesis
    llm_model: str = Field(
        default="gpt-4o-mini", 
        env="LLM_MODEL"
    )

    # Fast, low-cost model for logic/planning/rewriting
    logic_model: str = Field(
        default="gpt-4o-mini",
        env="LOGIC_MODEL"
    )

    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, env="LLM_MAX_TOKENS")
    embed_model: str = Field(default="text-embedding-3-small", env="EMBED_MODEL")

    # =========================================================================
    # DATA & INDEXES
    # =========================================================================
    data_dir: str = Field(default=os.path.join(PROJECT_ROOT, "data"), env="DATA_DIR")
    rag_ready_dir: str = Field(default=os.path.join(PROJECT_ROOT, "data/datasets/rag_ready"), env="RAG_READY_DIR")
    vector_db_dir: str = Field(..., env="CHROMA_DB_PATH")
    metadata_index: str = Field(..., env="METADATA_INDEX_PATH")
    cleaned_metadata_index: str = Field(
        default=os.path.join(PROJECT_ROOT, "data/indexes/metadata/metadata_index.cleaned.json"),
        env="CLEANED_METADATA_INDEX_PATH"
    )

    # =========================================================================
    # RETRIEVAL CONFIG
    # =========================================================================
    similarity_top_k: int = Field(default=5, env="SIMILARITY_TOP_K")
    chroma_collection: str = Field(..., env="CHROMA_COLLECTION")

    # =========================================================================
    # BRANDING
    # =========================================================================
    company_email: str = Field(..., env="COMPANY_EMAIL")
    company_name: str = Field(..., env="COMPANY_NAME")
    system_name: str = Field(..., env="SYSTEM_NAME")
    greeting: str = Field(default="Namaste", env="GREETING")

    # =========================================================================
    # INFRASTRUCTURE
    # =========================================================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ALLOWED_ORIGINS: str = Field(default="*", env="ALLOWED_ORIGINS")
    daily_token_limit: int = Field(default=1000000, env="DAILY_TOKEN_LIMIT")

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

settings = Settings()