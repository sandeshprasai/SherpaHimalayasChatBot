

import time
from functools import lru_cache
from fastapi import Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.pipeline import Pipeline
from src.services.logger import logger


limiter = Limiter(key_func=get_remote_address)


# ─────────────────────────────────────────────
# Pipeline Singleton
# ─────────────────────────────────────────────
_pipeline_instance: Pipeline | None = None


def get_pipeline() -> Pipeline:
   
    global _pipeline_instance
    if _pipeline_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI pipeline is not ready. Please try again in a moment."
        )
    return _pipeline_instance


def initialize_pipeline() -> None:
    
    global _pipeline_instance
    logger.info("Initializing AI Pipeline...")
    start = time.time()
    try:
        _pipeline_instance = Pipeline(debug=False)
        elapsed = round(time.time() - start, 2)
        logger.info(f"Pipeline ready in {elapsed}s")
    except Exception as e:
        logger.error(f"CRITICAL: Pipeline failed to initialize: {e}")
        # Do not raise — app will start but return 503 on chat requests
        _pipeline_instance = None