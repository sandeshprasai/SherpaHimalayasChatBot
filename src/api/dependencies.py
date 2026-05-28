import time
from fastapi import Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.pipeline import Pipeline
from src.services.logger import logger

limiter = Limiter(key_func=get_remote_address)

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
        
        logger.info("Warming up Vector Engine...")
        if not _pipeline_instance._vec:
            from src.retrieval.vector import VectorRetriever
            _pipeline_instance._vec = VectorRetriever()
        _pipeline_instance._vec._ensure_ready()
        
        elapsed = round(time.time() - start, 2)
        logger.info(f"Pipeline ready in {elapsed}s")
    except Exception as e:
        logger.error(f"CRITICAL: Pipeline failed to initialize: {e}")
        _pipeline_instance = None
        raise RuntimeError(f"AI Pipeline failed to start: {e}")