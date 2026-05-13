from fastapi import APIRouter
from src.api.dependencies import get_pipeline # <--- Change this
from src.config.settings import settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", summary="Service health check")
async def health_check():
    try:
        # Try to get the pipeline; if it exists, we are healthy
        p = get_pipeline()
        pipeline_ready = True
    except:
        pipeline_ready = False
        
    return {
        "status": "healthy" if pipeline_ready else "degraded",
        "pipeline_ready": pipeline_ready,
        "environment": settings.ENVIRONMENT,
        "service": f"{settings.company_name} Chatbot API"
    }