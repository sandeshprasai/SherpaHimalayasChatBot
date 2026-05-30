from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.dependencies import get_pipeline
from src.config.settings import settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", summary="Service health check")
async def health_check():
    try:
       
        get_pipeline()
        pipeline_ready = True
    except Exception:
        pipeline_ready = False
        
    content = {
        "status": "healthy" if pipeline_ready else "degraded",
        "pipeline_ready": pipeline_ready,
        "environment": settings.ENVIRONMENT,
        "service": f"{settings.company_name} Chatbot API"
    }
    
    status_code = status.HTTP_200_OK if pipeline_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(status_code=status_code, content=content)