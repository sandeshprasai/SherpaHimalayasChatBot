import uvicorn
from src.config.settings import settings

if __name__ == "__main__":
   
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.ENVIRONMENT == "development"),
        workers=1 
        )