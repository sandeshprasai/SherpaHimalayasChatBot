import uvicorn
import os
from src.config.settings import settings

if __name__ == "__main__":
    if settings.ENVIRONMENT == "production":
        print("Starting production server using Gunicorn...")
        os.system(f"gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind {settings.HOST}:{settings.PORT} --timeout 120")
    else:
        print("Starting development server...")
        uvicorn.run(
            "src.api.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            workers=1 
        )