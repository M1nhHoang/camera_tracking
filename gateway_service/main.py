import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import (
    web_router, 
    user_router, 
    detect_router, 
    camera_router, 
    static_router,
    dashboard_router
)
from config import settings

app = FastAPI(
    title="Camera Track Gateway",
    description="API Gateway and Web Interface for Camera Track System",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files for serving CSS, JS, etc
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)

# Include API routers with prefix
app.include_router(user_router.router, prefix="/api/users", tags=["users"])
app.include_router(detect_router.router, prefix="/api/detect", tags=["detect"])
app.include_router(camera_router.router, prefix="/api/camera", tags=["camera"])
app.include_router(dashboard_router.router, tags=["dashboard"])
app.include_router(static_router.router, tags=["static"])

# Include web router for serving pages
app.include_router(web_router.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, "static", "css"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "js"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "database_files"), exist_ok=True)

    uvicorn.run(
        "main:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=True
    )