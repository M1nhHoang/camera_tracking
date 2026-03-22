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
    dashboard_router,
)
from config import settings

app = FastAPI(
    title="Camera Tracking Backend",
    description="Backend service for Camera Tracking System",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files for CSS, JS
app.mount(
    "/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"
)

# API routers
app.include_router(user_router.router, prefix="/api/users", tags=["users"])
app.include_router(detect_router.router, prefix="/api/detect", tags=["detect"])
app.include_router(camera_router.router, prefix="/api/camera", tags=["camera"])
app.include_router(dashboard_router.router, tags=["dashboard"])
app.include_router(static_router.router, tags=["static"])

# Web UI router
app.include_router(web_router.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "static", "css"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "js"), exist_ok=True)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
