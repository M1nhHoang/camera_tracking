import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings

router = APIRouter(prefix="/api")


@router.get("/database-files/{file_path:path}")
async def get_image(file_path: str):
    """Serve static files (detection images) from shared volume."""
    full_path = os.path.join(settings.STATIC_DIR, file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(full_path, media_type="image/jpeg")
