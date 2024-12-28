from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
import aiohttp
from config import settings

router = APIRouter(prefix="/api")


@router.get("/database-files/{file_path:path}")
async def get_image(file_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{settings.DATABASE_SERVICE_URL}/static/{file_path}"
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=404, detail="Image not found")
            data = await response.read()
            return Response(content=data, media_type="image/jpeg")
