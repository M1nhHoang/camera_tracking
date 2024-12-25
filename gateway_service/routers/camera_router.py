from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, List
from pydantic import BaseModel
from services.camera_service import CameraService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class CameraCreate(BaseModel):
    name: str
    stream_url: str
    location: Optional[str] = None
    optimal_width: Optional[int] = 640
    optimal_height: Optional[int] = 480
    conf_threshold: Optional[float] = 0.7


class CameraResponse(BaseModel):
    id: str
    name: str
    stream_url: str
    location: Optional[str] = None
    status: str
    last_active: Optional[str] = None


@router.post("/add")
async def add_camera(
    camera: CameraCreate, service: CameraService = Depends(CameraService)
):
    """Add a new camera"""
    try:
        camera_dict = camera.model_dump()
        logger.info(f"Adding camera with data: {camera_dict}")

        camera_id = await service.add_camera(camera_dict)
        return {"camera_id": camera_id}
    except Exception as e:
        logger.error(f"Failed to add camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[CameraResponse])
async def list_cameras(service: CameraService = Depends(CameraService)):
    """Get list of all cameras"""
    try:
        cameras = await service.get_cameras()
        return cameras
    except Exception as e:
        logger.error(f"Failed to list cameras: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, service: CameraService = Depends(CameraService)):
    """Get camera details"""
    try:
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get camera {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{camera_id}")  # Changed from /delete/{camera_id}
async def delete_camera(
    camera_id: str, service: CameraService = Depends(CameraService)
):
    """Delete camera"""
    try:
        # First check if camera exists
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        success = await service.delete_camera(camera_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete camera")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete camera {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_id}/start")
async def start_camera(camera_id: str, service: CameraService = Depends(CameraService)):
    """Start camera streaming"""
    try:
        # Check if camera exists
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        success = await service.start_camera(camera_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start camera")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to start camera {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: str, service: CameraService = Depends(CameraService)):
    """Stop camera streaming"""
    try:
        # Check if camera exists
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        success = await service.stop_camera(camera_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop camera")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to stop camera {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/status")
async def camera_status(
    camera_id: str, service: CameraService = Depends(CameraService)
):
    """Get camera status"""
    try:
        # Check if camera exists
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        status = await service.camera_status(camera_id)
        return {"status": status}
    except Exception as e:
        logger.error(f"Failed to get camera status {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/stream")
async def stream_camera(
    camera_id: str, service: CameraService = Depends(CameraService)
):
    """Get camera video stream"""
    try:
        # Check if camera exists
        camera = await service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        return StreamingResponse(
            service.get_camera_stream(camera_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        logger.error(f"Failed to stream camera {camera_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
