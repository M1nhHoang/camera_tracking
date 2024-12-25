from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel
from service.cameras import CameraService

router = APIRouter()


class CameraCreate(BaseModel):
    name: str
    stream_url: str
    location: Optional[str] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    stream_url: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None


class CameraResponse(BaseModel):
    id: str
    name: str
    stream_url: str
    location: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    last_active: Optional[str] = None


@router.post("/add", response_model=dict)
async def add_camera(
    camera: CameraCreate, service: CameraService = Depends(CameraService)
):
    """Add a new camera"""
    camera_id = service.add_camera(
        name=camera.name, stream_url=camera.stream_url, location=camera.location
    )
    return {"camera_id": camera_id}


@router.get("/list", response_model=List[CameraResponse])
async def list_cameras(
    status: Optional[str] = None, service: CameraService = Depends(CameraService)
):
    """Get list of all cameras"""
    return service.list_cameras(status)


@router.get("/active", response_model=List[CameraResponse])
async def get_active_cameras(service: CameraService = Depends(CameraService)):
    """Get list of active cameras"""
    return service.get_active_cameras()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, service: CameraService = Depends(CameraService)):
    """Get camera details"""
    camera = service.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.put("/{camera_id}")
async def update_camera(
    camera_id: str,
    camera: CameraUpdate,
    service: CameraService = Depends(CameraService),
):
    """Update camera information"""
    update_data = {k: v for k, v in camera.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")

    success = service.update_camera(camera_id, update_data)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"success": True}


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: str, service: CameraService = Depends(CameraService)
):
    """Delete a camera"""
    success = service.delete_camera(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"success": True}


@router.put("/{camera_id}/status")
async def update_camera_status(
    camera_id: str, status: str, service: CameraService = Depends(CameraService)
):
    """Update camera status"""
    if status not in ["offline", "streaming", "error", "paused"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    success = service.update_camera_status(camera_id, status)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"success": True}
