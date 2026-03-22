from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

from service import DetectionService, SharedDetectionModel, RecognitionGrpcClient

router = APIRouter()


class CameraConfig(BaseModel):
    camera_id: str
    name: str
    stream_url: str
    location: Optional[str] = None
    optimal_width: Optional[int] = 640
    optimal_height: Optional[int] = 480
    conf_threshold: Optional[float] = 0.7


# These will be injected from main.py at startup
camera_services: dict = {}
services: dict = {}
shared_model: SharedDetectionModel = None
grpc_client: RecognitionGrpcClient = None


def init_router(
    _camera_services: dict, _services: dict, _shared_model: SharedDetectionModel,
    _grpc_client: RecognitionGrpcClient = None,
):
    """Inject shared state from main.py into this router module."""
    global camera_services, services, shared_model, grpc_client
    camera_services = _camera_services
    services = _services
    shared_model = _shared_model
    grpc_client = _grpc_client


@router.post("/add")
async def add_camera(config: CameraConfig):
    """Add new camera to monitoring"""
    try:
        service = DetectionService.create_from_config(
            config=config.model_dump(), services=services,
            shared_model=shared_model, grpc_client=grpc_client,
        )
        camera_services[config.camera_id] = service
        return {"camera_id": config.camera_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str):
    """Remove camera from monitoring"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_services[camera_id].stop()
    del camera_services[camera_id]
    return {"success": True}


@router.get("/{camera_id}/stream")
async def camera_feed(camera_id: str):
    """Get camera video stream"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        camera_services[camera_id].video_feed(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start camera streaming"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_services[camera_id].start()
    return {"success": True}


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stop camera streaming"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_services[camera_id].stop()
    return {"success": True}


@router.get("/{camera_id}/status")
async def camera_status(camera_id: str):
    """Get camera status"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {"status": camera_services[camera_id].get_status()}
