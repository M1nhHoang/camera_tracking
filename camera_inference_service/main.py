import threading
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from typing import Dict, Optional
from pydantic import BaseModel

from service import CameraInferenceService, SharedDetectionModel, DetectionConfig

app = FastAPI()

# Service configurations
services = {
    "database_service": {"hostname": "database_service", "port": "8003"},
    "recognition_service": {"hostname": "recognition_service", "port": "8002"},
}

# Shared model configuration (loaded once, shared across all cameras)
model_config = DetectionConfig(
    model_path="weights/yolo_person_face.pt",
    person_conf_threshold=0.5,
    face_conf_threshold=0.5,
    person_class_id=0,
    face_class_id=1,
)
shared_model = SharedDetectionModel.get_instance(model_config)

# Store camera services
camera_services = {}


class CameraConfig(BaseModel):
    camera_id: str
    name: str
    stream_url: str
    location: Optional[str] = None
    optimal_width: Optional[int] = 640
    optimal_height: Optional[int] = 480
    conf_threshold: Optional[float] = 0.7


async def initialize_cameras():
    """Initialize cameras from database when service starts"""
    try:
        # Get list of cameras from database service
        response = requests.get(
            f"http://{services['database_service']['hostname']}:{services['database_service']['port']}/cameras/list"
        )

        if response.status_code == 200:
            cameras = response.json()
            for camera in cameras:
                # Create camera configuration
                config = CameraConfig(
                    camera_id=camera["id"],
                    name=camera["name"],
                    stream_url=camera["stream_url"],
                    location=camera.get("location"),
                )

                # Create and store camera service
                service = CameraInferenceService.create_from_config(
                    config=config.model_dump(), services=services, shared_model=shared_model
                )
                camera_services[camera["id"]] = service

                # Start camera if it was previously streaming
                if camera.get("status") == "streaming":
                    service.start()

            print(f"Initialized {len(cameras)} cameras from database")
        else:
            print("Failed to get cameras from database")

    except Exception as e:
        print(f"Error initializing cameras: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Run initialization when FastAPI starts"""
    await initialize_cameras()


@app.post("/cameras/add")
async def add_camera(config: CameraConfig):
    """Add new camera to monitoring"""
    try:
        # Create camera service
        service = CameraInferenceService.create_from_config(
            config=config.model_dump(), services=services, shared_model=shared_model
        )

        camera_services[config.camera_id] = service
        return {"camera_id": config.camera_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """Remove camera from monitoring"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Stop camera service
    camera_services[camera_id].stop()
    del camera_services[camera_id]

    return {"success": True}


@app.get("/cameras/{camera_id}/stream")
async def camera_feed(camera_id: str):
    """Get camera video stream"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        camera_services[camera_id].video_feed(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start camera streaming"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_services[camera_id].start()
    return {"success": True}


@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stop camera streaming"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_services[camera_id].stop()
    return {"success": True}


@app.get("/cameras/{camera_id}/status")
async def camera_status(camera_id: str):
    """Get camera status"""
    if camera_id not in camera_services:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {"status": camera_services[camera_id].get_status()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
