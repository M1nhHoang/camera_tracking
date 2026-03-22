import uvicorn
import requests
from fastapi import FastAPI

from service import DetectionService, SharedDetectionModel, DetectionConfig, RecognitionGrpcClient
from router.cameras import router as cameras_router, init_router, CameraConfig

app = FastAPI()

# Service configurations
services = {
    "database_service": {"hostname": "database_service", "port": "8003"},
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

# gRPC client for recognition service (shared across all cameras)
grpc_client = RecognitionGrpcClient(host="recognition_service", port=50051)

# Store camera services
camera_services = {}

# Inject shared state into router
init_router(camera_services, services, shared_model, grpc_client)

# Include routers
app.include_router(cameras_router, prefix="/cameras", tags=["cameras"])


async def initialize_cameras():
    """Initialize cameras from database when service starts"""
    try:
        response = requests.get(
            f"http://{services['database_service']['hostname']}:{services['database_service']['port']}/cameras/list"
        )

        if response.status_code == 200:
            cameras = response.json()
            for camera in cameras:
                config = CameraConfig(
                    camera_id=camera["id"],
                    name=camera["name"],
                    stream_url=camera["stream_url"],
                    location=camera.get("location"),
                )

                service = DetectionService.create_from_config(
                    config=config.model_dump(),
                    services=services,
                    shared_model=shared_model,
                    grpc_client=grpc_client,
                )
                camera_services[camera["id"]] = service

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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
