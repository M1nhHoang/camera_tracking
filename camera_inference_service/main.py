import threading
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

from service import CameraInferenceService

app = FastAPI()

# ENV load
camera_streams = {
    "video_feed": "D:/camera/data/output_video.avi"
    # "10.9.5.39": "rtsp://admin:12345abcde@10.9.5.39:554",
    # "camera2": "rtsp://admin:12345abcde@10.9.5.40:554",
    # Add more cameras as needed
}
database_serivce = {
    "hostname": "database_service",
    "port": "8003",
}
face_idtification_service = {
    "hostname": "face_identify_service",
    "port": "8002",
}
model_path = "weights/yolo8n_human_detect.pt"

# Create instances of CameraInferenceService for each camera
camera_services = {
    name: CameraInferenceService(
        stream_url=url,
        database_serivce=database_serivce,
        face_idtification_service=face_idtification_service,
        model_path=model_path,
    )
    for name, url in camera_streams.items()
}


@app.on_event("startup")
def startup_event():
    for name, service in camera_services.items():
        video_thread = threading.Thread(target=service.video_stream, daemon=True)
        video_thread.start()


@app.get("/camera/{camera_name}")
async def camera_feed(request: Request, camera_name: str):
    if camera_name in camera_services:
        return StreamingResponse(
            camera_services[camera_name].video_feed(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    return {"error": "Camera not found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
