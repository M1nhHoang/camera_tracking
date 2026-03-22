import threading
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import List

from service import RecognitionService
from grpc_server import start_grpc_server
from router import users

app = FastAPI()

# Service configurations
MONGO_URI = "mongodb://mongo_db:27017/"
DATABASE_NAME = "camera_traking"
embedding_service = {"name": "face_embedding_service", "grpc_port": 50052}
vector_db_service = {"name": "chroma_db", "port": 8000}
model_path = "weights/yolo8n_face_detect.pt"

# Init recognition service (connects to MongoDB directly)
recognition_service = RecognitionService(
    mongo_uri=MONGO_URI,
    database_name=DATABASE_NAME,
    embedding_service=embedding_service,
    vector_db_service=vector_db_service,
    model_path=model_path,
)

# Inject service into routers
users.init_router(recognition_service)

# Include routers
app.include_router(users.router, prefix="/users", tags=["users"])

@app.post("/face_upload")
async def face_upload(
    files: List[UploadFile] = File(...),
    identifier: str = Form(...),
    user_name: str = Form(...),
):
    """Upload multiple face images to register a new user."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = {
        "success": False,
        "processed_images": 0,
        "failed_images": 0,
        "failed_reasons": [],
        "details": [],
    }

    for file in files:
        try:
            file_content = await file.read()
            recognition_service.process_face_image_upload(
                {"identifier": identifier, "user_name": user_name}, file_content
            )
            results["processed_images"] += 1
            results["details"].append(
                {"filename": file.filename, "status": "success", "error": None}
            )
        except ValueError as ve:
            results["failed_images"] += 1
            results["failed_reasons"].append(str(ve))
            results["details"].append(
                {"filename": file.filename, "status": "failed", "error": str(ve)}
            )
        except Exception as e:
            results["failed_images"] += 1
            results["failed_reasons"].append(str(e))
            results["details"].append(
                {"filename": file.filename, "status": "failed", "error": str(e)}
            )

    if results["processed_images"] > 0:
        results["success"] = True
        return results
    else:
        raise HTTPException(status_code=400, detail={
            "message": "No images were processed successfully",
            "total_failed": results["failed_images"],
            "reasons": results["failed_reasons"],
        })


# Start gRPC server alongside FastAPI (port 50051)
grpc_thread = threading.Thread(
    target=start_grpc_server,
    args=(recognition_service,),
    daemon=True,
)
grpc_thread.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, workers=1)
