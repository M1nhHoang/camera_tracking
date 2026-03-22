import threading
import uvicorn
from fastapi import FastAPI

from service import RecognitionService
from grpc_server import start_grpc_server

app = FastAPI()

# Service configurations
MONGO_URI = "mongodb://mongo_db:27017/"
DATABASE_NAME = "camera_traking"
embedding_service = {"name": "face_embedding_service", "grpc_port": 50052}
vector_db_service = {"name": "chroma_db", "port": 8000}

# Init recognition service (connects to MongoDB directly, no AI model loaded)
recognition_service = RecognitionService(
    mongo_uri=MONGO_URI,
    database_name=DATABASE_NAME,
    embedding_service=embedding_service,
    vector_db_service=vector_db_service,
)

# TODO: User management endpoints (/face_upload, /users/*) have been removed.
# These will be implemented in backend_service which will:
#   - Handle user CRUD directly in MongoDB
#   - Call recognition_service via gRPC for embedding generation
#   - Manage ChromaDB embeddings for user faces
# Endpoints to implement in backend_service:
#   POST /face_upload — register new user with face images
#   DELETE /users/{user_id} — delete user + embeddings
#   DELETE /users/{user_id}/images/{image_path} — delete specific embedding
#   POST /users/{user_id}/images/upload — add images + embeddings

# Start gRPC server alongside FastAPI (port 50051)
grpc_thread = threading.Thread(
    target=start_grpc_server,
    args=(recognition_service,),
    daemon=True,
)
grpc_thread.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, workers=1)
