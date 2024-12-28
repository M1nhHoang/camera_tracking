import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import numpy as np
from PIL import Image
from io import BytesIO

from service import FaceIdentifyService

app = FastAPI()

# Service configurations
database_service = {"name": "database_service", "port": 8003}
embedding_service = {"name": "face_embedding_service", "port": 8001}
vector_db_service = {"name": "chroma_db", "port": 8000}
model_path = "weights/yolo8n_face_detect.pt"

face_identify_service = FaceIdentifyService(
    database_service=database_service,
    embedding_service=embedding_service,
    vector_db_service=vector_db_service,
    model_path=model_path,
)


# Existing endpoints
@app.post("/face_identification")
async def face_identification(
    origin_image: UploadFile = File(...),
    detect_image: UploadFile = File(...),
    detect_id: int = -1,
    camera_id: Optional[str] = None,
):
    if detect_id == -1:
        raise HTTPException(status_code=400, detail="Detect id is required.")

    origin_image = Image.open(BytesIO(await origin_image.read()))
    origin_image = origin_image.convert("RGB")

    detect_image = Image.open(BytesIO(await detect_image.read()))
    detect_image = detect_image.convert("RGB")

    origin_image = np.array(origin_image)
    detect_image = np.array(detect_image)

    face_identify_service.process_detect_queue(
        detect_id, origin_image, detect_image, camera_id=camera_id
    )

    return {"success": True}


@app.post("/face_upload")
async def face_upload(
    file: UploadFile = File(...),
    identifier: str = Form(...),
    user_name: str = Form(...),
):
    """Upload face image with user info via form-data"""
    try:
        # Đọc nội dung file
        file_content = await file.read()

        # Process face image upload
        face_identify_service.process_face_image_upload(
            {"identifier": identifier, "user_name": user_name}, file_content
        )

        return {"success": True}

    except Exception as e:
        print(f"Error in face_upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# New embedding management endpoints
class EmbeddingCreate(BaseModel):
    embedding_id: str
    embedding: List[float]
    metadata: dict


class EmbeddingUpdate(BaseModel):
    embedding: List[float]
    metadata: dict


@app.post("/embeddings")
async def create_embedding(data: EmbeddingCreate):
    """Create a new face embedding"""
    success = face_identify_service.create_face_embedding(
        data.embedding_id, data.embedding, data.metadata
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    return {"success": True}


@app.get("/embeddings/{embedding_id}")
async def get_embedding(embedding_id: str):
    """Get face embedding by ID"""
    embedding = face_identify_service.get_face_embedding(embedding_id)
    if not embedding:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding


@app.get("/embeddings")
async def get_all_embeddings(limit: int = 100, offset: int = 0):
    """Get all face embeddings with pagination"""
    embeddings = face_identify_service.get_all_face_embeddings(limit, offset)
    return {"success": True, "data": embeddings, "limit": limit, "offset": offset}


@app.put("/embeddings/{embedding_id}")
async def update_embedding(embedding_id: str, data: EmbeddingUpdate):
    """Update an existing face embedding"""
    success = face_identify_service.update_face_embedding(
        embedding_id, data.embedding, data.metadata
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update embedding")
    return {"success": True}


@app.delete("/embeddings/{embedding_id}")
async def delete_embedding(embedding_id: str):
    """Delete a face embedding"""
    success = face_identify_service.delete_face_embedding(embedding_id)
    if not success:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return {"success": True}


@app.post("/embeddings/search")
async def search_embeddings(query_embedding: List[float], n_results: Optional[int] = 5):
    """Search for similar face embeddings"""
    results = face_identify_service.search_similar_faces(query_embedding, n_results)
    return JSONResponse(content=results)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, workers=1)
