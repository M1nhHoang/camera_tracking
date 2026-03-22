import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
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
    face_image: Optional[UploadFile] = File(None),
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

    # Pre-cropped face from camera_inference_service (if available)
    face_image_arr = None
    if face_image is not None:
        face_image_arr = Image.open(BytesIO(await face_image.read()))
        face_image_arr = np.array(face_image_arr.convert("RGB"))

    face_identify_service.process_detect_queue(
        detect_id, origin_image, detect_image, camera_id=camera_id,
        face_image=face_image_arr,
    )

    return {"success": True}


@app.post("/face_upload")
async def face_upload(
    files: List[UploadFile] = File(...),
    identifier: str = Form(...),
    user_name: str = Form(...),
):
    """Upload multiple face images with user info via form-data"""
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
            # Đọc nội dung file
            file_content = await file.read()

            # Process face image upload
            face_identify_service.process_face_image_upload(
                {"identifier": identifier, "user_name": user_name}, file_content
            )

            # Track successful processing
            results["processed_images"] += 1
            results["details"].append(
                {"filename": file.filename, "status": "success", "error": None}
            )

        except ValueError as ve:
            # Handle validation errors (e.g., no face, multiple faces, quality issues)
            results["failed_images"] += 1
            results["failed_reasons"].append(str(ve))
            results["details"].append(
                {"filename": file.filename, "status": "failed", "error": str(ve)}
            )

        except Exception as e:
            # Handle other unexpected errors
            results["failed_images"] += 1
            results["failed_reasons"].append(str(e))
            results["details"].append(
                {"filename": file.filename, "status": "failed", "error": str(e)}
            )

    # Set overall success if at least one image was processed successfully
    if results["processed_images"] > 0:
        results["success"] = True
        return results
    else:
        # If no images were processed successfully, raise an error with details
        error_detail = {
            "message": "No images were processed successfully",
            "total_failed": results["failed_images"],
            "reasons": results["failed_reasons"],
        }
        raise HTTPException(status_code=400, detail=error_detail)


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


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get user details and embeddings"""
    try:
        # Query ChromaDB for user's embeddings
        results = face_identify_service.chroma_collection.get(
            where={"user_id": user_id},
            include=["metadatas"],  # Only include metadata, not embeddings
        )

        if not results or not results["metadatas"]:
            raise HTTPException(status_code=404, detail="User not found")

        # Get the first metadata entry for user info
        user_metadata = results["metadatas"][0]

        # Return formatted user data
        return {
            "user_id": user_id,
            "username": user_metadata.get("user_name"),
            "identifier": user_metadata.get("identifier"),
            "embedding_count": len(results["metadatas"]),
            "images": [
                metadata.get("truth_image_path")
                for metadata in results["metadatas"]
                if metadata.get("truth_image_path")
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete all face embeddings for a specific user"""
    try:
        success = face_identify_service.delete_user_embeddings(user_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or no embeddings to delete",
            )
        return {"message": f"All embeddings for user {user_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/users/{user_id}")
async def update_user(user_id: str, metadata: Dict):
    """Update metadata for all embeddings of a user"""
    try:
        success = face_identify_service.update_user_metadata(user_id, metadata)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or no embeddings to update",
            )
        return {"message": f"Metadata for user {user_id} updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/count")
async def count_user_embeddings(user_id: str):
    """Get count of embeddings for a user"""
    try:
        count = face_identify_service.count_user_embeddings(user_id)
        return {"user_id": user_id, "embeddings_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users")
async def list_users(skip: int = 0, limit: int = 100):
    """List all users with embeddings"""
    try:
        results = face_identify_service.chroma_collection.get()
        if not results or not results["metadatas"]:
            return {"users": [], "total": 0}

        # Extract unique users from metadata
        users = {}
        for metadata in results["metadatas"]:
            if metadata.get("user_id") and metadata.get("user_name"):
                users[metadata["user_id"]] = {
                    "user_id": metadata["user_id"],
                    "user_name": metadata["user_name"],
                    "identifier": metadata.get("identifier"),
                }

        # Convert to list and apply pagination
        users_list = list(users.values())
        total = len(users_list)
        paginated_users = users_list[skip : skip + limit]

        return {"users": paginated_users, "total": total, "skip": skip, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}/images/{image_path}")
async def delete_user_image(user_id: str, image_path: str):
    """Delete face embedding for specific image"""
    # try:
    # Get embedding that matches user_id and image_path
    results = face_identify_service.chroma_collection.get(
        where={
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"truth_image_path": {"$eq": image_path}},
            ]
        }
    )

    print(results)
    print(user_id, image_path)

    if results and results["ids"]:
        # Delete matching embedding(s)
        face_identify_service.chroma_collection.delete(ids=results["ids"])
        return {"success": True}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for user {user_id} and image {image_path}",
        )

    # except Exception as e:
    #     raise HTTPException(
    #         status_code=500, detail=f"Error deleting embedding: {str(e)}"
    #     )


@app.post("/users/{user_id}/images/upload")
async def add_user_images(
    user_id: str,
    identifier: str = Form(...),
    user_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload and process new images for existing user"""
    results = {
        "success": False,
        "processed_images": 0,
        "failed_images": 0,
        "failed_reasons": [],
        "details": [],
    }

    for file in files:
        try:
            # Read and validate image
            file_content = await file.read()
            image = Image.open(BytesIO(file_content))
            image = np.array(image)

            # Validate face in image
            face_image = face_identify_service.face_validate(image)

            # Convert face image to base64 for storage
            face_image_base64 = face_identify_service.convert_image_to_base64(
                face_image
            )

            # Get face embedding
            face_embedding = face_identify_service.embedding(face_image)
            if face_embedding is None:
                raise ValueError(f"Failed to generate embedding for {file.filename}")

            # Insert into ChromaDB
            metadata = {
                "user_id": user_id,
                "user_name": user_name,
                "identifier": identifier,
                "image_name": file.filename,
                "image_path": face_image_base64,
            }
            face_identify_service.chromadb_insert(metadata, face_embedding)

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

    # Set overall success if at least one image was processed
    results["success"] = results["processed_images"] > 0
    return results


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, workers=1)
