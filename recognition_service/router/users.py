from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import List, Dict
import numpy as np
from PIL import Image
from io import BytesIO

router = APIRouter()

# Injected from main.py
recognition_service = None


def init_router(_recognition_service):
    global recognition_service
    recognition_service = _recognition_service


@router.get("/")
async def list_users(skip: int = 0, limit: int = 100):
    """List all users with embeddings."""
    try:
        results = recognition_service.chroma_collection.get()
        if not results or not results["metadatas"]:
            return {"users": [], "total": 0}

        users = {}
        for metadata in results["metadatas"]:
            if metadata.get("user_id") and metadata.get("user_name"):
                users[metadata["user_id"]] = {
                    "user_id": metadata["user_id"],
                    "user_name": metadata["user_name"],
                    "identifier": metadata.get("identifier"),
                }

        users_list = list(users.values())
        total = len(users_list)
        paginated_users = users_list[skip : skip + limit]

        return {"users": paginated_users, "total": total, "skip": skip, "limit": limit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Get user details and embeddings."""
    try:
        results = recognition_service.chroma_collection.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )

        if not results or not results["metadatas"]:
            raise HTTPException(status_code=404, detail="User not found")

        user_metadata = results["metadatas"][0]

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """Delete all face embeddings for a user."""
    try:
        success = recognition_service.delete_user_embeddings(user_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or no embeddings to delete",
            )
        return {"message": f"All embeddings for user {user_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}")
async def update_user(user_id: str, metadata: Dict):
    """Update metadata for all embeddings of a user."""
    try:
        success = recognition_service.update_user_metadata(user_id, metadata)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or no embeddings to update",
            )
        return {"message": f"Metadata for user {user_id} updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/count")
async def count_user_embeddings(user_id: str):
    """Get count of embeddings for a user."""
    try:
        count = recognition_service.count_user_embeddings(user_id)
        return {"user_id": user_id, "embeddings_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}/images/{image_path}")
async def delete_user_image(user_id: str, image_path: str):
    """Delete face embedding for specific image."""
    results = recognition_service.chroma_collection.get(
        where={
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"truth_image_path": {"$eq": image_path}},
            ]
        }
    )

    if results and results["ids"]:
        recognition_service.chroma_collection.delete(ids=results["ids"])
        return {"success": True}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for user {user_id} and image {image_path}",
        )


@router.post("/{user_id}/images/upload")
async def add_user_images(
    user_id: str,
    identifier: str = Form(...),
    user_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload and process new images for existing user."""
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
            image = Image.open(BytesIO(file_content))
            image = np.array(image)

            face_image = recognition_service.face_validate(image)
            face_image_base64 = recognition_service.convert_image_to_base64(face_image)
            face_embedding = recognition_service.embedding(face_image)
            if face_embedding is None:
                raise ValueError(f"Failed to generate embedding for {file.filename}")

            metadata = {
                "user_id": user_id,
                "user_name": user_name,
                "identifier": identifier,
                "image_name": file.filename,
                "image_path": face_image_base64,
            }
            recognition_service.chromadb_insert(metadata, face_embedding)

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

    results["success"] = results["processed_images"] > 0
    return results
