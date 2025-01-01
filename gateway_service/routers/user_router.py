from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import List, Optional
from services.user_service import UserService
from pydantic import BaseModel

router = APIRouter()


class UserResponse(BaseModel):
    id: str
    username: str
    identifier: str
    face_images_path: List[str]
    last_detection: Optional[str] = None


@router.get("/list")
async def list_users(service: UserService = Depends(UserService)):
    """Get list of all users"""
    return await service.get_all_users()


@router.get("/{user_id}")
async def get_user(user_id: str, service: UserService = Depends(UserService)):
    """Get user details by ID"""
    user = await service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}/images")
async def get_user_images(user_id: str, service: UserService = Depends(UserService)):
    """Get user's face images"""
    images = await service.get_user_images(user_id)
    if not images:
        raise HTTPException(status_code=404, detail="Images not found")
    return images


@router.post("/upload")
async def upload_user(
    identifier: str = Form(...),
    user_name: str = Form(...),
    files: List[UploadFile] = File(...),
    service: UserService = Depends(UserService),
):
    """Upload new user with multiple face images"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Validate file types
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {file.filename}. Only images are allowed",
                )

        result = await service.create_user(identifier, user_name, files)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create user")

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


@router.delete("/{user_id}")
async def delete_user(user_id: str, service: UserService = Depends(UserService)):
    """Delete user by ID"""
    success = await service.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True}


@router.put("/{user_id}")
async def update_user(
    user_id: str,
    user_name: str,
    identifier: str,
    service: UserService = Depends(UserService),
):
    """Update user information"""
    success = await service.update_user(user_id, user_name, identifier)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True}


@router.delete("/{user_id}/images/{image_path}")
async def delete_user_image(
    user_id: str, image_path: str, service: UserService = Depends(UserService)
):
    """Delete specific image from user"""
    try:
        # Delete from database service
        db_response = await service.delete_user_image(user_id, image_path)
        if not db_response:
            raise HTTPException(status_code=404, detail="Image not found")

        # Delete from face identify service
        identify_response = await service.delete_user_image_embedding(
            user_id, image_path
        )
        if not identify_response:
            # Log warning but don't fail if embedding deletion fails
            print(f"Warning: Failed to delete embedding for image {image_path}")

        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{user_id}/images/upload")
async def upload_user_images(
    user_id: str,
    files: List[UploadFile] = File(...),
    service: UserService = Depends(UserService),
):
    """Upload new images for existing user"""
    try:
        # Upload to database service first
        db_response = await service.upload_user_images(user_id, files)

        # Process face embeddings for new images
        identify_response = await service.process_user_images(
            user_id, db_response["added_images"]
        )

        return {
            "success": True,
            "processed_images": len(db_response["added_images"]),
            "failed_images": 0,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
