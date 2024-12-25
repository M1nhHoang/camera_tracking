from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
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
    identifier: str,
    user_name: str,
    file: UploadFile = File(...),
    service: UserService = Depends(UserService),
):
    """Upload new user with face image"""
    try:
        user_id = await service.create_user(identifier, user_name, file)
        return {"user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
