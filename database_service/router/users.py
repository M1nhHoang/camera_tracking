# router/users.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from service.users import UserService

router = APIRouter()


# Pydantic models for request validation
class UserCreate(BaseModel):
    username: str
    identifier: str
    face_images: Optional[List[str]] = None


class UserUpdate(BaseModel):
    username: Optional[str] = None
    identifier: Optional[str] = None
    face_images: Optional[List[str]] = None


@router.post("/create")
async def create_user(
    user_data: UserCreate, user_service: UserService = Depends(UserService)
):
    """Create a new user"""
    try:
        user_id = user_service.create_user(
            username=user_data.username,
            identifier=user_data.identifier,
            face_images=user_data.face_images,
        )
        return JSONResponse(
            status_code=201,
            content={"message": "User created successfully", "user_id": user_id},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_user(user_id: str, user_service: UserService = Depends(UserService)):
    """Get user by ID"""
    try:
        user = user_service.get_user(user_id)
        return JSONResponse(status_code=200, content=user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_all_users(
    skip: int = 0, limit: int = 100, user_service: UserService = Depends(UserService)
):
    """Get all users with pagination"""
    try:
        users = user_service.get_all_users(skip=skip, limit=limit)
        return JSONResponse(
            status_code=200,
            content={"users": users, "total": len(users), "skip": skip, "limit": limit},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}")
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    user_service: UserService = Depends(UserService),
):
    """Update user information"""
    try:
        # Convert Pydantic model to dict and remove None values
        update_data = user_data.dict(exclude_unset=True)
        success = user_service.update_user(user_id, update_data)
        if success:
            return JSONResponse(
                status_code=200, content={"message": "User updated successfully"}
            )
        raise HTTPException(status_code=400, detail="Update failed")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def delete_user(user_id: str, user_service: UserService = Depends(UserService)):
    """Delete a user"""
    try:
        success = user_service.delete_user(user_id)
        if success:
            return JSONResponse(
                status_code=200, content={"message": "User deleted successfully"}
            )
        raise HTTPException(status_code=400, detail="Delete failed")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Existing endpoint
@router.post("/user_update")
async def user_update(
    user_data: dict = {
        "identifier": "",
        "user_name": "",
        "face_images": [],
    },
    user_service: UserService = Depends(UserService),
):
    # init
    identifier = user_data["identifier"]
    user_name = user_data["user_name"]
    face_images = user_data["face_images"]

    # tracking
    user_id, face_images = user_service.user_update(user_name, identifier, face_images)

    if user_id:
        return JSONResponse(
            status_code=200,
            content={"user_id": user_id, "face_image_path": face_images[0]},
        )

    return JSONResponse(status_code=500, content={"message": "Update failed"})
