from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from service.users import UserService

router = APIRouter()


class UserBase(BaseModel):
    username: str
    identifier: str


class UserCreate(UserBase):
    face_images: Optional[List[str]] = None


class UserUpdate(BaseModel):
    username: Optional[str] = None
    identifier: Optional[str] = None
    face_images: Optional[List[str]] = None


class UserResponse(UserBase):
    id: str
    face_images_path: Optional[List[str]] = (
        []
    )  # Make face_images_path optional with default empty list
    created_at: str
    updated_at: Optional[str] = None
    last_detection: Optional[str] = None


@router.post("/add", response_model=dict)
async def create_user(user: UserCreate, service: UserService = Depends(UserService)):
    """Create a new user"""
    try:
        user_id = service.create_user(
            username=user.username,
            identifier=user.identifier,
            face_images=user.face_images,
        )
        return {"user_id": user_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list", response_model=List[UserResponse])
async def list_users(
    skip: int = 0, limit: int = 100, service: UserService = Depends(UserService)
):
    """Get list of all users excluding unknown user"""
    try:
        # Get all users
        users = service.get_all_users(skip=skip, limit=limit)
        # Filter out unknown user and format response
        formatted_users = []
        for user in users:
            if user.get("identifier") != "unknown":
                # Get the ID whether it's in _id or id field
                user_id = str(user.get("_id", user.get("id", "")))
                formatted_user = {
                    "id": user_id,
                    "_id": user_id,
                    "username": user.get("username", ""),
                    "identifier": user.get("identifier", ""),
                    "face_images_path": user.get("face_images_path", []),
                    "created_at": user.get("created_at", ""),
                    "updated_at": user.get("updated_at"),
                    "last_detection": user.get("last_detection"),
                }
                formatted_users.append(formatted_user)

        return formatted_users
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, service: UserService = Depends(UserService)):
    """Get user details"""
    try:
        user = service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.get("identifier") == "unknown":
            raise HTTPException(status_code=404, detail="Cannot access unknown user")

        # Ensure face_images_path exists
        if "face_images_path" not in user:
            user["face_images_path"] = []

        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{user_id}/images", response_model=List[str])
async def get_user_images(user_id: str, service: UserService = Depends(UserService)):
    """Get user's face images"""
    try:
        # Check if user is unknown
        user = service.get_user(user_id)
        if user and user.get("identifier") == "unknown":
            raise HTTPException(
                status_code=404, detail="Cannot access unknown user images"
            )

        images = service.get_user_images(user_id)
        return images
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{user_id}")
async def update_user(
    user_id: str, user: dict, service: UserService = Depends(UserService)
):
    """Update user information"""
    try:
        # Check if user exists
        existing_user = service.get_user(user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check if user is unknown
        if existing_user.get("identifier") == "unknown":
            raise HTTPException(status_code=400, detail="Cannot update unknown user")

        # Remove None values and empty strings
        update_data = {k: v for k, v in user.items() if v is not None and v != ""}

        # Validate required fields if they are being updated
        if "identifier" in update_data and not update_data["identifier"]:
            raise HTTPException(status_code=400, detail="Identifier cannot be empty")
        if "username" in update_data and not update_data["username"]:
            raise HTTPException(status_code=400, detail="Username cannot be empty")

        # Check if new identifier already exists
        if "identifier" in update_data and update_data[
            "identifier"
        ] != existing_user.get("identifier"):
            if service.db_manager.find_one({"identifier": update_data["identifier"]}):
                raise HTTPException(status_code=400, detail="Identifier already exists")

        success = service.update_user(user_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update user")

        # Return updated user data
        updated_user = service.get_user(user_id)
        return updated_user

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def delete_user(user_id: str, service: UserService = Depends(UserService)):
    """Delete user"""
    try:
        # Check if user is unknown
        existing_user = service.get_user(user_id)
        if existing_user and existing_user.get("identifier") == "unknown":
            raise HTTPException(status_code=400, detail="Cannot delete unknown user")

        success = service.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/user_update")
async def user_update(
    data: dict = {"identifier": "", "user_name": "", "face_images": []},
    service: UserService = Depends(UserService),
):
    """Update or create user with face images"""
    try:
        # Check if trying to update unknown user
        if data["identifier"] == "unknown":
            raise HTTPException(status_code=400, detail="Cannot update unknown user")

        # Convert face images to list if it's not already
        face_images = data.get("face_images", [])
        if not isinstance(face_images, list):
            face_images = [face_images]

        # First try to create a new user
        try:
            user_id = service.create_user(
                username=data["user_name"],
                identifier=data["identifier"],
                face_images=face_images,
            )

            # Get face image path from newly created user
            images = service.get_user_images(user_id)
            face_image_path = images[len(images) - 1] if images else None

            return {"user_id": user_id, "face_image_path": face_image_path}

        except ValueError as e:
            # If user exists, update instead
            existing_user = service.db_manager.find_one(
                {"identifier": data["identifier"]}
            )
            if not existing_user:
                raise HTTPException(
                    status_code=400, detail=f"User create/update failed: {str(e)}"
                )

            user_id = str(existing_user["_id"])
            update_data = {
                "username": data["user_name"],
                "face_images": face_images,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Update existing user
            updated = service.update_user(user_id, update_data)
            if not updated:
                raise HTTPException(
                    status_code=400, detail="Failed to update existing user"
                )

            # Get latest face image path
            images = service.get_user_images(user_id)
            face_image_path = images[len(images) - 1] if images else None

            return {"user_id": user_id, "face_image_path": face_image_path}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in user_update: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Failed to create or update user: {str(e)}"
        )


@router.delete("/{user_id}/images/{image_path}")
async def delete_user_image(
    user_id: str, image_path: str, service: UserService = Depends(UserService)
):
    """Delete a specific image from user's face images"""
    try:
        success = await service.delete_user_image(user_id, image_path)
        if not success:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{user_id}/images/upload")
async def upload_user_images(
    user_id: str,
    files: List[UploadFile] = File(...),
    service: UserService = Depends(UserService),
):
    """Add new images to existing user"""
    try:
        # Process and save new images
        new_image_paths = []
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not an image"
                )

            content = await file.read()
            new_path = service.save_user_image(content)
            new_image_paths.append(new_path)

        # Update user's image list
        success = await service.add_user_images(user_id, new_image_paths)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        return {"success": True, "added_images": new_image_paths}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
