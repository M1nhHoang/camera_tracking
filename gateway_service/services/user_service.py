import aiohttp
import os
from fastapi import UploadFile
from typing import List, Optional, Dict
from config import settings


class UserService:
    def __init__(self):
        self.database_url = settings.DATABASE_SERVICE_URL
        self.database_static_url = settings.DATABASE_STATIC_URL
        self.face_identify_url = settings.FACE_IDENTIFY_SERVICE_URL

    def _format_user_response(self, user: Dict) -> Dict:
        """Format user response to ensure consistent id fields and image paths"""
        if not user:
            return None

        # Ensure id is properly set
        if "_id" in user:
            user["id"] = str(user["_id"])
        elif "id" in user:
            user["_id"] = str(user["id"])

        # Format image paths to use database service URL
        if "face_images_path" in user and user["face_images_path"]:
            user["face_images_path"] = [
                f"{self.database_static_url}/{path}"
                for path in user["face_images_path"]
            ]
        else:
            user["face_images_path"] = []

        # Ensure other fields exist
        user.setdefault("created_at", "")
        user.setdefault("updated_at", None)
        user.setdefault("last_detection", None)

        return user

    async def get_all_users(self) -> List[Dict]:
        """Get all users from database service"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.database_url}/users/list") as response:
                if response.status == 200:
                    users = await response.json()
                    return [self._format_user_response(user) for user in users]
                return []

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user details by ID"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.database_url}/users/{user_id}") as response:
                if response.status == 200:
                    user = await response.json()
                    return self._format_user_response(user)
                return None

    async def get_user_images(self, user_id: str) -> List[str]:
        """Get user's face images"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.database_url}/users/{user_id}/images"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []

    async def create_user(
        self, identifier: str, user_name: str, files: List[UploadFile]
    ) -> Optional[Dict]:
        """Create new user with multiple face images"""
        try:
            # Create form data
            form_data = aiohttp.FormData()
            form_data.add_field("identifier", identifier)
            form_data.add_field("user_name", user_name)

            # Add each file to form data
            for file in files:
                content = await file.read()
                form_data.add_field(
                    "files",
                    content,
                    filename=file.filename,
                    content_type=file.content_type,
                )

            # Upload to face identify service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.face_identify_url}/face_upload",
                    data=form_data,
                    headers={"Accept": "application/json"},
                ) as response:
                    response_text = await response.text()
                    if response.status != 200:
                        print(f"Error response: {response_text}")
                        raise ValueError(f"Face upload failed: {response_text}")

                    try:
                        result = await response.json()
                        return result
                    except Exception as e:
                        print(f"Error parsing response: {response_text}")
                        raise ValueError(f"Invalid response format: {str(e)}")

        except Exception as e:
            print(f"Error in create_user: {str(e)}")
            raise ValueError(f"Failed to create user: {str(e)}")

    async def delete_user(self, user_id: str) -> bool:
        """Delete user and associated data"""
        try:
            async with aiohttp.ClientSession() as session:
                # Delete user from face identify service first
                async with session.delete(
                    f"{self.face_identify_url}/users/{user_id}"
                ) as response:
                    if response.status not in [
                        200,
                        404,
                    ]:  # Allow 404 as user might not have embeddings
                        error_text = await response.text()
                        print(
                            f"Error deleting from face identify service: {error_text}"
                        )
                        return False

                # If face identify delete succeeded, delete from database
                async with session.delete(
                    f"{self.database_url}/users/{user_id}"
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Error deleting from database: {error_text}")
                        return False

                return True

        except Exception as e:
            print(f"Error in delete_user: {str(e)}")
            return False

    async def update_user(self, user_id: str, user_data: dict) -> bool:
        """Update user information"""
        # Convert field names to match database service expectations
        update_data = {
            "username": user_data.get("username"),
            "identifier": user_data.get("identifier"),
        }
        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.database_url}/users/{user_id}", json=update_data
            ) as response:
                if response.status == 200:
                    return True
                return False

    async def delete_user_image(self, user_id: str, image_path: str) -> bool:
        """Delete specific image from database service"""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.database_url}/users/{user_id}/images/{image_path}"
            ) as response:
                return response.status == 200

    async def delete_user_image_embedding(self, user_id: str, image_path: str) -> bool:
        """Delete image embedding from face identify service"""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.face_identify_url}/users/{user_id}/images/{image_path}"
            ) as response:
                return response.status == 200

    async def upload_user_images(self, user_id: str, files: List[UploadFile]) -> Dict:
        """Upload and process images for user"""
        try:
            # First get user details
            user = await self.get_user_by_id(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")

            # Upload to database service
            async with aiohttp.ClientSession() as session:
                # Database service upload
                db_form = aiohttp.FormData()
                for file in files:
                    db_form.add_field(
                        "files",
                        await file.read(),
                        filename=file.filename,
                        content_type=file.content_type,
                    )
                    # Reset file cursor for next use
                    await file.seek(0)

                async with session.post(
                    f"{self.database_url}/users/{user_id}/images/upload", data=db_form
                ) as response:
                    if response.status != 200:
                        raise ValueError(await response.text())
                    db_result = await response.json()

                # Face identify service upload
                identify_form = aiohttp.FormData()
                identify_form.add_field("identifier", user["identifier"])
                identify_form.add_field("user_name", user["username"])
                for file in files:
                    identify_form.add_field(
                        "files",
                        await file.read(),
                        filename=file.filename,
                        content_type=file.content_type,
                    )

                async with session.post(
                    f"{self.face_identify_url}/users/{user_id}/images/upload",
                    data=identify_form,
                ) as response:
                    if response.status != 200:
                        # Log warning but don't fail completely
                        print(
                            f"Warning: Face identify processing failed: {await response.text()}"
                        )
                    identify_result = await response.json()

                return {
                    "success": True,
                    "database_result": db_result,
                    "identify_result": identify_result,
                }

        except Exception as e:
            raise ValueError(f"Error uploading images: {str(e)}")

    async def process_user_images(self, user_id: str, image_paths: List[str]) -> Dict:
        """Process images with face identify service"""
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("user_id", user_id)
            for path in image_paths:
                form.add_field("image_paths", path)

            async with session.post(
                f"{self.face_identify_url}/process_images", data=form
            ) as response:
                if response.status != 200:
                    # Log error but don't fail
                    print(f"Warning: Failed to process images: {await response.text()}")
                return await response.json()
