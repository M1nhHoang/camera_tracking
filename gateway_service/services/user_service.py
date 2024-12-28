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
        self, identifier: str, user_name: str, file: UploadFile
    ) -> Optional[Dict]:
        """Create new user with face image"""
        try:
            # Read file content
            content = await file.read()

            # Create form data
            form_data = aiohttp.FormData()
            form_data.add_field("identifier", identifier)
            form_data.add_field("user_name", user_name)
            form_data.add_field(
                "file", content, filename=file.filename, content_type=file.content_type
            )

            # Upload to face identify service
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.face_identify_url}/face_upload", data=form_data
                ) as response:
                    if response.status != 200:
                        error_detail = await response.text()
                        raise ValueError(f"Face upload failed: {error_detail}")

                    result = await response.json()
                    return result

        except Exception as e:
            print(f"Error in create_user: {str(e)}")
            raise ValueError(f"Failed to create user: {str(e)}")

    async def delete_user(self, user_id: str) -> bool:
        """Delete user and associated data"""
        async with aiohttp.ClientSession() as session:
            # Delete user from database
            async with session.delete(
                f"{self.database_url}/users/{user_id}"
            ) as response:
                if response.status != 200:
                    return False

            # Delete user embeddings from face identify service
            async with session.delete(
                f"{self.face_identify_url}/embeddings/{user_id}"
            ) as response:
                return response.status == 200

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
