import aiohttp
import os
from fastapi import UploadFile
from typing import List, Optional, Dict
from config import settings


class UserService:
    def __init__(self):
        self.database_url = settings.DATABASE_SERVICE_URL
        self.face_identify_url = settings.FACE_IDENTIFY_SERVICE_URL

    async def get_all_users(self) -> List[Dict]:
        """Get all users from database service"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.database_url}/users/list") as response:
                if response.status == 200:
                    return await response.json()
                return []

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user details by ID"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.database_url}/users/{user_id}") as response:
                if response.status == 200:
                    return await response.json()
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
    ) -> str:
        """Create new user with face image"""
        # First, save the image to face identify service
        form_data = aiohttp.FormData()
        form_data.add_field("identifier", identifier)
        form_data.add_field("user_name", user_name)

        # Read file content
        content = await file.read()
        form_data.add_field(
            "file", content, filename=file.filename, content_type=file.content_type
        )

        async with aiohttp.ClientSession() as session:
            # Upload to face identify service
            async with session.post(
                f"{self.face_identify_url}/face_upload", data=form_data
            ) as response:
                if response.status != 200:
                    raise Exception("Failed to upload face image")
                result = await response.json()
                return result.get("user_id")

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

    async def update_user(self, user_id: str, user_name: str, identifier: str) -> bool:
        """Update user information"""
        data = {"user_name": user_name, "identifier": identifier}

        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.database_url}/users/{user_id}", json=data
            ) as response:
                return response.status == 200
