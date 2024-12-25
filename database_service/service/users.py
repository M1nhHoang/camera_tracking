# service/users.py

from database import MongoDBManager
from utils import base64_to_image, save_image_to_folder
from bson.objectid import ObjectId
from typing import List, Optional
import time


class UserService:
    def __init__(self):
        self.db_manager = MongoDBManager(collection_name="users")
        self.static_files = "static_files"
        self.traking_id_cache = {}

        # Create index
        self.db_manager.get_collection().create_index("identifier", unique=True)

    def get_unkow_user_id(self):

        user = self.db_manager.find_one({"username": "unknown"})
        if not user:
            new_user = {
                "identifier": "unknown",
                "username": "unknown",
                "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
            }
            result = self.db_manager.insert_one(new_user)
            return str(result.inserted_id)
        return str(user["_id"])

    def get_user_name_by_id(self, user_id):
        if isinstance(user_id, ObjectId):
            user = self.db_manager.find_one({"_id": user_id})
        else:
            user = self.db_manager.find_one({"_id": ObjectId(user_id)})

        if not user:
            return "Not found user"
        return user["username"]

    def create_user(
        self, username: str, identifier: str, face_images: Optional[List[str]] = None
    ) -> str:
        """Create a new user"""
        if not username or not identifier:
            raise ValueError("Username and identifier are required")

        # Check if user already exists
        existing_user = self.db_manager.find_one({"identifier": identifier})
        if existing_user:
            raise ValueError(f"User with identifier {identifier} already exists")

        # Process face images if provided
        processed_images = []
        if face_images:
            processed_images = [
                save_image_to_folder(base64_to_image(image), self.static_files)
                for image in face_images
            ]

        new_user = {
            "username": username,
            "identifier": identifier,
            "face_images_path": processed_images,
            "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
            "updated_at": time.strftime("%d-%m-%Y %H:%M:%S"),
        }

        result = self.db_manager.insert_one(new_user)
        return str(result.inserted_id)

    def update_user(self, user_id: str, update_data: dict) -> bool:
        """Update user information"""
        try:
            user_id_obj = ObjectId(user_id)
        except:
            raise ValueError("Invalid user ID format")

        # Check if user exists
        existing_user = self.db_manager.find_one({"_id": user_id_obj})
        if not existing_user:
            raise ValueError(f"User with ID {user_id} not found")

        # Process face images if provided
        if "face_images" in update_data:
            face_images = update_data.pop("face_images")
            processed_images = [
                save_image_to_folder(base64_to_image(image), self.static_files)
                for image in face_images
            ]
            update_data["face_images_path"] = processed_images

        # Add updated timestamp
        update_data["updated_at"] = time.strftime("%d-%m-%Y %H:%M:%S")

        # Update user
        result = self.db_manager.update_one({"_id": user_id_obj}, update_data)
        return bool(result.modified_count)

    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            user_id_obj = ObjectId(user_id)
        except:
            raise ValueError("Invalid user ID format")

        # Check if user exists
        existing_user = self.db_manager.find_one({"_id": user_id_obj})
        if not existing_user:
            raise ValueError(f"User with ID {user_id} not found")

        # Don't allow deletion of unknown user
        if existing_user.get("identifier") == "unknown":
            raise ValueError("Cannot delete the unknown user")

        result = self.db_manager.delete_one({"_id": user_id_obj})
        return bool(result.deleted_count)

    def get_user(self, user_id: str) -> dict:
        """Get user by ID"""
        try:
            user_id_obj = ObjectId(user_id)
        except:
            raise ValueError("Invalid user ID format")

        user = self.db_manager.find_one({"_id": user_id_obj})
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        return user

    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[dict]:
        """Get all users with pagination"""
        users = self.db_manager.find_all({})

        # Convert ObjectId to string
        for user in users:
            user["_id"] = str(user["_id"])

        # Apply pagination
        return users[skip : skip + limit]

    def user_update(self, username, identifier, face_images):
        if not username or not identifier or not face_images:
            return False

        # conver face images
        face_images = [
            save_image_to_folder(base64_to_image(image), self.static_files)
            for image in face_images
        ]

        user = self.db_manager.find_one({"identifier": identifier})
        if not user:
            new_user = {
                "username": username,
                "identifier": identifier,
                "face_images_path": face_images,
                "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
            }
            result = self.db_manager.insert_one(new_user)
            return str(result.inserted_id), face_images  # return user id
        else:
            for face_image in face_images:
                self.db_manager.get_collection().update_one(
                    {"identifier": user["identifier"]},
                    {"$push": {"face_images_path": face_image}},
                )

        return str(user["_id"]), face_images
