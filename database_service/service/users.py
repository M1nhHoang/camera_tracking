from database import MongoDBManager
from utils import base64_to_image, save_image_to_folder
from bson.objectid import ObjectId
from datetime import datetime
from typing import List, Optional, Dict


class UserService:
    def __init__(self):
        self.db_manager = MongoDBManager(collection_name="users")
        self.static_files = "static_files"

        # Create indexes
        self.db_manager.get_collection().create_index("identifier", unique=True)
        self.db_manager.get_collection().create_index("username")

    def _format_user(self, user: Dict) -> Dict:
        """Format user document for API response"""
        if not user:
            return None
        # Make a copy of the user dict to avoid modifying the original
        formatted_user = {}
        # Get the ID whether it's in _id or id field
        user_id = str(user.get("_id", user.get("id", "")))

        formatted_user.update(
            {
                "id": user_id,
                "_id": user_id,
                "username": user.get("username", ""),
                "identifier": user.get("identifier", ""),
                "face_images_path": user.get("face_images_path", []),
                "created_at": user.get("created_at", ""),
                "updated_at": user.get("updated_at"),
                "last_detection": user.get("last_detection"),
            }
        )

        return formatted_user

    def create_user(
        self, username: str, identifier: str, face_images: Optional[List[str]] = None
    ) -> str:
        """Create a new user"""
        try:
            # Validate input
            if not username or not identifier:
                raise ValueError("Username and identifier are required")

            # Check if user already exists
            existing_user = self.db_manager.find_one({"identifier": identifier})
            if existing_user:
                raise ValueError(f"User with identifier {identifier} already exists")

            # Process face images if provided
            face_image_paths = []
            if face_images:
                for image in face_images:
                    try:
                        image_data = base64_to_image(image)
                        image_path = save_image_to_folder(image_data, self.static_files)
                        face_image_paths.append(image_path)
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                        continue

            # Create user document
            new_user = {
                "username": username,
                "identifier": identifier,
                "face_images_path": face_image_paths,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_detection": None,
            }

            result = self.db_manager.insert_one(new_user)
            if not result:
                raise ValueError("Failed to insert user into database")

            return str(result.inserted_id)

        except Exception as e:
            print(f"Error in create_user: {str(e)}")
            raise ValueError(f"Error creating user: {str(e)}")

    def update_user(self, user_id: str, update_data: Dict) -> bool:
        """Update user information"""
        try:
            user_id_obj = ObjectId(user_id)

            # Verify user exists
            existing_user = self.db_manager.find_one({"_id": user_id_obj})
            if not existing_user:
                raise ValueError(f"User with ID {user_id} not found")

            # Create update document
            update_document = {}

            # Handle basic fields
            if "username" in update_data:
                update_document["username"] = update_data["username"]
            if "identifier" in update_data:
                # Check if new identifier already exists
                if update_data["identifier"] != existing_user["identifier"]:
                    existing = self.db_manager.find_one(
                        {"identifier": update_data["identifier"]}
                    )
                    if existing:
                        raise ValueError("Identifier already exists")
                update_document["identifier"] = update_data["identifier"]

            # Process face images if provided
            if "face_images" in update_data:
                face_images = update_data["face_images"]
                face_image_paths = []
                for image in face_images:
                    try:
                        image_data = base64_to_image(image)
                        image_path = save_image_to_folder(image_data, self.static_files)
                        face_image_paths.append(image_path)
                    except Exception as e:
                        print(f"Error processing image: {str(e)}")
                        continue

                # Append new paths to existing ones
                existing_paths = existing_user.get("face_images_path", [])
                update_document["face_images_path"] = existing_paths + face_image_paths

            # Add updated timestamp
            update_document["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Perform update
            result = self.db_manager.update_one({"_id": user_id_obj}, update_document)
            return result.modified_count > 0

        except Exception as e:
            print(f"Error in update_user: {str(e)}")
            raise ValueError(f"Error updating user: {str(e)}")

    def get_user(self, user_id: str) -> Dict:
        """Get user by ID"""
        try:
            user_id_obj = ObjectId(user_id)
            user = self.db_manager.find_one({"_id": user_id_obj})
            return self._format_user(user)
        except Exception as e:
            raise ValueError(f"Error getting user: {str(e)}")

    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """Get all users with pagination"""
        try:
            users = self.db_manager.find_all({})
            formatted_users = [
                self._format_user(user) for user in users[skip : skip + limit]
            ]
            return formatted_users
        except Exception as e:
            raise ValueError(f"Error getting users: {str(e)}")

    def update_user(self, user_id: str, update_data: Dict) -> bool:
        """Update user information"""
        try:
            user_id_obj = ObjectId(user_id)

            # Verify user exists
            existing_user = self.db_manager.find_one({"_id": user_id_obj})
            if not existing_user:
                raise ValueError(f"User with ID {user_id} not found")

            # Create update document
            update_document = {}

            # Handle basic fields
            if "username" in update_data:
                update_document["username"] = update_data["username"]
            if "identifier" in update_data:
                update_document["identifier"] = update_data["identifier"]

            # Process face images if provided
            if "face_images" in update_data:
                face_images = update_data["face_images"]
                face_image_paths = []
                for image in face_images:
                    image_data = base64_to_image(image)
                    image_path = save_image_to_folder(image_data, self.static_files)
                    face_image_paths.append(image_path)
                update_document["face_images_path"] = face_image_paths

            # Add updated timestamp
            update_document["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Perform update
            result = self.db_manager.update_one({"_id": user_id_obj}, update_document)

            if result.modified_count > 0:
                # Return updated user data
                updated_user = self.db_manager.find_one({"_id": user_id_obj})
                return self._format_user(updated_user)
            return None

        except Exception as e:
            raise ValueError(f"Error updating user: {str(e)}")

    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            user_id_obj = ObjectId(user_id)

            # Verify user exists and is not the unknown user
            existing_user = self.db_manager.find_one({"_id": user_id_obj})
            if not existing_user:
                raise ValueError(f"User with ID {user_id} not found")
            if existing_user.get("identifier") == "unknown":
                raise ValueError("Cannot delete the unknown user")

            result = self.db_manager.delete_one({"_id": user_id_obj})
            return result.deleted_count > 0

        except Exception as e:
            raise ValueError(f"Error deleting user: {str(e)}")

    def get_user_images(self, user_id: str) -> List[str]:
        """Get user's face images"""
        try:
            user_id_obj = ObjectId(user_id)
            user = self.db_manager.find_one({"_id": user_id_obj})
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
            return user.get("face_images_path", [])
        except Exception as e:
            raise ValueError(f"Error getting user images: {str(e)}")

    def get_unknown_user_id(self) -> str:
        """Get or create unknown user ID"""
        try:
            user = self.db_manager.find_one({"identifier": "unknown"})
            if not user:
                new_user = {
                    "identifier": "unknown",
                    "username": "Unknown",
                    "face_images_path": [],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                result = self.db_manager.insert_one(new_user)
                return str(result.inserted_id)
            return str(user["_id"])
        except Exception as e:
            raise ValueError(f"Error getting unknown user: {str(e)}")

    def update_last_detection(self, user_id: str, timestamp: str) -> bool:
        """Update user's last detection time"""
        try:
            user_id_obj = ObjectId(user_id)
            result = self.db_manager.update_one(
                {"_id": user_id_obj}, {"last_detection": timestamp}
            )
            return result.modified_count > 0
        except Exception as e:
            raise ValueError(f"Error updating last detection: {str(e)}")
