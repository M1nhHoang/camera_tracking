from database import RedisManager
from utils import base64_to_image, save_image_to_folder
import time
import uuid


class UserService:
    def __init__(self):
        self.redis_manager = RedisManager()
        self.static_files = "static_files"

    def get_unkow_user_id(self):
        unknown_user = self.redis_manager.find_one("users", {"username": "unknown"})
        if not unknown_user:
            new_user = {
                "identifier": "unknown",
                "username": "unknown",
                "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
            }
            return self.redis_manager.insert_one("users", new_user)
        return unknown_user["_id"]

    def get_user_name_by_id(self, user_id):
        user = self.redis_manager.get_by_id("users", user_id)
        if not user:
            return "Not found user"
        return user["username"]

    def user_update(self, username, identifier, face_images):
        if not username or not identifier or not face_images:
            return False

        face_images = [
            save_image_to_folder(base64_to_image(image), self.static_files)
            for image in face_images
        ]

        user = self.redis_manager.find_one("users", {"identifier": identifier})
        if not user:
            new_user = {
                "username": username,
                "identifier": identifier,
                "face_images_path": face_images,
                "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
            }
            user_id = self.redis_manager.insert_one("users", new_user)
        else:
            user_id = user["_id"]
            self.redis_manager.update_one(
                "users",
                {"_id": user_id},
                {"$push": {"face_images_path": {"$each": face_images}}},
            )

        return user_id, face_images
