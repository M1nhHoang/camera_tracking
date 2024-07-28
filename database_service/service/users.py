from database import MongoDBManager
from utils import base64_to_image, save_image_to_folder
from bson.objectid import ObjectId

import cv2
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
