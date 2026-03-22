import os
import logging
from typing import List, Optional, Dict
from datetime import datetime
from bson import ObjectId

from config import settings
from database import MongoDB

logger = logging.getLogger(__name__)

# TODO: User face upload workflow
# When user uploads face images, backend needs to:
#   1. Save images to static_files/
#   2. Call detection_service (borrow YOLO model) for face validation
#   3. Call recognition_service (gRPC) for embedding generation
#   4. Store embedding in ChromaDB via recognition_service
# This flow will be implemented after the detection pipeline is tested.


class UserService:
    """User management — directly accesses MongoDB."""

    def __init__(self):
        self.collection = MongoDB.get_collection("users")
        self.static_dir = settings.STATIC_DIR

    def _format_user(self, user: Dict) -> Optional[Dict]:
        if not user:
            return None
        user_id = str(user.get("_id", ""))
        return {
            "id": user_id,
            "_id": user_id,
            "username": user.get("username", ""),
            "identifier": user.get("identifier", ""),
            "face_images_path": user.get("face_images_path", []),
            "created_at": user.get("created_at", ""),
            "updated_at": user.get("updated_at"),
            "last_detection": user.get("last_detection"),
        }

    async def get_all_users(self) -> List[Dict]:
        """Get all users excluding unknown."""
        users = list(self.collection.find({"identifier": {"$ne": "unknown"}}))
        return [self._format_user(u) for u in users]

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        user = self.collection.find_one({"_id": ObjectId(user_id)})
        return self._format_user(user)

    async def update_user(self, user_id: str, user_data: dict) -> bool:
        update = {k: v for k, v in user_data.items() if v is not None}
        update["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = self.collection.update_one(
            {"_id": ObjectId(user_id)}, {"$set": update}
        )
        return result.modified_count > 0

    async def delete_user(self, user_id: str) -> bool:
        """Delete user from MongoDB."""
        # TODO: Also delete embeddings from ChromaDB via recognition_service gRPC
        result = self.collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0

    async def delete_user_image(self, user_id: str, image_path: str) -> bool:
        """Delete specific image from user."""
        user = self.collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return False

        current = user.get("face_images_path", [])
        if image_path not in current:
            return False

        updated = [img for img in current if img != image_path]
        result = self.collection.update_one(
            {"_id": ObjectId(user_id)}, {"$set": {"face_images_path": updated}}
        )

        if result.modified_count > 0:
            full_path = os.path.join(self.static_dir, image_path)
            if os.path.exists(full_path):
                os.remove(full_path)
            # TODO: Also delete embedding from ChromaDB
            return True
        return False
