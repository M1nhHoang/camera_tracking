from database import MongoDBManager
from bson.objectid import ObjectId
from bson.json_util import dumps, loads
import json
import time


class CameraService:
    def __init__(self):
        self.db_manager = MongoDBManager(collection_name="cameras")

    def _serialize_document(self, document):
        """Convert MongoDB document to JSON serializable dict"""
        if document:
            document["id"] = str(document.pop("_id"))
            return document
        return None

    def add_camera(self, name: str, stream_url: str, location: str = None):
        """Add a new camera"""
        camera = {
            "name": name,
            "stream_url": stream_url,
            "location": location,
            "status": "offline",
            "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
            "last_active": None,
        }
        result = self.db_manager.insert_one(camera)
        return str(result.inserted_id)

    def get_camera(self, camera_id: str):
        """Get camera by ID"""
        if isinstance(camera_id, str):
            camera_id = ObjectId(camera_id)
        camera = self.db_manager.find_one({"_id": camera_id})
        return self._serialize_document(camera)

    def list_cameras(self, status=None):
        """Get list of all cameras"""
        query = {}
        if status:
            query["status"] = status
        cameras = self.db_manager.find_all(query)
        return [self._serialize_document(camera) for camera in cameras]

    def update_camera(self, camera_id: str, update_data: dict):
        """Update camera information"""
        if isinstance(camera_id, str):
            camera_id = ObjectId(camera_id)
        result = self.db_manager.update_one({"_id": camera_id}, update_data)
        return result.modified_count > 0

    def delete_camera(self, camera_id: str):
        """Delete a camera"""
        if isinstance(camera_id, str):
            camera_id = ObjectId(camera_id)
        result = self.db_manager.delete_one({"_id": camera_id})
        return result.deleted_count > 0

    def update_camera_status(self, camera_id: str, status: str):
        """Update camera status"""
        if isinstance(camera_id, str):
            camera_id = ObjectId(camera_id)
        update_data = {
            "status": status,
            "last_active": (
                time.strftime("%d-%m-%Y %H:%M:%S") if status == "streaming" else None
            ),
        }
        result = self.db_manager.update_one({"_id": camera_id}, update_data)
        return result.modified_count > 0

    def get_active_cameras(self):
        """Get list of active cameras"""
        cameras = self.db_manager.find_all({"status": "streaming"})
        return [self._serialize_document(camera) for camera in cameras]
