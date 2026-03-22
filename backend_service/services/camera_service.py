import time
import aiohttp
import logging
from typing import Optional, Dict, List, AsyncGenerator
from bson import ObjectId

from config import settings
from database import MongoDB

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self):
        self.collection = MongoDB.get_collection("cameras")
        self.detection_url = settings.DETECTION_SERVICE_URL

    async def add_camera(self, camera_data: Dict) -> str:
        """Add new camera to MongoDB + register with detection_service."""
        try:
            # Save to MongoDB
            camera = {
                "name": camera_data["name"],
                "stream_url": camera_data["stream_url"],
                "location": camera_data.get("location"),
                "status": "offline",
                "created_at": time.strftime("%d-%m-%Y %H:%M:%S"),
                "last_active": None,
            }
            result = self.collection.insert_one(camera)
            camera_id = str(result.inserted_id)

            # Register with detection service
            camera_config = {
                "camera_id": camera_id,
                "name": camera_data["name"],
                "stream_url": camera_data["stream_url"],
                "location": camera_data.get("location"),
                "optimal_width": camera_data.get("optimal_width", 640),
                "optimal_height": camera_data.get("optimal_height", 480),
                "conf_threshold": camera_data.get("conf_threshold", 0.7),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.detection_url}/cameras/add", json=camera_config
                ) as response:
                    if response.status != 200:
                        # Rollback DB insert
                        self.collection.delete_one({"_id": result.inserted_id})
                        raise Exception(f"Detection service error: {await response.text()}")

                # Start camera
                async with session.post(
                    f"{self.detection_url}/cameras/{camera_id}/start"
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to start camera: {camera_id}")

            return camera_id

        except Exception as e:
            logger.error(f"Error adding camera: {str(e)}")
            raise

    async def get_cameras(self) -> List[Dict]:
        """Get all cameras from MongoDB."""
        cameras = list(self.collection.find({}))
        return [self._serialize(cam) for cam in cameras]

    async def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get camera details + live status from detection_service."""
        try:
            camera = self.collection.find_one({"_id": ObjectId(camera_id)})
            if not camera:
                return None

            result = self._serialize(camera)

            # Get live status from detection service
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.detection_url}/cameras/{camera_id}/status"
                ) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        result["status"] = status_data.get("status", "offline")
                    else:
                        result["status"] = "offline"

            return result
        except Exception as e:
            logger.error(f"Error getting camera: {str(e)}")
            return None

    async def delete_camera(self, camera_id: str) -> bool:
        """Delete camera from detection_service + MongoDB."""
        try:
            async with aiohttp.ClientSession() as session:
                try:
                    await session.post(f"{self.detection_url}/cameras/{camera_id}/stop")
                    await session.delete(f"{self.detection_url}/cameras/{camera_id}")
                except Exception as e:
                    logger.warning(f"Error deleting from detection service: {str(e)}")

            result = self.collection.delete_one({"_id": ObjectId(camera_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting camera: {str(e)}")
            return False

    async def start_camera(self, camera_id: str) -> bool:
        """Start camera streaming."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.detection_url}/cameras/{camera_id}/start"
                ) as response:
                    if response.status != 200:
                        return False

            self.collection.update_one(
                {"_id": ObjectId(camera_id)},
                {"$set": {"status": "streaming", "last_active": time.strftime("%d-%m-%Y %H:%M:%S")}},
            )
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False

    async def stop_camera(self, camera_id: str) -> bool:
        """Stop camera streaming."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.detection_url}/cameras/{camera_id}/stop"
                ) as response:
                    if response.status != 200:
                        return False

            self.collection.update_one(
                {"_id": ObjectId(camera_id)},
                {"$set": {"status": "offline"}},
            )
            return True
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
            return False

    async def get_camera_stream(self, camera_id: str) -> AsyncGenerator[bytes, None]:
        """Proxy camera video stream from detection_service."""
        camera = await self.get_camera(camera_id)
        if not camera:
            raise Exception("Camera not found")
        if camera["status"] != "streaming":
            raise Exception("Camera is not streaming")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.detection_url}/cameras/{camera_id}/stream",
                timeout=None,
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get stream: {await response.text()}")
                async for chunk in response.content.iter_chunked(1024):
                    yield chunk

    @staticmethod
    def _serialize(doc: Dict) -> Dict:
        if doc and "_id" in doc:
            doc["id"] = str(doc.pop("_id"))
        return doc
