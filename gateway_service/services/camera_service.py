import aiohttp
from typing import Optional, Dict, List, AsyncGenerator
from config import settings
import logging

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self):
        self.database_url = settings.DATABASE_SERVICE_URL
        self.camera_inference_url = settings.CAMERA_INFERENCE_SERVICE_URL

    async def add_camera(self, camera_data: Dict) -> str:
        """Add new camera"""
        try:
            # First add to database
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.database_url}/cameras/add",
                    json={
                        "name": camera_data["name"],
                        "stream_url": camera_data["stream_url"],
                        "location": camera_data.get("location"),
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Database error: {error_text}")
                        raise Exception(
                            f"Failed to add camera to database: {error_text}"
                        )

                    db_result = await response.json()
                    camera_id = db_result["camera_id"]
                    logger.info(f"Camera added to database with ID: {camera_id}")

                # Then register with inference service
                camera_config = {
                    "camera_id": camera_id,
                    "name": camera_data["name"],
                    "stream_url": camera_data["stream_url"],
                    "location": camera_data.get("location"),
                    "optimal_width": camera_data.get("optimal_width", 640),
                    "optimal_height": camera_data.get("optimal_height", 480),
                    "conf_threshold": camera_data.get("conf_threshold", 0.7),
                }

                async with session.post(
                    f"{self.camera_inference_url}/cameras/add", json=camera_config
                ) as response:
                    if response.status != 200:
                        # Log error and try to rollback
                        error_text = await response.text()
                        logger.error(f"Inference service error: {error_text}")
                        try:
                            await self.delete_camera(camera_id)
                        except Exception as e:
                            logger.error(f"Rollback failed: {str(e)}")
                        raise Exception(
                            f"Failed to initialize camera in inference service: {error_text}"
                        )

                    logger.info(f"Camera initialized in inference service: {camera_id}")

                # Start the camera
                async with session.post(
                    f"{self.camera_inference_url}/cameras/{camera_id}/start"
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to start camera: {camera_id}")

                return camera_id

        except Exception as e:
            logger.error(f"Error adding camera: {str(e)}")
            raise

    async def get_cameras(self) -> List[Dict]:
        """Get list of all cameras"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.database_url}/cameras/list") as response:
                    if response.status == 200:
                        return await response.json()
                    logger.error(f"Error getting cameras: {await response.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error listing cameras: {str(e)}")
            return []

    async def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get camera details"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get camera details from database
                async with session.get(
                    f"{self.database_url}/cameras/{camera_id}"
                ) as response:
                    if response.status != 200:
                        return None
                    camera = await response.json()

                # Get camera status from inference service
                async with session.get(
                    f"{self.camera_inference_url}/cameras/{camera_id}/status"
                ) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        camera["status"] = status_data.get("status", "offline")
                    else:
                        camera["status"] = "offline"

                return camera

        except Exception as e:
            logger.error(f"Error getting camera: {str(e)}")
            return None

    async def delete_camera(self, camera_id: str) -> bool:
        """Delete camera"""
        try:
            async with aiohttp.ClientSession() as session:
                # First stop and delete from inference service
                try:
                    # Stop camera first
                    await session.post(
                        f"{self.camera_inference_url}/cameras/{camera_id}/stop"
                    )

                    # Then delete from inference service
                    async with session.delete(
                        f"{self.camera_inference_url}/cameras/{camera_id}"
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"Failed to delete from inference service: {await response.text()}"
                            )
                except Exception as e:
                    logger.warning(f"Error deleting from inference service: {str(e)}")

                # Then delete from database
                async with session.delete(
                    f"{self.database_url}/cameras/{camera_id}"
                ) as response:
                    success = response.status == 200
                    if not success:
                        logger.error(
                            f"Failed to delete from database: {await response.text()}"
                        )
                    return success
        except Exception as e:
            logger.error(f"Error deleting camera: {str(e)}")
            return False

    async def start_camera(self, camera_id: str) -> bool:
        """Start camera streaming"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.camera_inference_url}/cameras/{camera_id}/start"
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to start camera: {error_text}")
                        return False

                    # Update status in database
                    await session.put(
                        f"{self.database_url}/cameras/{camera_id}/status",
                        params={"status": "streaming"},
                    )
                    return True
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False

    async def stop_camera(self, camera_id: str) -> bool:
        """Stop camera streaming"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.camera_inference_url}/cameras/{camera_id}/stop"
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to stop camera: {error_text}")
                        return False

                    # Update status in database
                    await session.put(
                        f"{self.database_url}/cameras/{camera_id}/status",
                        params={"status": "offline"},
                    )
                    return True
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
            return False

    async def get_camera_stream(self, camera_id: str) -> AsyncGenerator[bytes, None]:
        """Get camera video stream"""
        try:
            async with aiohttp.ClientSession() as session:
                # First check if camera exists and is running
                camera = await self.get_camera(camera_id)
                if not camera:
                    raise Exception("Camera not found")
                if camera["status"] != "streaming":
                    raise Exception("Camera is not streaming")

                async with session.get(
                    f"{self.camera_inference_url}/cameras/{camera_id}/stream",
                    timeout=None,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to get camera stream: {error_text}")
                        raise Exception(f"Failed to get camera stream: {error_text}")

                    async for chunk in response.content.iter_chunked(1024):
                        yield chunk
        except Exception as e:
            logger.error(f"Error in camera stream: {str(e)}")
            raise
