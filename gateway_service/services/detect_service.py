import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
from config import settings


class DetectionService:
    def __init__(self):
        self.database_url = settings.DATABASE_SERVICE_URL

    async def get_detections(
        self,
        page: int = 1,
        per_page: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        user_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        sort_by: Optional[str] = "time_stamp",
        sort_order: Optional[str] = "desc",
    ) -> Dict:
        """Get filtered list of detections with pagination"""
        # Build query parameters
        params = {
            "page": page,
            "per_page": per_page,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }

        # Add optional filters
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if user_id:
            params["user_id"] = user_id
        if camera_id:
            params["camera_id"] = camera_id

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.database_url}/detected/list", params=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return {
                        "total_records": 0,
                        "total_pages": 0,
                        "current_page": page,
                        "per_page": per_page,
                        "detections": [],
                    }
        except Exception as e:
            print(f"Error getting detections: {str(e)}")
            return {
                "total_records": 0,
                "total_pages": 0,
                "current_page": page,
                "per_page": per_page,
                "detections": [],
            }

    async def get_detection(self, detection_id: str) -> Optional[Dict]:
        """Get detection details by ID"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.database_url}/detected/{detection_id}"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            print(f"Error getting detection details: {str(e)}")
            return None

    async def get_daily_stats(
        self, date_from: Optional[str] = None, date_to: Optional[str] = None
    ) -> List[Dict]:
        """Get daily detection statistics"""
        params = {}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.database_url}/detected/stats/daily", params=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            print(f"Error getting daily stats: {str(e)}")
            return []

    async def get_user_stats(self) -> List[Dict]:
        """Get detection statistics by user"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.database_url}/detected/stats/users"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            print(f"Error getting user stats: {str(e)}")
            return []

    async def get_camera_stats(self) -> List[Dict]:
        """Get detection statistics by camera"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.database_url}/detected/stats/cameras"
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            print(f"Error getting camera stats: {str(e)}")
            return []

    async def update_detection_user(self, detection_id: str, update_data: dict) -> bool:
        """Update user info in detection log"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.database_url}/detected/{detection_id}/update_user",
                    json=update_data,
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"Error updating detection user: {str(e)}")
            return False
