import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
from config import settings


class DetectionService:
    def __init__(self):
        self.database_url = settings.DATABASE_SERVICE_URL

    async def get_detections(self, filters: dict) -> Dict:
        """Get filtered list of detections with pagination"""
        # Build query parameters
        params = {
            "page": filters.get("page", 1),
            "per_page": filters.get("per_page", 10),
        }

        if filters.get("date_from"):
            params["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            params["date_to"] = filters["date_to"]
        if filters.get("user_id"):
            params["user_id"] = filters["user_id"]
        if filters.get("camera_id"):
            params["camera_id"] = filters["camera_id"]

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.database_url}/detected/list", params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {"detections": [], "total_pages": 0, "total_records": 0}

    async def get_detection(self, detection_id: str) -> Optional[Dict]:
        """Get detection details by ID"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.database_url}/detected/{detection_id}"
            ) as response:
                if response.status == 200:
                    return await response.json()
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

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.database_url}/detected/stats/daily", params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []

    async def get_user_stats(self) -> List[Dict]:
        """Get detection statistics by user"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.database_url}/detected/stats/users"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []

    async def get_camera_stats(self) -> List[Dict]:
        """Get detection statistics by camera"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.database_url}/detected/stats/cameras"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return []
