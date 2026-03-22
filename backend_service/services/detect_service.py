import logging
from typing import Dict, List, Optional
from datetime import datetime
from bson import ObjectId

from database import MongoDB

logger = logging.getLogger(__name__)


class DetectionService:
    """Detection log queries — reads directly from MongoDB."""

    def __init__(self):
        self.detected = MongoDB.get_collection("detected_logs")
        self.users = MongoDB.get_collection("users")
        self.cameras = MongoDB.get_collection("cameras")

    def _get_user_name(self, user_id) -> str:
        user = self.users.find_one({"_id": user_id})
        return user["username"] if user else "Unknown"

    def _get_camera_name(self, camera_id) -> str:
        if not camera_id:
            return "Unknown"
        camera = self.cameras.find_one({"_id": camera_id})
        return camera["name"] if camera else "Unknown"

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
        """Get filtered detections with pagination."""
        query = {}

        if date_from or date_to:
            query["time_stamp"] = {}
            if date_from:
                from_date = datetime.strptime(date_from, "%Y-%m-%d")
                query["time_stamp"]["$gte"] = from_date.strftime("%d-%m-%Y %H:%M:%S")
            if date_to:
                to_date = datetime.strptime(date_to, "%Y-%m-%d")
                query["time_stamp"]["$lte"] = to_date.strftime("%d-%m-%Y %H:%M:%S")
        if user_id:
            query["user_id"] = ObjectId(user_id)
        if camera_id:
            query["camera_id"] = ObjectId(camera_id)

        skip = (page - 1) * per_page
        sort_direction = -1 if sort_order == "desc" else 1

        total_count = self.detected.count_documents(query)
        total_pages = (total_count + per_page - 1) // per_page

        records = (
            self.detected.find(query)
            .skip(skip)
            .limit(per_page)
            .sort(sort_by, sort_direction)
        )

        detections = []
        for record in records:
            detections.append({
                "id": str(record["_id"]),
                "user_id": str(record["user_id"]),
                "user_name": self._get_user_name(record["user_id"]),
                "camera_name": self._get_camera_name(record.get("camera_id")),
                "time_stamp": record["time_stamp"],
                "distance": record["distance"],
                "origin_image_path": record.get("origin_image_path"),
                "face_image_path": record.get("face_image_path"),
                "detect_image_path": record.get("detect_image_path"),
            })

        return {
            "total_records": total_count,
            "total_pages": total_pages,
            "current_page": page,
            "per_page": per_page,
            "detections": detections,
        }

    async def get_detection(self, detection_id: str) -> Optional[Dict]:
        """Get detection details by ID."""
        try:
            record = self.detected.find_one({"_id": ObjectId(detection_id)})
            if not record:
                return None

            return {
                "id": str(record["_id"]),
                "user_id": str(record["user_id"]),
                "user_name": self._get_user_name(record["user_id"]),
                "camera_name": self._get_camera_name(record.get("camera_id")),
                "time_stamp": record["time_stamp"],
                "distance": record["distance"],
                "origin_image_path": record.get("origin_image_path"),
                "face_image_path": record.get("face_image_path"),
                "detect_image_path": record.get("detect_image_path"),
                "truth_image_path": record.get("truth_image_path"),
            }
        except Exception as e:
            logger.error(f"Error getting detection: {str(e)}")
            return None

    async def get_daily_stats(
        self, date_from: Optional[str] = None, date_to: Optional[str] = None
    ) -> List[Dict]:
        """Daily detection statistics."""
        query = {}
        if date_from:
            query.setdefault("time_stamp", {})["$gte"] = date_from
        if date_to:
            query.setdefault("time_stamp", {})["$lte"] = date_to

        pipeline = [
            {"$match": query},
            {"$group": {"_id": {"$substr": ["$time_stamp", 0, 10]}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
        return [{"date": r["_id"], "count": r["count"]} for r in self.detected.aggregate(pipeline)]

    async def get_user_stats(self) -> List[Dict]:
        """Detection statistics by user."""
        pipeline = [
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
        ]
        stats = []
        for r in self.detected.aggregate(pipeline):
            stats.append({
                "user_id": str(r["_id"]),
                "user_name": self._get_user_name(r["_id"]),
                "count": r["count"],
            })
        return sorted(stats, key=lambda x: x["count"], reverse=True)

    async def get_camera_stats(self) -> List[Dict]:
        """Detection statistics by camera."""
        pipeline = [
            {"$group": {"_id": "$camera_id", "count": {"$sum": 1}}},
        ]
        stats = []
        for r in self.detected.aggregate(pipeline):
            if r["_id"]:
                stats.append({
                    "camera_id": str(r["_id"]),
                    "camera_name": self._get_camera_name(r["_id"]),
                    "count": r["count"],
                })
        return sorted(stats, key=lambda x: x["count"], reverse=True)

    async def update_detection_user(self, detection_id: str, update_data: dict) -> bool:
        """Update user assignment in detection log."""
        try:
            result = self.detected.update_one(
                {"_id": ObjectId(detection_id)},
                {"$set": {
                    "user_id": ObjectId(update_data["user_id"]),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }},
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating detection user: {str(e)}")
            return False
