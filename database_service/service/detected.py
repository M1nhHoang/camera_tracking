from database import MongoDBManager
from utils import genarate_id, save_image_to_folder, base64_to_image, is_better_quality
from service.users import UserService
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import cv2
import time


class DetectedService:
    def __init__(self):
        self.db_manager = MongoDBManager(collection_name="detected_logs")
        self.static_files = "static_files"
        self.traking_id_cache = {}

        # Create index for detect_id field
        self.db_manager.get_collection().create_index("detect_id", unique=True)

    def traking(
        self,
        detect_id,
        origin_image,
        detect_image,
        face_image,
        truth_image_path,
        distance,
        camera_id=None,
        user_id=None,
    ):
        # current time
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")

        # get traking id
        tracking_id = self.traking_id_cache.get(detect_id, None)
        if not tracking_id:
            self.traking_id_cache[detect_id] = genarate_id()
            tracking_id = self.traking_id_cache[detect_id]

        # Resize
        origin_image = base64_to_image(origin_image)
        origin_image = cv2.resize(origin_image, (640, 480))

        # conver to image
        face_image = base64_to_image(face_image)
        detect_image = base64_to_image(detect_image)

        # Check if record with detect id exists
        existing_record = self.db_manager.find_one({"detect_id": tracking_id})
        if existing_record:
            unknow_user_id = UserService().get_unknown_user_id()

            # Determine if we should override:
            # 1. Better face image quality (sharper)
            # 2. Better identification (lower distance = more confident match)
            old_face_image_path = existing_record["face_image_path"]
            old_face_image = cv2.imread(f"/{self.static_files}/{old_face_image_path}")

            has_better_image = is_better_quality(face_image, old_face_image)
            old_distance = existing_record.get("distance", float("inf"))
            has_better_match = distance < old_distance

            if has_better_image or has_better_match:
                # Override images with better quality shots
                old_origin_image_path = existing_record["origin_image_path"]
                old_detect_image_path = existing_record["detect_image_path"]

                if has_better_image:
                    save_image_to_folder(
                        face_image, self.static_files, path=old_face_image_path
                    )
                    save_image_to_folder(
                        origin_image, self.static_files, path=old_origin_image_path
                    )
                    save_image_to_folder(
                        detect_image, self.static_files, path=old_detect_image_path
                    )

                # Update identification if better match
                update_data = {
                    "time_stamp": current_time,
                }
                if has_better_match:
                    update_data["user_id"] = ObjectId(user_id) if user_id else unknow_user_id
                    update_data["distance"] = distance
                    update_data["truth_image_path"] = truth_image_path
                if camera_id:
                    update_data["camera_id"] = ObjectId(camera_id)

                self.db_manager.update_one(
                    {"detect_id": tracking_id}, update_data
                )

        else:
            # Save new record
            unknow_user_id = UserService().get_unknown_user_id()
            self.db_manager.insert_one(
                {
                    "detect_id": tracking_id,
                    "user_id": ObjectId(user_id) if user_id else unknow_user_id,
                    "camera_id": ObjectId(camera_id) if camera_id else None,
                    "face_image_path": save_image_to_folder(
                        face_image, self.static_files
                    ),
                    "origin_image_path": save_image_to_folder(
                        origin_image, self.static_files
                    ),
                    "detect_image_path": save_image_to_folder(
                        detect_image, self.static_files
                    ),
                    "truth_image_path": truth_image_path,
                    "distance": distance,
                    "time_stamp": current_time,
                }
            )

    def get_daily_stats(self, date_from=None, date_to=None):
        """Get daily detection statistics"""
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": {"$toDate": "$time_stamp"},
                        }
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        if date_from or date_to:
            match_query = {}
            if date_from:
                match_query["$gte"] = date_from
            if date_to:
                match_query["$lte"] = date_to
            if match_query:
                pipeline.insert(0, {"$match": {"time_stamp": match_query}})

        results = list(self.db_manager.get_collection().aggregate(pipeline))
        return [{"date": doc["_id"], "count": doc["count"]} for doc in results]

    def get_user_stats(self):
        """Get detection statistics by user"""
        pipeline = [
            {
                "$lookup": {
                    "from": "users",
                    "localField": "user_id",
                    "foreignField": "_id",
                    "as": "user_info",
                }
            },
            {"$unwind": "$user_info"},
            {
                "$group": {
                    "_id": "$user_id",
                    "user_name": {"$first": "$user_info.username"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
        ]

        results = list(self.db_manager.get_collection().aggregate(pipeline))
        return [
            {
                "user_id": str(doc["_id"]),
                "user_name": doc["user_name"],
                "count": doc["count"],
            }
            for doc in results
        ]

    def get_camera_stats(self):
        """Get detection statistics by camera"""
        pipeline = [
            {
                "$lookup": {
                    "from": "cameras",
                    "localField": "camera_id",
                    "foreignField": "_id",
                    "as": "camera_info",
                }
            },
            {"$unwind": "$camera_info"},
            {
                "$group": {
                    "_id": "$camera_id",
                    "camera_name": {"$first": "$camera_info.name"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"count": -1}},
        ]

        results = list(self.db_manager.get_collection().aggregate(pipeline))
        return [
            {
                "camera_id": str(doc["_id"]),
                "camera_name": doc["camera_name"],
                "count": doc["count"],
            }
            for doc in results
        ]

    async def get_detections(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        user_id: Optional[ObjectId] = None,
        camera_id: Optional[ObjectId] = None,
        page: int = 1,
        per_page: int = 10,
    ) -> Dict:
        """Get filtered detections with pagination"""
        # Build query
        query = {}
        if from_date:
            query["time_stamp"] = {"$gte": from_date.strftime("%d-%m-%Y %H:%M:%S")}
        if to_date:
            if "time_stamp" in query:
                query["time_stamp"]["$lte"] = to_date.strftime("%d-%m-%Y %H:%M:%S")
            else:
                query["time_stamp"] = {"$lte": to_date.strftime("%d-%m-%Y %H:%M:%S")}
        if user_id:
            query["user_id"] = user_id
        if camera_id:
            query["camera_id"] = camera_id

        # Calculate skip value for pagination
        skip = (page - 1) * per_page

        # Get total count
        total_records = len(self.db_manager.find_all(query))
        total_pages = (total_records + per_page - 1) // per_page

        # Get paginated results
        detections = self.db_manager.find_all(query).skip(skip).limit(per_page)

        # Format results
        formatted_detections = []
        for detection in detections:
            formatted_detections.append(
                {
                    "_id": str(detection["_id"]),
                    "user_id": str(detection["user_id"]),
                    "user_name": self._get_user_name(detection["user_id"]),
                    "camera_id": str(detection.get("camera_id", "")),
                    "camera_name": self._get_camera_name(detection.get("camera_id")),
                    "time_stamp": detection["time_stamp"],
                    "distance": detection["distance"],
                    "origin_image_path": detection.get("origin_image_path"),
                    "face_image_path": detection.get("face_image_path"),
                    "detect_image_path": detection.get("detect_image_path"),
                }
            )

        return {
            "detections": formatted_detections,
            "total_pages": total_pages,
            "total_records": total_records,
        }

    async def get_daily_stats(
        self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Get daily detection statistics"""
        query = {}
        if from_date:
            query["time_stamp"] = {"$gte": from_date.strftime("%d-%m-%Y %H:%M:%S")}
        if to_date:
            if "time_stamp" in query:
                query["time_stamp"]["$lte"] = to_date.strftime("%d-%m-%Y %H:%M:%S")
            else:
                query["time_stamp"] = {"$lte": to_date.strftime("%d-%m-%Y %H:%M:%S")}

        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {"$substr": ["$time_stamp", 0, 10]},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        results = self.db_manager.get_collection().aggregate(pipeline)

        stats = []
        for result in results:
            stats.append({"date": result["_id"], "count": result["count"]})

        return stats

    async def get_user_stats(self) -> List[Dict]:
        """Get detection statistics by user"""
        pipeline = [
            {
                "$group": {
                    "_id": "$user_id",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": {"$subtract": [100, "$distance"]}},
                }
            }
        ]

        results = self.db_manager.get_collection().aggregate(pipeline)

        stats = []
        for result in results:
            user_name = self._get_user_name(result["_id"])
            stats.append(
                {
                    "user_id": str(result["_id"]),
                    "user_name": user_name,
                    "count": result["count"],
                    "avg_confidence": round(result["avg_confidence"], 2),
                }
            )

        return sorted(stats, key=lambda x: x["count"], reverse=True)

    async def get_camera_stats(self) -> List[Dict]:
        """Get detection statistics by camera"""
        pipeline = [{"$group": {"_id": "$camera_id", "count": {"$sum": 1}}}]

        results = self.db_manager.get_collection().aggregate(pipeline)

        stats = []
        for result in results:
            if result["_id"]:  # Skip if camera_id is None
                camera_name = self._get_camera_name(result["_id"])
                stats.append(
                    {
                        "camera_id": str(result["_id"]),
                        "camera_name": camera_name,
                        "count": result["count"],
                    }
                )

        return sorted(stats, key=lambda x: x["count"], reverse=True)

    async def get_detection(self, detection_id: str):
        """Get detection details by ID"""
        try:
            # Convert string ID to ObjectId
            detection_id = ObjectId(detection_id)

            # Get detection record
            record = self.db_manager.find_one({"_id": detection_id})
            if not record:
                return None

            # Get user info
            user_db = MongoDBManager(collection_name="users")
            user = user_db.find_one({"_id": record["user_id"]})
            username = user["username"] if user else "Unknown"

            # Get camera info if exists
            camera_name = "Unknown"
            if "camera_id" in record:
                camera_db = MongoDBManager(collection_name="cameras")
                camera = camera_db.find_one({"_id": record["camera_id"]})
                if camera:
                    camera_name = (
                        f"{camera['name']} ({camera['location']})"
                        if "location" in camera
                        else camera["name"]
                    )

            # Format response
            detection = {
                "id": str(record["_id"]),
                "user_id": str(record["user_id"]),
                "user_name": username,
                "camera_name": camera_name,
                "time_stamp": record["time_stamp"],
                "distance": record["distance"],
                "origin_image_path": record.get("origin_image_path"),
                "face_image_path": record.get("face_image_path"),
                "detect_image_path": record.get("detect_image_path"),
                "truth_image_path": record.get("truth_image_path"),
            }

            return detection

        except Exception as e:
            print(f"Error getting detection: {str(e)}")
            return None

    def _get_user_name(self, user_id: ObjectId) -> str:
        """Get user name from user ID"""
        user_db = MongoDBManager(collection_name="users")
        user = user_db.find_one({"_id": user_id})
        return user["username"] if user else "Unknown"

    def _get_camera_name(self, camera_id: Optional[ObjectId]) -> str:
        """Get camera name from camera ID"""
        if not camera_id:
            return "Unknown"
        camera_db = MongoDBManager(collection_name="cameras")
        camera = camera_db.find_one({"_id": camera_id})
        return camera["name"] if camera else "Unknown"

    async def update_detection_user(self, detection_id: str, update_data: dict) -> bool:
        """Update user info in detection"""
        try:
            detection_id_obj = ObjectId(detection_id)
            update_data = {
                "user_id": ObjectId(update_data["user_id"]),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            result = self.db_manager.update_one({"_id": detection_id_obj}, update_data)
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating detection user: {str(e)}")
            return False
