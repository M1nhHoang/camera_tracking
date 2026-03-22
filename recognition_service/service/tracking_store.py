import cv2
import os
import time
import base64
import random
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from bson import ObjectId
from typing import Optional

from service.database import MongoDBManager


def _generate_id(length: int = 24) -> str:
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))


def _base64_to_image(image_data) -> np.ndarray:
    if isinstance(image_data, np.ndarray):
        return image_data
    elif isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
        return np.array(Image.open(BytesIO(image_bytes)))
    elif isinstance(image_data, bytes):
        return np.array(Image.open(BytesIO(image_data)))
    raise ValueError("Unsupported image format")


def _is_better_quality(new_img, old_img) -> bool:
    new_img = _base64_to_image(new_img)
    old_img = _base64_to_image(old_img)

    gray1 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    if gray1.shape != gray2.shape:
        gray1 = cv2.resize(gray1, gray2.shape[:2][::-1])
        gray2 = cv2.resize(gray2, gray1.shape[:2][::-1])

    return cv2.Laplacian(gray1, cv2.CV_64F).var() > cv2.Laplacian(gray2, cv2.CV_64F).var()


def _save_image(image, static_dir: str, path: str = None) -> str:
    image = _base64_to_image(image)
    os.makedirs(static_dir, exist_ok=True)

    if path:
        image_path = os.path.join(static_dir, path)
    else:
        image_name = _generate_id() + ".jpg"
        image_path = os.path.join(static_dir, image_name)

    cv2.imwrite(image_path, image)
    return os.path.basename(image_path)


class TrackingStore:
    """
    Handles tracking log persistence directly to MongoDB.
    Migrated from database_service/service/detected.py.
    """

    def __init__(self, mongo_uri: str, database_name: str, static_dir: str = "/static_files"):
        self.db = MongoDBManager(mongo_uri, database_name, "detected_logs")
        self.users_db = MongoDBManager(mongo_uri, database_name, "users")
        self.static_dir = static_dir
        self.tracking_id_cache = {}

        # Create index
        self.db.get_collection().create_index("detect_id", unique=True)

    def _get_unknown_user_id(self) -> ObjectId:
        """Get or create the 'unknown' user."""
        user = self.users_db.find_one({"identifier": "unknown"})
        if not user:
            result = self.users_db.insert_one({
                "identifier": "unknown",
                "username": "Unknown",
                "face_images_path": [],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return result.inserted_id
        return user["_id"]

    def save_tracking(
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
        """
        Save or update tracking record with best-shot override logic.
        Override when: better image quality OR lower distance (more confident match).
        """
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")

        # Get or create tracking ID for this detect_id
        tracking_id = self.tracking_id_cache.get(detect_id)
        if not tracking_id:
            tracking_id = _generate_id()
            self.tracking_id_cache[detect_id] = tracking_id

        # Decode images
        origin_image = _base64_to_image(origin_image)
        origin_image = cv2.resize(origin_image, (640, 480))
        face_image = _base64_to_image(face_image)
        detect_image = _base64_to_image(detect_image)

        unknown_user_id = self._get_unknown_user_id()

        existing = self.db.find_one({"detect_id": tracking_id})
        if existing:
            self._update_existing(
                existing, tracking_id, origin_image, detect_image, face_image,
                truth_image_path, distance, camera_id, user_id,
                unknown_user_id, current_time,
            )
        else:
            self._insert_new(
                tracking_id, origin_image, detect_image, face_image,
                truth_image_path, distance, camera_id, user_id,
                unknown_user_id, current_time,
            )

    def _update_existing(
        self, existing, tracking_id, origin_image, detect_image, face_image,
        truth_image_path, distance, camera_id, user_id,
        unknown_user_id, current_time,
    ):
        """Update existing record if better quality or better match."""
        old_face_path = existing["face_image_path"]
        old_face = cv2.imread(os.path.join(self.static_dir, old_face_path))

        has_better_image = _is_better_quality(face_image, old_face) if old_face is not None else True
        old_distance = existing.get("distance", float("inf"))
        has_better_match = distance < old_distance

        if not (has_better_image or has_better_match):
            return

        if has_better_image:
            _save_image(face_image, self.static_dir, path=existing["face_image_path"])
            _save_image(origin_image, self.static_dir, path=existing["origin_image_path"])
            _save_image(detect_image, self.static_dir, path=existing["detect_image_path"])

        update_data = {"time_stamp": current_time}
        if has_better_match:
            update_data["user_id"] = ObjectId(user_id) if user_id else unknown_user_id
            update_data["distance"] = distance
            update_data["truth_image_path"] = truth_image_path
        if camera_id:
            update_data["camera_id"] = ObjectId(camera_id)

        self.db.update_one({"detect_id": tracking_id}, update_data)

    def _insert_new(
        self, tracking_id, origin_image, detect_image, face_image,
        truth_image_path, distance, camera_id, user_id,
        unknown_user_id, current_time,
    ):
        """Insert new tracking record."""
        self.db.insert_one({
            "detect_id": tracking_id,
            "user_id": ObjectId(user_id) if user_id else unknown_user_id,
            "camera_id": ObjectId(camera_id) if camera_id else None,
            "face_image_path": _save_image(face_image, self.static_dir),
            "origin_image_path": _save_image(origin_image, self.static_dir),
            "detect_image_path": _save_image(detect_image, self.static_dir),
            "truth_image_path": truth_image_path,
            "distance": distance,
            "time_stamp": current_time,
        })
