import cv2
import os
import time
import logging
import numpy as np
from uuid import uuid4
from bson import ObjectId

from service.database import MongoDBManager

# Image save quality (only used for face_image which arrives as numpy)
JPEG_QUALITY = 80

# TODO: Static files storage
# Current: shared Docker volume (static_files/ mounted across services)
# This works for single-node deployment only.
# Future: migrate to Object Storage (MinIO/S3) when scaling to multi-node.
# Migration path:
#   1. Deploy MinIO container
#   2. Replace _save_bytes/_save_numpy with MinIO client upload
#   3. Update static file serving in backend_service

STATIC_DIR = "/static_files"


def _generate_id() -> str:
    return str(uuid4())


def _save_bytes(jpeg_bytes: bytes, directory: str, path: str = None) -> str:
    """Save raw JPEG bytes directly to disk. No re-encoding."""
    os.makedirs(directory, exist_ok=True)
    filename = path or (_generate_id() + ".jpg")
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        f.write(jpeg_bytes)
    return filename


def _save_numpy(image: np.ndarray, directory: str, path: str = None) -> str:
    """Save numpy image as JPEG Q=80."""
    os.makedirs(directory, exist_ok=True)
    filename = path or (_generate_id() + ".jpg")
    filepath = os.path.join(directory, filename)
    cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tofile(filepath)
    return filename


def _quality_from_bytes(jpeg_bytes: bytes) -> float:
    """Compute Laplacian variance from JPEG bytes without full decode."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def _quality_from_file(filepath: str) -> float:
    """Compute Laplacian variance from saved file."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


class TrackingStore:
    """
    Handles tracking log persistence directly to MongoDB.
    Origin/detect images arrive as raw JPEG bytes — saved directly, no re-encoding.
    Face images arrive as numpy — encoded once as JPEG Q=80.
    """

    def __init__(self, mongo_uri: str, database_name: str, static_dir: str = STATIC_DIR):
        self.db = MongoDBManager(mongo_uri, database_name, "detected_logs")
        self.users_db = MongoDBManager(mongo_uri, database_name, "users")
        self.static_dir = static_dir
        self.tracking_id_cache = {}

        self.db.get_collection().create_index("detect_id", unique=True)

    def _get_unknown_user_id(self) -> ObjectId:
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
        origin_image_bytes: bytes,
        detect_image_bytes: bytes,
        face_image: np.ndarray,
        truth_image_path,
        distance,
        camera_id=None,
        user_id=None,
    ):
        """
        Save or update tracking record.
        origin_image_bytes/detect_image_bytes: raw JPEG bytes (direct to disk)
        face_image: numpy array (encode once as JPEG Q=80)
        """
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")

        tracking_id = self.tracking_id_cache.get(detect_id)
        if not tracking_id:
            tracking_id = _generate_id()
            self.tracking_id_cache[detect_id] = tracking_id

        unknown_user_id = self._get_unknown_user_id()

        existing = self.db.find_one({"detect_id": tracking_id})
        if existing:
            self._update_existing(
                existing, tracking_id,
                origin_image_bytes, detect_image_bytes, face_image,
                truth_image_path, distance, camera_id, user_id,
                unknown_user_id, current_time,
            )
        else:
            self._insert_new(
                tracking_id,
                origin_image_bytes, detect_image_bytes, face_image,
                truth_image_path, distance, camera_id, user_id,
                unknown_user_id, current_time,
            )

    def _update_existing(
        self, existing, tracking_id,
        origin_image_bytes, detect_image_bytes, face_image,
        truth_image_path, distance, camera_id, user_id,
        unknown_user_id, current_time,
    ):
        # Compare quality: new face vs old saved face
        old_face_path = os.path.join(self.static_dir, existing["face_image_path"])
        new_quality = _quality_from_bytes(
            cv2.imencode(".jpg", face_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        ) if face_image is not None else 0.0
        old_quality = _quality_from_file(old_face_path)

        has_better_image = new_quality > old_quality
        old_distance = existing.get("distance", float("inf"))
        has_better_match = distance < old_distance

        if not (has_better_image or has_better_match):
            return

        if has_better_image:
            # Overwrite files — bytes direct, face as numpy
            _save_bytes(origin_image_bytes, self.static_dir, path=existing["origin_image_path"])
            _save_bytes(detect_image_bytes, self.static_dir, path=existing["detect_image_path"])
            _save_numpy(face_image, self.static_dir, path=existing["face_image_path"])

        update_data = {"time_stamp": current_time}
        if has_better_match:
            update_data["user_id"] = ObjectId(user_id) if user_id else unknown_user_id
            update_data["distance"] = distance
            update_data["truth_image_path"] = truth_image_path
        if camera_id:
            update_data["camera_id"] = ObjectId(camera_id)

        self.db.update_one({"detect_id": tracking_id}, update_data)

    def _insert_new(
        self, tracking_id,
        origin_image_bytes, detect_image_bytes, face_image,
        truth_image_path, distance, camera_id, user_id,
        unknown_user_id, current_time,
    ):
        self.db.insert_one({
            "detect_id": tracking_id,
            "user_id": ObjectId(user_id) if user_id else unknown_user_id,
            "camera_id": ObjectId(camera_id) if camera_id else None,
            "origin_image_path": _save_bytes(origin_image_bytes, self.static_dir),
            "detect_image_path": _save_bytes(detect_image_bytes, self.static_dir),
            "face_image_path": _save_numpy(face_image, self.static_dir),
            "truth_image_path": truth_image_path,
            "distance": distance,
            "time_stamp": current_time,
        })
