from database import RedisManager
from utils import genarate_id, save_image_to_folder, base64_to_image, is_better_quality
from service.users import UserService
import json
import time
import cv2


class DetectedService:
    def __init__(self):
        self.redis_manager = RedisManager()
        self.static_files = "static_files"

    def get_tracking_id(self, detect_id):
        tracking_id = self.redis_manager.client.get(f"tracking:{detect_id}")
        if not tracking_id:
            tracking_id = genarate_id()
            self.redis_manager.client.set(f"tracking:{detect_id}", tracking_id)
        return tracking_id

    def traking(
        self,
        detect_id,
        origin_image,
        detect_image,
        face_image,
        truth_image_path,
        distance,
        force_update,
        user_id=None,
    ):
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")

        tracking_id = self.get_tracking_id(detect_id)

        origin_image = base64_to_image(origin_image)
        origin_image = cv2.resize(origin_image, (640, 480))

        face_image = base64_to_image(face_image)
        detect_image = base64_to_image(detect_image)

        existing_record = self.redis_manager.get_by_id("detected_logs", tracking_id)

        if existing_record:
            unknow_user_id = UserService().get_unkow_user_id()

            if existing_record["user_id"] == unknow_user_id:
                return None

            old_face_image_path = existing_record["face_image_path"]
            old_origin_image_path = existing_record["origin_image_path"]
            old_detect_image_path = existing_record["detect_image_path"]

            old_face_image = cv2.imread(f"/{self.static_files}/{old_face_image_path}")

            if is_better_quality(face_image, old_face_image) or force_update:
                save_image_to_folder(
                    face_image, self.static_files, path=old_face_image_path
                )
                save_image_to_folder(
                    origin_image, self.static_files, path=old_origin_image_path
                )
                save_image_to_folder(
                    detect_image, self.static_files, path=old_detect_image_path
                )

                self.redis_manager.update_one(
                    "detected_logs",
                    {"detect_id": tracking_id},
                    {
                        "user_id": user_id if user_id else unknow_user_id,
                        "guess_uesr_id": unknow_user_id,
                        "distance": distance,
                        "time_stamp": current_time,
                    },
                )
        else:
            new_record = {
                "detect_id": tracking_id,
                "user_id": user_id if user_id else UserService().get_unkow_user_id(),
                "guess_uesr_id": UserService().get_unkow_user_id(),
                "face_image_path": save_image_to_folder(face_image, self.static_files),
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
            self.redis_manager.insert_one("detected_logs", new_record)

    def get_tracking_info(self, detect_id):
        tracking_id = self.get_tracking_id(detect_id)
        detected = self.redis_manager.get_by_id("detected_logs", tracking_id)
        if not detected:
            return "Not found", True

        user_id = detected["user_id"]
        user_name = UserService().get_user_name_by_id(user_id)

        return user_name, user_name == "unknown"
