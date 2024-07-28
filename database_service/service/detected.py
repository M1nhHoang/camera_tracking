from database import MongoDBManager
from utils import genarate_id, save_image_to_folder, base64_to_image, is_better_quality
from service.users import UserService
from bson.objectid import ObjectId

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
        force_update,
        user_id=None,
    ):
        # current time
        current_time = time.strftime("%d-%m-%Y %H:%M:%S")

        # get traking id
        tracking_id = self.traking_id_cache.get(detect_id, None)
        if not tracking_id:
            self.traking_id_cache[detect_id] = genarate_id()

        # Resize
        origin_image = base64_to_image(origin_image)
        origin_image = cv2.resize(origin_image, (640, 480))

        # conver to image
        face_image = base64_to_image(face_image)
        detect_image = base64_to_image(detect_image)

        # Check if record with detect id exists
        existing_record = self.db_manager.find_one({"detect_id": tracking_id})
        if existing_record:
            unknow_user_id = UserService().get_unkow_user_id()

            # check is detect user
            if existing_record["user_id"] == unknow_user_id:
                return None

            # get image path
            old_face_image_path = existing_record["face_image_path"]
            old_origin_image_path = existing_record["origin_image_path"]
            old_detect_image_path = existing_record["detect_image_path"]

            # read image from path
            old_face_image = cv2.imread(f"/{self.static_files}/{old_face_image_path}")

            # Is better quality
            if is_better_quality(face_image, old_face_image) or force_update:
                # overwrite image
                save_image_to_folder(
                    face_image, self.static_files, path=old_face_image_path
                )
                save_image_to_folder(
                    origin_image, self.static_files, path=old_origin_image_path
                )
                save_image_to_folder(
                    detect_image, self.static_files, path=old_detect_image_path
                )

                self.db_manager.update_one(
                    {"detect_id": tracking_id},
                    {
                        "user_id": ObjectId(user_id) if user_id else unknow_user_id,
                        "guess_uesr_id": ObjectId(
                            unknow_user_id
                        ),  # update later, save as similar user face
                        "distance": distance,
                        "time_stamp": current_time,
                    },
                )

        else:
            # save new record
            self.db_manager.insert_one(
                {
                    "detect_id": tracking_id,
                    "user_id": ObjectId(user_id)
                    if user_id
                    else UserService().get_unkow_user_id(),
                    "guess_uesr_id": ObjectId(
                        UserService().get_unkow_user_id()
                    ),  # update later, save as similar user face
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

    def get_tracking_info(self, detect_id):
        tracking_id = self.traking_id_cache.get(detect_id, None)
        if not tracking_id:
            return "Peding detect...", True

        detected = self.db_manager.find_one({"detect_id": tracking_id})
        user_id = detected["user_id"]
        user_name = UserService().get_user_name_by_id(user_id)

        return user_name, user_name == "unknown"
