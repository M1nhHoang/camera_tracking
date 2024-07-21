from pymongo import MongoClient
from queue import Queue
import threading
import time
import cv2
import base64
import numpy as np
from PIL import Image
import io
import random


class Database:
    def __init__(
        self,
        uri="mongodb://database:27017/",
        db_name="camera_traking",
        collection_name="detect_record",
    ):
        # Connect to MongoDB
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Create index for detect_id field
        self.collection.create_index("detect_id", unique=True)

        # Create a queue to store records
        self.detect_queue = Queue()

        # Cache
        self.cache_key_value = {}

        # Create and run a thread to process the queue
        self.thread = threading.Thread(target=self._process_detect_queue)
        self.thread.daemon = True
        self.thread.start()

    def genarate_detect_id(self):
        char_list = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
        )
        detect_id = ""
        for _ in range(24):
            detect_id += random.choice(char_list)
        return detect_id

    def save_to_database(
        self,
        detect_id,
        detect_name,
        origin_image,
        detect_image,
        face_image,
        force_update=False,
        time_stamp=time.strftime("%d-%m-%Y %H:%M:%S"),
    ):
        self.detect_queue.put(
            (
                detect_id,
                detect_name,
                origin_image,
                detect_image,
                face_image,
                force_update,
                time_stamp,
            )
        )

    def convert_image_to_base64(self, image):
        # Check if image is a valid base64 string
        if isinstance(image, str):
            return image
        elif isinstance(image, np.ndarray):
            # If image is a numpy array, convert it to bytes
            img_byte_arr = cv2.imencode(".jpg", image)[1].tobytes()
        elif isinstance(image, bytes):
            # If image is bytes, process it normally
            img_data = cv2.imdecode(
                np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            img_byte_arr = cv2.imencode(".jpg", img_data)[1].tobytes()
        else:
            raise ValueError("Unsupported image format")

        # Convert bytes to base64 string
        image_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
        return f"{image_base64}"

    def base64_to_image(self, image_data):
        if isinstance(image_data, np.ndarray):
            # If image is a numpy array (cv2)
            return image_data
        elif isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image_np = np.array(Image.open(io.BytesIO(image_bytes)))
            return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, bytes):
            # If image is image_bytes
            image_np = np.array(Image.open(io.BytesIO(image_data)))
            return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image format")

    def is_better_quality(self, new_img, old_img):
        new_img = self.base64_to_image(new_img)
        old_img = self.base64_to_image(old_img)

        # Convert images to grayscale and compare their sizes
        gray1 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
        if gray1.shape != gray2.shape:
            gray1 = cv2.resize(gray1, gray2.shape[:2][::-1])
            gray2 = cv2.resize(gray2, gray1.shape[:2][::-1])

        laplacian_var1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
        laplacian_var2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

        return laplacian_var1 > laplacian_var2

    def _process_detect_queue(self):
        while True:
            # Get record from queue and save to database
            record = self.detect_queue.get()
            if record is None:
                time.sleep(1)
                continue
            (
                detect_id,
                detect_name,
                origin_image,
                detect_image,
                face_image,
                force_update,
                time_stamp,
            ) = record

            # Resize
            origin_image = self.base64_to_image(origin_image)
            origin_image = cv2.resize(origin_image, (640, 480))

            # Check if detect id is in cache
            key_value = self.cache_key_value.get(detect_id)
            if not key_value:
                key_value = self.genarate_detect_id()
                self.cache_key_value[detect_id] = key_value

            # Check if record with detect id exists
            existing_record = self.collection.find_one({"detect_id": key_value})
            if existing_record:
                old_face_record = self.base64_to_image(existing_record["face_image"])
                # Is better quality
                if self.is_better_quality(face_image, old_face_record) or force_update:
                    self.collection.update_one(
                        {"detect_id": key_value},
                        {
                            "$set": {
                                "face_image": self.convert_image_to_base64(face_image),
                                "detect_name": detect_name,
                                "origin_image": self.convert_image_to_base64(
                                    origin_image
                                ),
                                "detect_image": self.convert_image_to_base64(
                                    detect_image
                                ),
                                "time_stamp": time_stamp,
                            }
                        },
                    )

            else:
                self.collection.insert_one(
                    {
                        "detect_id": key_value,
                        "face_image": self.convert_image_to_base64(face_image),
                        "detect_name": detect_name,
                        "origin_image": self.convert_image_to_base64(origin_image),
                        "detect_image": self.convert_image_to_base64(detect_image),
                        "time_stamp": time_stamp,
                    }
                )

            # Mark the queue task as done
            self.detect_queue.task_done()
