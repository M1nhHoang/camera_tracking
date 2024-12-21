from queue import Queue
import threading
import time
import cv2
import base64
import numpy as np
from PIL import Image
import io
import requests
import chromadb

from uuid import uuid4
from ultralytics import YOLO
from io import BytesIO
import logging


class FaceIdentifyService:
    def __init__(
        self,
        database_service: dict,
        embedding_service: dict,
        vector_db_service: dict,
        model_path: str,
        conf_threshold=0.7,
    ):
        # Init service info
        self.database_name = database_service["name"]
        self.database_port = database_service["port"]
        self.embedding_name = embedding_service["name"]
        self.embedding_port = embedding_service["port"]
        self.vector_db_name = vector_db_service["name"]
        self.vector_db_port = vector_db_service["port"]

        # face detect threshold
        self.conf_threshold = conf_threshold

        # load model
        self.model = YOLO(model_path)

        # connect to chromadb
        chroma_client = chromadb.HttpClient(
            host=self.vector_db_name, port=self.vector_db_port
        )
        try:
            self.chroma_collection = chroma_client.create_collection("face_tracking")
            logging.info("Collection created successfully")
            print("Collection created successfully")
        except:
            self.chroma_collection = chroma_client.get_collection("face_tracking")
            logging.info("Collection already exists")
            print("Collection already exists")

        # Create a queue to store records
        self.detect_queue = Queue()

        # Create and run a thread to process the queue
        self.thread = threading.Thread(target=self._face_identification_queue)
        self.thread.daemon = True
        self.thread.start()

    def create_face_embedding(
        self, embedding_id: str, embedding: list, metadata: dict
    ) -> bool:
        """Add new face embedding to ChromaDB"""
        try:
            self.chroma_collection.add(
                ids=[embedding_id], embeddings=[embedding], metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error creating face embedding: {str(e)}")
            return False

    def get_face_embedding(self, embedding_id: str) -> dict:
        """Get face embedding by ID"""
        try:
            result = self.chroma_collection.get(
                ids=[embedding_id],
                include=["metadatas"],  # Remove embeddings from include
            )
            if result["ids"]:
                return {"id": result["ids"][0], "metadata": result["metadatas"][0]}
            return None
        except Exception as e:
            print(f"Error getting face embedding: {str(e)}")
            return None

    def get_all_face_embeddings(self, limit: int = 100, offset: int = 0) -> list:
        """Get all face embeddings with pagination"""
        try:
            result = self.chroma_collection.get(
                limit=limit,
                offset=offset,
                include=["metadatas"],  # Remove embeddings from include
            )

            formatted_result = {"ids": result["ids"], "metadatas": result["metadatas"]}
            return formatted_result
        except Exception as e:
            print(f"Error getting all face embeddings: {str(e)}")
            return {"ids": [], "metadatas": []}

    def update_face_embedding(
        self, embedding_id: str, embedding: list, metadata: dict
    ) -> bool:
        """Update existing face embedding"""
        try:
            # ChromaDB doesn't have direct update, so delete and re-add
            self.chroma_collection.delete(ids=[embedding_id])
            self.chroma_collection.add(
                ids=[embedding_id], embeddings=[embedding], metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating face embedding: {str(e)}")
            return False

    def delete_face_embedding(self, embedding_id: str) -> bool:
        """Delete face embedding"""
        try:
            self.chroma_collection.delete(ids=[embedding_id])
            return True
        except Exception as e:
            print(f"Error deleting face embedding: {str(e)}")
            return False

    def search_similar_faces(self, query_embedding: list, n_results: int = 5) -> list:
        """Search for similar face embeddings"""
        try:
            result = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],  # Remove embeddings from include
            )

            # Convert numpy arrays to lists
            formatted_result = {
                "ids": result["ids"],
                "metadatas": result["metadatas"],
                "distances": [
                    [float(d) for d in distance_list]
                    for distance_list in result["distances"]
                ],
            }
            return formatted_result
        except Exception as e:
            print(f"Error searching face embeddings: {str(e)}")
            return {"ids": [], "metadatas": [], "distances": []}

    def process_detect_queue(
        self,
        detect_id,
        origin_image,
        detect_image,
        detect_threshold=1,
        # threshold=1,
        force_update=False,
    ):
        self.detect_queue.put(
            (
                detect_id,
                origin_image,
                detect_image,
                detect_threshold,
                force_update,
            )
        )

    def user_update(self, identifier, user_name, face_images):
        headers = {"accept": "application/json", "Content-Type": "application/json"}

        data = {
            "identifier": identifier,
            "user_name": user_name,
            "face_images": face_images,
        }

        response = requests.post(
            f"http://{self.database_name}:{self.database_port}/users/user_update",
            headers=headers,
            json=data,
        )

        if response.status_code != 200:
            logging.error("Error when connecting to database server")
            raise ValueError("Error when connecting to database server")

        user_id = response.json()["user_id"]
        face_image_path = response.json()["face_image_path"]

        return user_id, face_image_path

    def chromadb_insert(self, user_info: dict, image_ebedding):
        self.chroma_collection.add(
            ids=[str(uuid4())],
            embeddings=[image_ebedding],
            metadatas=[user_info],
        )

    def face_validate(self, detect_image):
        face_images = self.face_detect(detect_image, is_counter=True)
        if len(face_images) != 1:
            logging.error("Must be only one face in your image.")
            raise ValueError("Must be only one face in your image.")

        face_image = face_images[0]
        if self.is_image_quality(face_image, 500) is False:
            logging.error("Image quality is not good.")
            raise ValueError("Image quality is not good.")

        return face_image

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
            logging.error("Unsupported image format")
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
            return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image_data, bytes):
            # If image is image_bytes
            image_np = np.array(Image.open(io.BytesIO(image_data)))
            return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            logging.error("Unsupported image format")
            raise ValueError("Unsupported image format")

    def process_face_image_upload(self, user_info: dict, btye_image):
        image = Image.open(BytesIO(btye_image))
        image = image.convert("RGB")
        image = np.array(image)

        face_image = self.face_validate(image)

        user_id, truth_image_path = self.user_update(
            identifier=user_info["identifier"],
            user_name=user_info["user_name"],
            face_images=[self.convert_image_to_base64(face_image)],
        )
        user_info["user_id"] = user_id
        user_info["truth_image_path"] = truth_image_path

        face_image_ebedding = self.embedding(face_image)
        self.chromadb_insert(user_info, face_image_ebedding)

    def is_image_quality(self, new_img, threshold=200):
        # Convert image to grayscale
        gray1 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

        # Calculate the variance of the Laplacian image
        laplacian_var1 = cv2.Laplacian(gray1, cv2.CV_64F).var()

        # Check if the variance of the new image is greater than 200
        return laplacian_var1 > threshold

    def face_detect(self, detect_image, detect_id=None, is_counter=False):
        # if not is_counter:
        #     # check face id is exit and compare quality
        #     if self.is_image_quality(detect_image, 200):
        #         return None

        object_count = []
        results = self.model(detect_image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if cls == 0 and conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_face = detect_image[y1:y2, x1:x2]

                    if not is_counter:
                        return cropped_face
                    else:
                        object_count.append(cropped_face)

        return object_count

    def embedding(self, face_image):
        # Convert face_image to bytes
        image_bytes = BytesIO()
        Image.fromarray(face_image).save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()

        # Perform embedding
        response = requests.post(
            f"http://{self.embedding_name}:{self.embedding_port}/embed",
            files={"file": ("face.jpg", image_bytes, "image/jpeg")},
        )
        if response.status_code != 200:
            print("Embedding error")
            return None
        embedding_data = response.json()
        embedding = embedding_data["embedding"]

        return embedding

    def vector_sreach(self, embedding):
        # vector sreach
        sreach_results = self.chroma_collection.query(
            query_embeddings=[embedding], n_results=1
        )

        return sreach_results

    def _face_identification_queue(self):
        while True:
            # Get record from queue and save to database
            track_queue = self.detect_queue.get()
            if track_queue is None:
                time.sleep(1)
                continue

            (
                detect_id,
                origin_image,
                detect_image,
                detect_threshold,
                force_update,
            ) = track_queue

            # face detect
            face_image = self.face_detect(detect_image, detect_id)
            if face_image == [] or face_image is None:
                # Mark the queue task as done
                self.detect_queue.task_done()
                continue

            # Perform embedding
            try:
                embedding = self.embedding(face_image)
            except Exception as e:
                print(e)
                # Mark the queue task as done
                self.detect_queue.task_done()
                continue

            # vector sreach
            sreach_results = self.vector_sreach(embedding)

            # get sreach results
            distances = sreach_results["distances"][0][0]
            metadata = sreach_results["metadatas"][0][0]

            # perform frame traking
            tracking_data = {
                "user_id": (
                    metadata["user_id"] if distances < detect_threshold else None
                ),
                "detect_id": detect_id,
                "origin_image": self.convert_image_to_base64(origin_image),
                "detect_image": self.convert_image_to_base64(detect_image),
                "face_image": self.convert_image_to_base64(face_image),
                "truth_image_path": metadata["truth_image_path"],
                "distance": distances,
                "force_update": distances < detect_threshold,
            }
            response = requests.post(
                f"http://{self.database_name}:{self.database_port}/detected/tracking",
                json=tracking_data,
            )

            if response.status_code != 200:
                print("Error when connecting to database server")

            # Mark the queue task as done
            self.detect_queue.task_done()
