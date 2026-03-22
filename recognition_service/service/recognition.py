import logging
from datetime import datetime

from service.chromadb_ops import ChromaDBClient
from service.face_utils import FaceProcessor
from service.embedding_client import EmbeddingClient
from service.pipeline import IdentificationPipeline
from service.tracking_store import TrackingStore
from service.database import MongoDBManager


class RecognitionService:
    """
    Main recognition service — AI identification pipeline.
    Connects directly to MongoDB for tracking logs and user registration.

    NOTE: User registration logic (face_upload, face_validate, user_update)
    is temporarily kept here but should be migrated to a separate business
    logic service in the future.
    """

    def __init__(
        self,
        mongo_uri: str,
        database_name: str,
        embedding_service: dict,
        vector_db_service: dict,
        model_path: str,
        static_dir: str = "/static_files",
        conf_threshold: float = 0.85,
        detect_threshold: float = 0.40,
    ):
        # Face detection/validation
        self.face_processor = FaceProcessor(model_path, conf_threshold)

        # Embedding client (gRPC to face_embedding_service)
        self.embedding_client = EmbeddingClient(
            host=embedding_service["name"], port=embedding_service["grpc_port"],
        )

        # ChromaDB client
        self.chromadb = ChromaDBClient(
            host=vector_db_service["name"], port=vector_db_service["port"],
        )

        # MongoDB direct access for tracking logs
        self.tracking_store = TrackingStore(
            mongo_uri=mongo_uri,
            database_name=database_name,
            static_dir=static_dir,
        )

        # MongoDB for user operations
        self.users_db = MongoDBManager(mongo_uri, database_name, "users")

        # Identification pipeline (async batch processing)
        self.pipeline = IdentificationPipeline(
            face_processor=self.face_processor,
            embedding_client=self.embedding_client,
            chromadb_client=self.chromadb,
            tracking_store=self.tracking_store,
            detect_threshold=detect_threshold,
        )

    # --- Core AI pipeline (called by gRPC) ---

    def process_detect_queue(self, detect_id, origin_image, detect_image,
                             detect_threshold=None, camera_id=None, face_image=None):
        """Enqueue detection for async batch processing."""
        self.pipeline.enqueue(
            detect_id, origin_image, detect_image,
            detect_threshold=detect_threshold,
            camera_id=camera_id, face_image=face_image,
        )

    # --- TODO: Migrate to separate business logic service ---

    def process_face_image_upload(self, user_info: dict, byte_image):
        """Process uploaded face image: validate → register user → embed → store."""
        from PIL import Image
        from io import BytesIO
        import numpy as np

        image = np.array(Image.open(BytesIO(byte_image)))
        face_image = self.face_processor.validate(image)

        user_id, truth_image_path = self._user_update(
            identifier=user_info["identifier"],
            user_name=user_info["user_name"],
            face_images=[self.face_processor.to_base64(face_image)],
        )
        user_info["user_id"] = user_id
        user_info["truth_image_path"] = truth_image_path

        embedding = self.embedding_client.get_embedding(face_image)
        self.chromadb.insert(user_info, embedding)

    def _user_update(self, identifier, user_name, face_images):
        """Create or update user directly in MongoDB."""
        existing = self.users_db.find_one({"identifier": identifier})

        if existing:
            # Update existing user
            current_paths = existing.get("face_images_path", [])
            # Save new face images
            from service.tracking_store import _base64_to_image, _save_image
            new_paths = []
            for img_b64 in face_images:
                path = _save_image(img_b64, self.tracking_store.static_dir)
                new_paths.append(path)

            self.users_db.update_one(
                {"_id": existing["_id"]},
                {
                    "username": user_name,
                    "face_images_path": current_paths + new_paths,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            user_id = str(existing["_id"])
            face_image_path = new_paths[-1] if new_paths else None
        else:
            # Create new user
            from service.tracking_store import _save_image
            new_paths = []
            for img_b64 in face_images:
                path = _save_image(img_b64, self.tracking_store.static_dir)
                new_paths.append(path)

            result = self.users_db.insert_one({
                "username": user_name,
                "identifier": identifier,
                "face_images_path": new_paths,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_detection": None,
            })
            user_id = str(result.inserted_id)
            face_image_path = new_paths[-1] if new_paths else None

        return user_id, face_image_path

    # --- Convenience accessors for routers ---

    @property
    def chroma_collection(self):
        return self.chromadb.collection

    def face_validate(self, image):
        return self.face_processor.validate(image)

    def convert_image_to_base64(self, image):
        return self.face_processor.to_base64(image)

    def embedding(self, face_image):
        return self.embedding_client.get_embedding(face_image)

    def chromadb_insert(self, metadata, embedding):
        return self.chromadb.insert(metadata, embedding)

    def delete_user_embeddings(self, user_id):
        return self.chromadb.delete_by_user(user_id)

    def update_user_metadata(self, user_id, metadata):
        return self.chromadb.update_user_metadata(user_id, metadata)

    def count_user_embeddings(self, user_id):
        return self.chromadb.count_by_user(user_id)
