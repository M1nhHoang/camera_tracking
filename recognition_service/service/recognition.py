import logging

from service.chromadb_ops import ChromaDBClient
from service.embedding_client import EmbeddingClient
from service.pipeline import IdentificationPipeline
from service.tracking_store import TrackingStore
from service.database import MongoDBManager


class RecognitionService:
    """
    Recognition service — AI identification pipeline only.
    No user management, no face validation, no YOLO model.

    Responsibilities:
    - Receive face images from detection_service (gRPC)
    - Generate embeddings via face_embedding_service (gRPC)
    - Search ChromaDB for matching identity
    - Cache tracking results for detection_service
    - Save tracking logs to MongoDB + images to disk
    """

    def __init__(
        self,
        mongo_uri: str,
        database_name: str,
        embedding_service: dict,
        vector_db_service: dict,
        static_dir: str = "/static_files",
        detect_threshold: float = 0.40,
    ):
        # Embedding client (gRPC to face_embedding_service)
        self.embedding_client = EmbeddingClient(
            host=embedding_service["name"], port=embedding_service["grpc_port"],
        )

        # ChromaDB client
        self.chromadb = ChromaDBClient(
            host=vector_db_service["name"], port=vector_db_service["port"],
        )

        # Tracking log storage (MongoDB + disk)
        self.tracking_store = TrackingStore(
            mongo_uri=mongo_uri,
            database_name=database_name,
            static_dir=static_dir,
        )

        # Identification pipeline (async batch processing)
        self.pipeline = IdentificationPipeline(
            embedding_client=self.embedding_client,
            chromadb_client=self.chromadb,
            tracking_store=self.tracking_store,
            detect_threshold=detect_threshold,
        )

    def process_detect_queue(self, detect_id, origin_image_bytes, detect_image_bytes,
                             detect_threshold=None, camera_id=None, face_image=None):
        """
        Enqueue detection for async batch processing.
        origin_image_bytes/detect_image_bytes: raw JPEG bytes (saved directly)
        face_image: numpy array (needed for embedding)
        """
        self.pipeline.enqueue(
            detect_id, origin_image_bytes, detect_image_bytes,
            detect_threshold=detect_threshold,
            camera_id=camera_id, face_image=face_image,
        )
