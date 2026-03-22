import grpc
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Optional

from generated import embedding_pb2, embedding_pb2_grpc


class EmbeddingClient:
    """gRPC client for face_embedding_service."""

    def __init__(self, host: str, port: int):
        self.target = f"{host}:{port}"
        self.channel = grpc.insecure_channel(
            self.target,
            options=[
                ("grpc.max_send_message_size", 10 * 1024 * 1024),
                ("grpc.max_receive_message_size", 10 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )
        self.stub = embedding_pb2_grpc.EmbeddingServiceStub(self.channel)
        logging.info(f"Embedding gRPC client targeting {self.target}")

    def _encode_image(self, face_image: np.ndarray) -> bytes:
        """Encode numpy array to JPEG bytes."""
        buf = BytesIO()
        Image.fromarray(face_image).save(buf, format="JPEG")
        return buf.getvalue()

    def get_embedding(self, face_image: np.ndarray) -> Optional[list]:
        """Single image embedding. Returns list of floats or None."""
        try:
            request = embedding_pb2.EmbeddingRequest(
                face_image=self._encode_image(face_image),
            )
            response = self.stub.GetEmbedding(request, timeout=5.0)
            if response.success:
                return list(response.embedding)
            logging.error(f"Embedding error: {response.message}")
            return None
        except grpc.RpcError as e:
            logging.error(f"Embedding gRPC error: {e.code()}: {e.details()}")
            return None

    def get_embedding_batch(self, face_images: List[np.ndarray]) -> List[Optional[list]]:
        """
        Batch embedding — N images in 1 forward pass.
        Returns list of embedding vectors (or None for failed items).
        """
        try:
            encoded = [self._encode_image(img) for img in face_images]
            request = embedding_pb2.EmbeddingBatchRequest(face_images=encoded)
            response = self.stub.GetEmbeddingBatch(request, timeout=10.0)

            if response.success:
                return [list(vec.values) for vec in response.embeddings]

            logging.error(f"Batch embedding error: {response.message}")
            return [None] * len(face_images)
        except grpc.RpcError as e:
            logging.error(f"Batch embedding gRPC error: {e.code()}: {e.details()}")
            return [None] * len(face_images)

    def close(self):
        self.channel.close()
