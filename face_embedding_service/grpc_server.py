import grpc
import logging
import numpy as np
from concurrent import futures
from PIL import Image
from io import BytesIO

from deepface import DeepFace
from generated import embedding_pb2, embedding_pb2_grpc

GRPC_PORT = 50052
MODEL_NAME = "Facenet"
NORMALIZATION = "Facenet"


class EmbeddingGrpcServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    """gRPC servicer for face embedding generation."""

    def __init__(self):
        DeepFace.build_model(MODEL_NAME)
        logging.info(f"Embedding model '{MODEL_NAME}' loaded")

    def GetEmbedding(self, request, context):
        """Single image embedding."""
        try:
            if len(request.face_image) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("face_image is required")
                return embedding_pb2.EmbeddingResponse(
                    success=False, message="face_image is required"
                )

            image = np.array(
                Image.open(BytesIO(request.face_image)).convert("RGB")
            )

            result = DeepFace.represent(
                image, model_name=MODEL_NAME, normalization=NORMALIZATION,
            )

            return embedding_pb2.EmbeddingResponse(
                success=True, embedding=result[0]["embedding"],
            )

        except Exception as e:
            logging.error(f"gRPC GetEmbedding error: {e}")
            return embedding_pb2.EmbeddingResponse(
                success=False, message=str(e)
            )

    def GetEmbeddingBatch(self, request, context):
        """Batch embedding — N images in 1 forward pass."""
        try:
            if not request.face_images:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("face_images is required")
                return embedding_pb2.EmbeddingBatchResponse(
                    success=False, message="face_images is required"
                )

            # Decode all JPEG bytes → numpy arrays
            images = []
            for img_bytes in request.face_images:
                image = np.array(
                    Image.open(BytesIO(img_bytes)).convert("RGB")
                )
                images.append(image)

            # Single forward pass for all images
            embeddings = DeepFace.represent_batch(
                images, model_name=MODEL_NAME, normalization=NORMALIZATION,
            )

            # Build response
            embedding_vectors = [
                embedding_pb2.EmbeddingVector(values=emb)
                for emb in embeddings
            ]

            return embedding_pb2.EmbeddingBatchResponse(
                success=True, embeddings=embedding_vectors,
            )

        except Exception as e:
            logging.error(f"gRPC GetEmbeddingBatch error: {e}")
            return embedding_pb2.EmbeddingBatchResponse(
                success=False, message=str(e)
            )


def start_grpc_server():
    """Start gRPC server (blocking)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_size", 10 * 1024 * 1024),
            ("grpc.max_send_message_size", 10 * 1024 * 1024),
        ],
    )
    servicer = EmbeddingGrpcServicer()
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logging.info(f"Embedding gRPC server started on port {GRPC_PORT}")
    server.wait_for_termination()
