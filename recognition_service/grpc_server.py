import grpc
import logging
import numpy as np
from concurrent import futures
from PIL import Image
from io import BytesIO

from generated import recognition_pb2, recognition_pb2_grpc

GRPC_PORT = 50051


class RecognitionGrpcServicer(recognition_pb2_grpc.RecognitionServiceServicer):
    """gRPC servicer that wraps the existing RecognitionService."""

    def __init__(self, recognition_service):
        self.recognition_service = recognition_service

    def IdentifyFace(self, request, context):
        try:
            if request.detect_id == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("detect_id is required")
                return recognition_pb2.FaceIdentificationResponse(
                    success=False, message="detect_id is required"
                )

            # Decode JPEG bytes → numpy arrays
            origin_image = np.array(
                Image.open(BytesIO(request.origin_image)).convert("RGB")
            )
            detect_image = np.array(
                Image.open(BytesIO(request.detect_image)).convert("RGB")
            )

            face_image_arr = None
            if len(request.face_image) > 0:
                face_image_arr = np.array(
                    Image.open(BytesIO(request.face_image)).convert("RGB")
                )

            # Enqueue for async processing (same as HTTP endpoint)
            self.recognition_service.process_detect_queue(
                request.detect_id,
                origin_image,
                detect_image,
                camera_id=request.camera_id,
                face_image=face_image_arr,
            )

            return recognition_pb2.FaceIdentificationResponse(success=True)

        except Exception as e:
            logging.error(f"gRPC IdentifyFace error: {e}")
            return recognition_pb2.FaceIdentificationResponse(
                success=False, message=str(e)
            )


def start_grpc_server(recognition_service):
    """Start gRPC server in the current thread (blocking)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_size", 10 * 1024 * 1024),
            ("grpc.max_send_message_size", 10 * 1024 * 1024),
        ],
    )
    servicer = RecognitionGrpcServicer(recognition_service)
    recognition_pb2_grpc.add_RecognitionServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logging.info(f"gRPC server started on port {GRPC_PORT}")
    server.wait_for_termination()
