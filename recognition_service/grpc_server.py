import grpc
import logging
import threading
import numpy as np
from concurrent import futures
from PIL import Image
from io import BytesIO

from generated import recognition_pb2, recognition_pb2_grpc

GRPC_PORT = 50051


class TrackingInfoCache:
    """Thread-safe cache for identification results, keyed by detect_id."""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def set(self, detect_id: int, user_name: str, is_unknown: bool):
        with self._lock:
            self._cache[detect_id] = {
                "user_name": user_name,
                "is_unknown": is_unknown,
            }

    def get(self, detect_id: int):
        with self._lock:
            return self._cache.get(detect_id)

    def remove(self, detect_id: int):
        with self._lock:
            self._cache.pop(detect_id, None)


# Singleton cache instance — shared between gRPC servicer and queue processing
tracking_cache = TrackingInfoCache()


class RecognitionGrpcServicer(recognition_pb2_grpc.RecognitionServiceServicer):
    """gRPC servicer that wraps the existing RecognitionService."""

    def __init__(self, recognition_service):
        self.recognition_service = recognition_service

    def GetTrackingInfo(self, request, context):
        """Return cached identification result for a detect_id."""
        result = tracking_cache.get(request.detect_id)
        if result is None:
            return recognition_pb2.TrackingInfoResponse(
                found=False, user_name="Unknown", is_unknown=True,
            )
        return recognition_pb2.TrackingInfoResponse(
            found=True,
            user_name=result["user_name"],
            is_unknown=result["is_unknown"],
        )

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
