import grpc
import logging

from generated import recognition_pb2, recognition_pb2_grpc


class RecognitionGrpcClient:
    """
    Thread-safe gRPC client for recognition service.
    Shared across all DetectionService instances.
    gRPC channels are inherently thread-safe.
    """

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
        self.stub = recognition_pb2_grpc.RecognitionServiceStub(self.channel)
        logging.info(f"gRPC client initialized targeting {self.target}")

    def identify_face_async(
        self,
        detect_id: int,
        camera_id: str,
        origin_image_bytes: bytes,
        detect_image_bytes: bytes,
        face_image_bytes: bytes = b"",
    ):
        """
        Non-blocking fire-and-forget. Returns a gRPC Future.
        Callback logs errors only — caller does not wait for result.
        """
        request = recognition_pb2.FaceIdentificationRequest(
            detect_id=detect_id,
            camera_id=camera_id,
            origin_image=origin_image_bytes,
            detect_image=detect_image_bytes,
            face_image=face_image_bytes,
        )
        future = self.stub.IdentifyFace.future(request, timeout=5.0)
        future.add_done_callback(self._on_identify_done)
        return future

    @staticmethod
    def _on_identify_done(future):
        """Callback for async identify — only logs errors."""
        try:
            result = future.result()
            if not result.success:
                logging.warning(f"IdentifyFace returned success=False: {result.message}")
        except grpc.RpcError as e:
            logging.error(f"IdentifyFace async error: {e.code()}: {e.details()}")

    def get_tracking_info(self, detect_id: int) -> dict:
        """
        Get cached tracking info from recognition service.
        Lightweight sync call — no images, just detect_id → user_name.
        """
        try:
            request = recognition_pb2.TrackingInfoRequest(detect_id=detect_id)
            response = self.stub.GetTrackingInfo(request, timeout=1.0)
            if response.found:
                return {
                    "user_name": response.user_name,
                    "is_unknown": response.is_unknown,
                }
        except grpc.RpcError as e:
            logging.error(f"GetTrackingInfo error: {e.code()}: {e.details()}")
        return None

    def close(self):
        self.channel.close()
