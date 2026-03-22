import cv2
import time
import torch
import queue
import requests
import threading
from typing import Optional, Generator, List, Dict

from yolox.tracker.byte_tracker import BYTETracker, STrack

from service.config import CameraStatus, BYTETrackerArgs
from service.detection_model import SharedDetectionModel
from service.association import associate_faces_to_persons, compute_iou
from service.grpc_client import RecognitionGrpcClient
from service.track_cache import TrackCache
from service.image_utils import encode_face_crop, encode_log_image


class DetectionService:
    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        database_service: dict,
        shared_model: SharedDetectionModel,
        grpc_client: RecognitionGrpcClient,
        camera_name: Optional[str] = None,
        location: Optional[str] = None,
        optimal_width: int = 640,
        optimal_height: int = 480,
        conf_threshold: float = 0.1,
        queue_size: int = 30,
    ):
        # Camera info
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.camera_name = camera_name
        self.location = location
        self.optimal_width = optimal_width
        self.optimal_height = optimal_height

        # Status tracking
        self._status = CameraStatus.OFFLINE
        self._stop_flag = False
        self._cap = None

        # Shared model (singleton, loaded once for all cameras)
        self.shared_model = shared_model
        self.conf_threshold = conf_threshold

        # ByteTrack is per-camera (tracks are camera-specific)
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

        # Best-shot selection + TTL cache (per-camera)
        self.track_cache = TrackCache()

        # Services
        self.database_hostname = database_service["hostname"]
        self.database_port = database_service["port"]
        self.grpc_client = grpc_client

        # Queue to store frames
        self.frame_queue = queue.Queue(maxsize=queue_size)

    @classmethod
    def create_from_config(
        cls, config: dict, services: dict, shared_model: SharedDetectionModel,
        grpc_client: RecognitionGrpcClient = None,
    ):
        """Create instance from config dictionary."""
        return cls(
            camera_id=config["camera_id"],
            stream_url=config["stream_url"],
            database_service=services["database_service"],
            shared_model=shared_model,
            grpc_client=grpc_client,
            camera_name=config.get("name"),
            location=config.get("location"),
            optimal_width=config.get("optimal_width", 640),
            optimal_height=config.get("optimal_height", 480),
            conf_threshold=config.get("conf_threshold", 0.7),
        )

    def get_status(self) -> str:
        """Get current camera status"""
        return self._status.value

    def start(self):
        """Start camera streaming"""
        self._stop_flag = False
        if self._status == CameraStatus.OFFLINE:
            video_thread = threading.Thread(target=self.video_stream, daemon=True)
            video_thread.start()

    def stop(self):
        """Stop camera streaming"""
        self._stop_flag = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._status = CameraStatus.OFFLINE
        # Clear the frame queue
        while not self.frame_queue.empty():
            self.frame_queue.get()

    def get_tracking_info(self, track_id: int) -> Optional[dict]:
        """Get tracking information from database"""
        try:
            response = requests.get(
                f"http://{self.database_hostname}:{self.database_port}/detected/get_tracking_info",
                params={"detect_id": track_id},
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting tracking info: {e}")
        return None

    def _match_track_to_face(
        self,
        track_tlbr,
        person_dets: List[list],
        face_associations: Dict[int, Optional[list]],
    ) -> Optional[list]:
        """
        Match a ByteTrack track back to original person detections,
        then return the associated face bbox (if any).

        ByteTrack may smooth/reorder bboxes, so we use IoU matching
        to find which original person detection this track corresponds to.
        """
        best_iou = 0.0
        best_idx = -1

        for i, det in enumerate(person_dets):
            iou = compute_iou(track_tlbr, det)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou > 0.5:
            return face_associations.get(best_idx)

        return None

    def detect_and_track(self, frame):
        """
        Unified detection and tracking pipeline:
        1. Single-pass YOLO inference → persons + faces
        2. Associate faces to persons (spatial matching)
        3. ByteTrack on person detections
        4. Match tracks → persons → faces
        5. Best-shot selection per track
        6. Send best shot to recognition_service when TTL expires (fire-and-forget)
        """
        detections = self.shared_model.detect(frame)
        person_dets = detections["persons"]
        face_dets = detections["faces"]
        draw_reg_list = []

        if not person_dets:
            return frame, draw_reg_list

        # Associate faces to persons before tracking
        face_associations = associate_faces_to_persons(person_dets, face_dets)

        # ByteTrack on person detections
        detections_tensor = torch.tensor(person_dets).float().cpu().numpy()
        tracks = self.byte_tracker.update(
            detections_tensor,
            [frame.shape[0], frame.shape[1]],
            [frame.shape[0], frame.shape[1]],
        )

        # Collect active track IDs for cache cleanup
        active_track_ids = set()

        for track in tracks:
            try:
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id
                active_track_ids.add(track_id)

                # Crop person image
                cropped_human = frame[y1:y2, x1:x2]
                if cropped_human.size == 0:
                    continue

                # Match track to associated face
                matched_face_bbox = self._match_track_to_face(
                    track.tlbr, person_dets, face_associations
                )

                # Crop face if associated
                cropped_face = None
                if matched_face_bbox is not None:
                    fx1, fy1, fx2, fy2 = map(int, matched_face_bbox[:4])
                    cropped_face = frame[fy1:fy2, fx1:fx2]
                    if cropped_face.size == 0:
                        cropped_face = None

                # Update best-shot cache (compare quality, keep best)
                self.track_cache.update_best_shot(
                    track_id, cropped_face, cropped_human, frame,
                )

                # Send to recognition only when TTL expires
                if self.track_cache.should_send(track_id):
                    entry = self.track_cache.get_entry(track_id)
                    self._send_to_recognition(track_id, entry)
                    self.track_cache.mark_sent(track_id)

                # Update identified status from tracking info
                tracking_info = self.get_tracking_info(track_id)
                label = "Unknown"
                if tracking_info:
                    label = tracking_info.get("user_name", "Unknown")
                    if not tracking_info.get("is_unknown", True):
                        self.track_cache.mark_identified(track_id)

                draw_reg_list.append((x1, y1, x2, y2, label))

            except Exception as e:
                print(f"Error processing track {track_id}: {e}")
                continue

        # Cleanup stale tracks from cache
        self.track_cache.cleanup(active_track_ids)

        return frame, draw_reg_list

    def _send_to_recognition(self, track_id: int, entry) -> None:
        """
        Encode and send best-shot via gRPC fire-and-forget.
        - Face crop: 160x160 JPEG Q=70 (~3-5KB) — realtime, for embedding
        - Origin + detect: JPEG Q=80 (~40KB each) — for logging, encoded in background
        """
        # Realtime path: face crop (small + fast)
        face_image_bytes = b""
        if entry.best_face is not None:
            face_image_bytes = encode_face_crop(entry.best_face)

        # Deferred path: log images (heavier encoding in background thread)
        detect_np = entry.best_detect
        origin_np = entry.best_origin

        def _encode_and_send():
            detect_bytes = encode_log_image(detect_np)
            origin_bytes = encode_log_image(origin_np)
            self.grpc_client.identify_face_async(
                detect_id=track_id,
                camera_id=self.camera_id,
                origin_image_bytes=origin_bytes,
                detect_image_bytes=detect_bytes,
                face_image_bytes=face_image_bytes,
            )

        threading.Thread(target=_encode_and_send, daemon=True).start()

    def video_stream(self):
        """Process video stream"""
        self._cap = cv2.VideoCapture(self.stream_url)
        prev_time = 0
        error_count = 0
        max_errors = 5

        while not self._stop_flag:
            time.sleep(0.01)

            try:
                self._status = CameraStatus.STREAMING
                ret, frame = self._cap.read()
                if not ret:
                    error_count += 1
                    if error_count >= max_errors:
                        self._status = CameraStatus.ERROR
                        break
                    continue

                error_count = 0  # Reset error count on successful frame read

                # Unified detect and track (person + face in single pass)
                frame, draw_reg_list = self.detect_and_track(frame)

                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # Draw bounding boxes and labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 3
                for x1, y1, x2, y2, label in draw_reg_list:
                    # Ensure label is string
                    label = str(label) if label is not None else "Unknown"

                    # Draw thicker bounding box with greater thickness
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )

                    # Draw background rectangle for text
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        (0, 255, 0),
                        -1,
                    )

                    # Draw text with greater size and thickness
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 5),
                        font,
                        font_scale,
                        (0, 0, 0),  # Black text on green background
                        thickness,
                        cv2.LINE_AA,
                    )

                # Display FPS with background
                fps_text = f"FPS: {fps:.2f}"
                (fps_width, fps_height), _ = cv2.getTextSize(
                    fps_text, font, font_scale, thickness
                )

                # Draw background rectangle for FPS
                cv2.rectangle(
                    frame, (10, 10), (20 + fps_width, 20 + fps_height), (0, 255, 0), -1
                )

                # Draw FPS text
                cv2.putText(
                    frame,
                    fps_text,
                    (15, 15 + fps_height),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness,
                    cv2.LINE_AA,
                )

                # Resize frame if needed
                frame = cv2.resize(frame, (self.optimal_width, self.optimal_height))

                # Encode frame
                _, jpeg = cv2.imencode(".jpg", frame)
                frame_bytes = jpeg.tobytes()

                # Update frame queue
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame_bytes)

            except Exception as e:
                print(f"Error in video stream: {e}")
                error_count += 1
                if error_count >= max_errors:
                    self._status = CameraStatus.ERROR
                    break
                time.sleep(1)

        if self._cap is not None:
            self._cap.release()
        self._status = CameraStatus.OFFLINE

    def video_feed(self) -> Generator[bytes, None, None]:
        """Generate video feed"""
        try:
            while True:
                if self._status != CameraStatus.STREAMING:
                    break
                frame_bytes = self.frame_queue.get()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        except Exception as e:
            print(f"Error in video feed: {e}")
            self._status = CameraStatus.ERROR
