import cv2
import time
import torch
import queue
import requests
import threading
from enum import Enum
from typing import Optional, Generator

from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker, STrack


class CameraStatus(Enum):
    OFFLINE = "offline"
    STREAMING = "streaming"
    PAUSED = "paused"
    ERROR = "error"


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 60
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class CameraInferenceService:
    def __init__(
        self,
        stream_url: str,
        database_service: dict,
        face_identify_service: dict,
        model_path: str,
        camera_name: Optional[str] = None,
        location: Optional[str] = None,
        optimal_width: int = 640,
        optimal_height: int = 480,
        conf_threshold: float = 0.1,
        queue_size: int = 30,
    ):
        # Camera info
        self.stream_url = stream_url
        self.camera_name = camera_name
        self.location = location
        self.optimal_width = optimal_width
        self.optimal_height = optimal_height

        # Status tracking
        self._status = CameraStatus.OFFLINE
        self._stop_flag = False
        self._cap = None

        # init detect model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

        # Initialize services
        self.database_hostname = database_service["hostname"]
        self.database_port = database_service["port"]
        self.face_identify_hostname = face_identify_service["hostname"]
        self.face_identify_port = face_identify_service["port"]

        # Queue to store frames
        self.frame_queue = queue.Queue(maxsize=queue_size)

    @classmethod
    def create_from_config(cls, config: dict, services: dict, model_path: str):
        """Create instance from config dictionary"""
        return cls(
            stream_url=config["stream_url"],
            database_service=services["database_service"],
            face_identify_service=services["face_identify_service"],
            model_path=model_path,
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

    def human_detect_and_track(self, frame):
        """Detect and track humans in frame"""
        results = self.model(frame)
        detections = []
        draw_reg_list = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if cls == 0 and conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2, conf])

        if detections:
            detections_tensor = torch.tensor(detections).float().cpu().numpy()
            tracks = self.byte_tracker.update(
                detections_tensor,
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]],
            )

            for track in tracks:
                try:
                    # Get track data
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    track_id = track.track_id

                    # Crop human image
                    cropped_human = frame[y1:y2, x1:x2]
                    if cropped_human.size == 0:
                        continue

                    # Convert images to bytes
                    image_bytes = BytesIO()
                    Image.fromarray(cropped_human).save(image_bytes, format="JPEG")
                    detect_image_bytes = image_bytes.getvalue()

                    image_bytes = BytesIO()
                    Image.fromarray(frame).save(image_bytes, format="JPEG")
                    origin_image_bytes = image_bytes.getvalue()

                    # Send to face identification service
                    response = requests.post(
                        f"http://{self.face_identify_hostname}:{self.face_identify_port}/face_identification",
                        params={"detect_id": track_id},
                        files={
                            "origin_image": (
                                "origin_image.jpg",
                                origin_image_bytes,
                                "image/jpeg",
                            ),
                            "detect_image": (
                                "detect_image.jpg",
                                detect_image_bytes,
                                "image/jpeg",
                            ),
                        },
                    )

                    # Get tracking info
                    tracking_info = self.get_tracking_info(track_id)
                    label = (
                        tracking_info.get("user_name", "Unknown")
                        if tracking_info
                        else "Unknown"
                    )

                    draw_reg_list.append((x1, y1, x2, y2, label))

                except Exception as e:
                    print(f"Error processing track {track_id}: {e}")
                    continue

        return frame, draw_reg_list

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

                # Detect and track humans
                frame, draw_reg_list = self.human_detect_and_track(frame)

                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # Draw bounding boxes and labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 3
                for x1, y1, x2, y2, label in draw_reg_list:
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
