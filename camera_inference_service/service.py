import cv2
import time
import torch
import queue
import requests

from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker, STrack


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
        database_serivce: dict,
        face_idtification_service: dict,
        model_path: str,
        stream_url: str,
        optimal_width=640,
        optimal_height=480,
        conf_threshold=0.7,
        queue_size=30,
    ):
        # init variables
        self.stream_url = stream_url
        self.optimal_width = optimal_width
        self.optimal_height = optimal_height

        # init detect model
        self.model = YOLO(model_path)

        # Confidence threshold
        self.conf_threshold = conf_threshold

        # create BYTETracker instance
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

        # Initialize database service
        self.database_hostname = database_serivce["hostname"]
        self.database_port = database_serivce["port"]

        # Initialize face identification service
        self.face_idtification_hostname = face_idtification_service["hostname"]
        self.face_idtification_port = face_idtification_service["port"]

        # Queue to store frames
        self.frame_queue = queue.Queue(maxsize=queue_size)

    def get_tracking_info(self, track_id):
        response = requests.get(
            f"http://{self.database_hostname}:{self.database_port}/detected/get_tracking_info?detect_id={track_id}"
        )

        if response.status_code == 200:
            return response.json()

        return None

    def human_detect_and_track(self, frame):
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
                # get track data
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id

                # Convert human image to bytes
                cropped_human = frame[y1:y2, x1:x2]

                # valid empty image because bug from the void
                if cropped_human.size != 0:
                    image_bytes = BytesIO()

                    # detect image to byte
                    Image.fromarray(cropped_human).save(image_bytes, format="JPEG")
                    detect_image_bytes = image_bytes.getvalue()

                    # frame to byte
                    Image.fromarray(frame).save(image_bytes, format="JPEG")
                    origin_image_bytes = image_bytes.getvalue()

                    # Perform identification
                start_time = time.time()
                response = requests.post(
                    f"http://{self.face_idtification_hostname}:{self.face_idtification_port}/face_identification?detect_id={track_id}",
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
                end_time = time.time()
                print(f"Process request time: {end_time - start_time} seconds")

                start_time = time.time()
                tracking_info = self.get_tracking_info(track_id)
                if not tracking_info:
                    label = "Error"
                else:
                    label = tracking_info["user_name"]
                end_time = time.time()
                print(f"Process traking info time: {end_time - start_time} seconds")

                draw_reg_list.append((x1, y1, x2, y2, label))

        return frame, draw_reg_list

    def video_stream(self):
        cap = cv2.VideoCapture(self.stream_url)
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and track humans
            frame, draw_reg_list = self.human_detect_and_track(frame)

            # define frame width and height
            height, width, _ = frame.shape

            # Resize the frame for optimal performance
            frame = cv2.resize(frame, (self.optimal_width, self.optimal_height))

            # Draw bounding boxes
            for x1, y1, x2, y2, label in draw_reg_list:
                # Scale position to optimal size
                x1 = int(x1 * self.optimal_width / width)
                y1 = int(y1 * self.optimal_height / height)
                x2 = int(x2 * self.optimal_width / width)
                y2 = int(y2 * self.optimal_height / height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Display FPS on the frame
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Encode the frame in JPEG format
            _, jpeg = cv2.imencode(".jpg", frame)
            frame_bytes = jpeg.tobytes()

            # Add frame to queue
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame_bytes)

        cap.release()

    def video_feed(self):
        while True:
            frame_bytes = self.frame_queue.get()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
