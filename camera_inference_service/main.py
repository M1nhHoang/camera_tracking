import sys

sys.path.append(f"ByteTrack")

import cv2
import time
import torch
import uvicorn
import requests

from fastapi import FastAPI, Request
from io import BytesIO
from PIL import Image
from starlette.responses import StreamingResponse
from ultralytics import YOLO

from database import Database
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch
from yolox.tracker.byte_tracker import BYTETracker, STrack


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 60
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


app = FastAPI()

# Initialize YOLOv8 model
model_path = "yolo8n_humman_detect.pt"
model = YOLO(model_path)

# Initialize Deep Sort
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())

# Confidence threshold
conf_threshold = 0.7

# tracking cache
tracking_cache = {}

# Video source
video_path = "demo.avi"

# Optimal frame size
optimal_width = 640
optimal_height = 480

# connect to mongodb
db = Database()


def human_detect_and_track(frame):
    results = model(frame)
    detections = []
    draw_reg_list = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls == 0 and conf > conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

    if detections:
        detections_tensor = torch.tensor(detections).float().cpu().numpy()
        tracks = byte_tracker.update(
            detections_tensor,
            [frame.shape[0], frame.shape[1]],
            [frame.shape[0], frame.shape[1]],
        )

        for track in tracks:
            # get track data
            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = track.track_id

            label = tracking_cache.get(track_id)
            if not label:
                detect_name = None
                face_image_base64 = None

                # Convert humman_image to bytes
                cropped_humman = frame[y1:y2, x1:x2]

                # valid empty image because bug from the void
                if cropped_humman.size != 0:
                    image_bytes = BytesIO()
                    Image.fromarray(cropped_humman).save(image_bytes, format="JPEG")
                    image_bytes = image_bytes.getvalue()

                    # Perform indentification
                    response = requests.post(
                        f"http://face_identify_service:8002/face_identification?detect_id={track_id}",
                        files={"file": ("face.jpg", image_bytes, "image/jpeg")},
                    )
                    if response.status_code != 200:
                        return frame, draw_reg_list
                    detect_infos = response.json()
                    detect_name = detect_infos["detect_name"]
                    face_image_base64 = detect_infos["face_image"]

                if detect_name and isinstance(detect_name, str):
                    label = f"{detect_name}"
                    tracking_cache[track_id] = label
                    db.save_to_database(
                        track_id,
                        label,
                        frame,
                        cropped_humman,
                        face_image_base64,
                        force_update=True,
                    )
                else:
                    label = (
                        "Unknown: " + str(round(float(detect_name), 2))
                        if detect_name
                        else "--"
                    )

                if face_image_base64:
                    db.save_to_database(
                        track_id, label, frame, cropped_humman, face_image_base64
                    )

            draw_reg_list.append((x1, y1, x2, y2, label))

    return frame, draw_reg_list


async def video_stream(request: Request):
    cap = cv2.VideoCapture(video_path)
    prev_time = 0
    while cap.isOpened():
        if await request.is_disconnected():
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track humans
        frame, draw_reg_list = human_detect_and_track(frame)

        # define frame width and height
        height, width, _ = frame.shape

        # Resize the frame for optimal performance
        frame = cv2.resize(frame, (optimal_width, optimal_height))

        # Draw bounding boxes
        print(draw_reg_list)
        for x1, y1, x2, y2, label in draw_reg_list:
            # Scale position to optimal size
            x1 = int(x1 * optimal_width / width)
            y1 = int(y1 * optimal_height / height)
            x2 = int(x2 * optimal_width / width)
            y2 = int(y2 * optimal_height / height)

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
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Encode the frame in JPEG format
        _, jpeg = cv2.imencode(".jpg", frame)
        frame_bytes = jpeg.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.get("/camera")
async def camera_feed(request: Request):
    return StreamingResponse(
        video_stream(request), media_type="multipart/x-mixed-replace; boundary=frame"
    )
