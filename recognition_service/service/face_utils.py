import cv2
import base64
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO


class FaceProcessor:
    """Face detection, validation, and image utilities."""

    def __init__(self, model_path: str, conf_threshold: float = 0.85):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray, return_all: bool = False):
        """
        Detect faces in image.
        Args:
            image: numpy array (BGR/RGB)
            return_all: if True return list of all faces, else return first face
        Returns:
            Single face crop (numpy) or list of face crops
        """
        faces = []
        results = self.model(image)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if cls == 0 and conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped = image[y1:y2, x1:x2]
                    if not return_all:
                        return cropped
                    faces.append(cropped)
        return faces

    def validate(self, image: np.ndarray, quality_threshold: int = 500) -> np.ndarray:
        """
        Validate image has exactly 1 face with good quality.
        Returns the cropped face.
        """
        faces = self.detect(image, return_all=True)
        if len(faces) != 1:
            raise ValueError("Must be only one face in your image.")

        face = faces[0]
        if not self.is_quality(face, quality_threshold):
            raise ValueError("Image quality is not good.")

        return face

    @staticmethod
    def is_quality(image: np.ndarray, threshold: int = 200) -> bool:
        """Check image sharpness via Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold

    @staticmethod
    def to_base64(image) -> str:
        """Convert image (numpy/bytes/str) to base64 string."""
        if isinstance(image, str):
            return image
        elif isinstance(image, np.ndarray):
            img_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        elif isinstance(image, bytes):
            img_data = cv2.imdecode(
                np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            img_bytes = cv2.imencode(".jpg", img_data)[1].tobytes()
        else:
            raise ValueError("Unsupported image format")
        return base64.b64encode(img_bytes).decode("utf-8")
