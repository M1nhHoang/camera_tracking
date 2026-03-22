import threading
from typing import Optional, Dict

from ultralytics import YOLO

from service.config import DetectionConfig


class SharedDetectionModel:
    """
    Thread-safe singleton that loads the YOLO model once
    and shares it across all camera instances.

    Usage:
        model = SharedDetectionModel.get_instance(config)
        detections = model.detect(frame)
        # detections = {"persons": [...], "faces": [...]}
    """

    _instance: Optional["SharedDetectionModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config: DetectionConfig):
        self._model = YOLO(config.model_path)
        self._config = config
        self._inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: DetectionConfig = None) -> "SharedDetectionModel":
        """Get or create the singleton instance (double-checked locking)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if config is None:
                        raise ValueError("Config required for first initialization")
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing or model hot-reload)."""
        with cls._lock:
            cls._instance = None

    # TODO: Support batch inference - accept list of frames from multiple cameras
    #       and run a single model(batch) call for better GPU throughput.
    def detect(self, frame) -> Dict[str, list]:
        """
        Run unified inference on a single frame.

        Returns:
            {"persons": [[x1,y1,x2,y2,conf], ...],
             "faces":   [[x1,y1,x2,y2,conf], ...]}
        """
        with self._inference_lock:
            results = self._model(frame)

        persons = []
        faces = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (
                    cls == self._config.person_class_id
                    and conf > self._config.person_conf_threshold
                ):
                    persons.append([x1, y1, x2, y2, conf])
                elif (
                    cls == self._config.face_class_id
                    and conf > self._config.face_conf_threshold
                ):
                    faces.append([x1, y1, x2, y2, conf])

        return {"persons": persons, "faces": faces}

    @property
    def config(self) -> DetectionConfig:
        return self._config
