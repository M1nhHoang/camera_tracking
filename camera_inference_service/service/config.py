from enum import Enum
from dataclasses import dataclass


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


@dataclass
class DetectionConfig:
    """Configuration for the shared detection model."""

    model_path: str
    person_conf_threshold: float = 0.5
    face_conf_threshold: float = 0.5
    person_class_id: int = 0
    face_class_id: int = 1
