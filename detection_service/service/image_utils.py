import cv2
import numpy as np

# JPEG quality settings
FACE_JPEG_QUALITY = 70   # Realtime path — small, fast
LOG_JPEG_QUALITY = 80     # Deferred path — good enough for UI display

# Face crop target size (matches embedding model input)
FACE_TARGET_SIZE = (160, 160)


def encode_face_crop(face_image: np.ndarray) -> bytes:
    """
    Resize face crop to model input size and encode as JPEG Q=70.
    Realtime path — optimized for small size (~3-5KB).
    """
    resized = cv2.resize(face_image, FACE_TARGET_SIZE)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, FACE_JPEG_QUALITY])
    return buf.tobytes()


def encode_log_image(image: np.ndarray) -> bytes:
    """
    Encode image as JPEG Q=80 for logging/UI display.
    Deferred path — good visual quality, ~40-60% smaller than default.
    """
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, LOG_JPEG_QUALITY])
    return buf.tobytes()
