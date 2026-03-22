import time
import cv2
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass, field


# TTL constants (seconds)
TTL_UNKNOWN = 3.0      # Unknown tracks: retry every 3s
TTL_IDENTIFIED = 300.0  # Identified tracks: retry every 5 minutes


@dataclass
class TrackEntry:
    """Cached state for a single tracked person."""
    track_id: int
    best_face: Optional[np.ndarray] = None       # Best face crop (numpy)
    best_face_quality: float = -1.0               # Laplacian variance of best face
    best_detect: Optional[np.ndarray] = None      # Best person crop at time of best face
    best_origin: Optional[np.ndarray] = None      # Best origin frame at time of best face
    last_sent_time: float = 0.0                   # Last time we sent to recognition
    is_identified: bool = False                    # Whether this track has been identified
    send_count: int = 0                           # Number of times sent to recognition


class TrackCache:
    """
    Per-camera cache that tracks best-shot selection and TTL-based sending.

    Logic:
    - Each frame: compare face quality, keep best shot
    - Every TTL: send best shot to recognition service
    - Unknown tracks: TTL = 3s (retry frequently)
    - Identified tracks: TTL = 5min (rarely re-verify)
    - Tracks are cleaned up when ByteTrack stops reporting them
    """

    def __init__(self):
        self._entries: Dict[int, TrackEntry] = {}

    def update_best_shot(
        self,
        track_id: int,
        face_crop: Optional[np.ndarray],
        detect_crop: np.ndarray,
        origin_frame: np.ndarray,
    ) -> None:
        """
        Update best shot for a track if the new face is better quality.
        Always updates detect_crop and origin_frame if face is better.
        """
        if track_id not in self._entries:
            self._entries[track_id] = TrackEntry(track_id=track_id)

        entry = self._entries[track_id]

        if face_crop is None or face_crop.size == 0:
            # No face this frame — still store detect/origin if first time
            if entry.best_detect is None:
                entry.best_detect = detect_crop
                entry.best_origin = origin_frame
            return

        # Compute face quality (Laplacian variance — higher = sharper)
        quality = self._compute_quality(face_crop)

        if quality > entry.best_face_quality:
            entry.best_face = face_crop
            entry.best_face_quality = quality
            entry.best_detect = detect_crop
            entry.best_origin = origin_frame

    def should_send(self, track_id: int) -> bool:
        """Check if this track should be sent to recognition service based on TTL."""
        entry = self._entries.get(track_id)
        if entry is None:
            return False

        # Nothing to send if no detect crop
        if entry.best_detect is None:
            return False

        now = time.time()
        ttl = TTL_IDENTIFIED if entry.is_identified else TTL_UNKNOWN
        return (now - entry.last_sent_time) >= ttl

    def mark_sent(self, track_id: int) -> None:
        """Mark that we just sent this track to recognition."""
        entry = self._entries.get(track_id)
        if entry:
            entry.last_sent_time = time.time()
            entry.send_count += 1

    def mark_identified(self, track_id: int) -> None:
        """Mark track as identified (switch to longer TTL)."""
        entry = self._entries.get(track_id)
        if entry:
            entry.is_identified = True

    def get_entry(self, track_id: int) -> Optional[TrackEntry]:
        return self._entries.get(track_id)

    def cleanup(self, active_track_ids: set) -> None:
        """Remove entries for tracks that ByteTrack no longer reports."""
        stale = [tid for tid in self._entries if tid not in active_track_ids]
        for tid in stale:
            del self._entries[tid]

    @staticmethod
    def _compute_quality(image: np.ndarray) -> float:
        """Compute image sharpness via Laplacian variance."""
        if image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
