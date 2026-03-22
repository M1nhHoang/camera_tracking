from typing import Optional, List, Dict


def associate_faces_to_persons(
    persons: List[list], faces: List[list]
) -> Dict[int, Optional[list]]:
    """
    Associate each face to the person whose bbox contains the face center.

    Args:
        persons: [[x1, y1, x2, y2, conf], ...]
        faces:   [[x1, y1, x2, y2, conf], ...]

    Returns:
        {person_index: face_bbox or None}
        If multiple faces fall inside one person, highest confidence wins.
    """
    associations: Dict[int, Optional[list]] = {}

    for pi, person in enumerate(persons):
        px1, py1, px2, py2, _ = person
        best_face = None
        best_conf = -1.0

        for face in faces:
            fx1, fy1, fx2, fy2, fconf = face
            face_cx = (fx1 + fx2) / 2.0
            face_cy = (fy1 + fy2) / 2.0

            if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                if fconf > best_conf:
                    best_face = face
                    best_conf = fconf

        associations[pi] = best_face

    return associations


def compute_iou(box_a, box_b) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2, ...]."""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
