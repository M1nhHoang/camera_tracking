import typing as t

from fastapi import APIRouter, Response, HTTPException, Query
from fastapi.responses import JSONResponse
from service.detected import DetectedService
from datetime import datetime
from typing import Optional, List
from bson import ObjectId

router = APIRouter()

# init
global detected_service
detected_service = DetectedService()


@router.post("/tracking")
async def tracking(
    traking_data: dict = {
        "user_id": "",
        "detect_id": "",
        "origin_image": "",
        "detect_image": "",
        "face_image": "",
        "truth_image_path": "",
        "distance": 0,
        "force_update": False,
    },
):
    # init
    user_id = traking_data.get("user_id", None)
    detect_id = traking_data["detect_id"]
    origin_image = traking_data["origin_image"]
    detect_image = traking_data["detect_image"]
    face_image = traking_data["face_image"]
    truth_image_path = traking_data["truth_image_path"]
    distance = traking_data["distance"]
    force_update = traking_data.get("force_update", False)

    # traking
    detected_service.traking(
        detect_id,
        origin_image,
        detect_image,
        face_image,
        truth_image_path,
        distance,
        force_update,
        user_id,
    )

    return Response(status_code=200)


@router.get("/get_tracking_info")
async def get_tracking_info(detect_id: int = -1):
    # valid
    if detect_id == -1:
        raise HTTPException(status_code=400, detail="Detect id is required.")

    user_name, is_unknown = detected_service.get_tracking_info(detect_id)

    return JSONResponse(
        status_code=200, content={"user_name": user_name, "is_unknown": is_unknown}
    )

    # return face_identify_service.get_name_by_detect_id(detect_id)


@router.get("/list")
async def list_detections(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
):
    """Get filtered list of detections with pagination"""
    service = DetectedService()

    # Convert string dates to datetime if provided
    from_date = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
    to_date = datetime.strptime(date_to, "%Y-%m-%d") if date_to else None

    # Convert string IDs to ObjectId
    user_obj_id = ObjectId(user_id) if user_id else None
    camera_obj_id = ObjectId(camera_id) if camera_id else None

    detections = await service.get_detections(
        from_date=from_date,
        to_date=to_date,
        user_id=user_obj_id,
        camera_id=camera_obj_id,
        page=page,
        per_page=per_page,
    )
    return detections


@router.get("/stats/daily")
async def get_daily_stats(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """Get daily detection statistics"""
    service = DetectedService()
    from_date = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
    to_date = datetime.strptime(date_to, "%Y-%m-%d") if date_to else None

    stats = await service.get_daily_stats(from_date, to_date)
    return stats


@router.get("/stats/users")
async def get_user_stats():
    """Get detection statistics by user"""
    service = DetectedService()
    stats = await service.get_user_stats()
    return stats


@router.get("/stats/cameras")
async def get_camera_stats():
    """Get detection statistics by camera"""
    service = DetectedService()
    stats = await service.get_camera_stats()
    return stats
