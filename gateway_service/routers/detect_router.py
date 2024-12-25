from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from services.detect_service import DetectionService

router = APIRouter()


class DetectionFilter(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    user_id: Optional[str] = None
    camera_id: Optional[str] = None
    page: int = 1
    per_page: int = 10


@router.get("/list")
async def list_detections(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    service: DetectionService = Depends(DetectionService),
):
    """Get list of detections with filters and pagination"""
    filters = {
        "date_from": date_from,
        "date_to": date_to,
        "user_id": user_id,
        "camera_id": camera_id,
        "page": page,
        "per_page": per_page,
    }

    detections = await service.get_detections(filters)
    return detections


@router.get("/{detection_id}")
async def get_detection(
    detection_id: str, service: DetectionService = Depends(DetectionService)
):
    """Get detection details by ID"""
    detection = await service.get_detection(detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return detection


@router.get("/stats/daily")
async def get_daily_stats(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    service: DetectionService = Depends(DetectionService),
):
    """Get daily detection statistics"""
    stats = await service.get_daily_stats(date_from, date_to)
    return stats


@router.get("/stats/users")
async def get_user_stats(service: DetectionService = Depends(DetectionService)):
    """Get detection statistics by user"""
    stats = await service.get_user_stats()
    return stats


@router.get("/stats/cameras")
async def get_camera_stats(service: DetectionService = Depends(DetectionService)):
    """Get detection statistics by camera"""
    stats = await service.get_camera_stats()
    return stats
