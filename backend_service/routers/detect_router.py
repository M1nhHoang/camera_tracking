from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from services.detect_service import DetectionService
from datetime import datetime

router = APIRouter()


@router.get("/list")
async def list_detections(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    user_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    sort_by: Optional[str] = "time_stamp",
    sort_order: Optional[str] = "asc",
    service: DetectionService = Depends(DetectionService),
):
    """Get filtered list of detections with pagination"""
    try:
        # Validate date format if provided
        if date_from:
            try:
                datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD"
                )

        if date_to:
            try:
                datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD"
                )

        # Validate sort parameters
        valid_sort_fields = ["time_stamp", "distance", "user_name", "camera_name"]
        if sort_by and sort_by not in valid_sort_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by field. Must be one of: {', '.join(valid_sort_fields)}",
            )

        valid_sort_orders = ["desc", "desc"]
        if sort_order and sort_order not in valid_sort_orders:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_order. Must be one of: {', '.join(valid_sort_orders)}",
            )

        # Get detections from service
        result = await service.get_detections(
            page=page,
            per_page=per_page,
            date_from=date_from,
            date_to=date_to,
            user_id=user_id,
            camera_id=camera_id,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{detection_id}")
async def get_detection(
    detection_id: str, service: DetectionService = Depends(DetectionService)
):
    """Get detection details by ID"""
    try:
        detection = await service.get_detection(detection_id)
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        return detection
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/daily")
async def get_daily_stats(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    service: DetectionService = Depends(DetectionService),
):
    """Get daily detection statistics"""
    try:
        # Validate date format if provided
        if date_from:
            try:
                datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD"
                )

        if date_to:
            try:
                datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD"
                )

        stats = await service.get_daily_stats(date_from, date_to)
        return stats
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/users")
async def get_user_stats(service: DetectionService = Depends(DetectionService)):
    """Get detection statistics by user"""
    try:
        stats = await service.get_user_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/cameras")
async def get_camera_stats(service: DetectionService = Depends(DetectionService)):
    """Get detection statistics by camera"""
    try:
        stats = await service.get_camera_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{detection_id}/update_user")
async def update_detection_user(
    detection_id: str,
    update_data: dict,
    service: DetectionService = Depends(DetectionService),
):
    """Update user info in detection log"""
    try:
        success = await service.update_detection_user(detection_id, update_data)
        if not success:
            raise HTTPException(status_code=404, detail="Detection not found")
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
