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
        "camera_id": None,
        "force_update": False,
    },
):
    # init
    user_id = traking_data.get("user_id", None)
    camera_id = traking_data.get("camera_id", None)
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
        user_id=user_id,
        camera_id=camera_id,
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
    sort_by: Optional[str] = "time_stamp",
    sort_order: Optional[str] = "desc",
):
    """Get paginated list of detections with filters

    Parameters:
    - date_from: Optional start date filter (YYYY-MM-DD)
    - date_to: Optional end date filter (YYYY-MM-DD)
    - user_id: Optional user ID filter
    - camera_id: Optional camera ID filter
    - page: Page number (starts from 1)
    - per_page: Number of items per page (10-100)
    - sort_by: Field to sort by (time_stamp, distance, etc)
    - sort_order: Sort order (asc/desc)
    """
    try:
        # Build filter query
        query = {}

        # Date filters
        if date_from or date_to:
            query["time_stamp"] = {}
            if date_from:
                from_date = datetime.strptime(date_from, "%Y-%m-%d")
                query["time_stamp"]["$gte"] = from_date.strftime("%d-%m-%Y %H:%M:%S")
            if date_to:
                to_date = datetime.strptime(date_to, "%Y-%m-%d")
                query["time_stamp"]["$lte"] = to_date.strftime("%d-%m-%Y %H:%M:%S")

        # User filter
        if user_id:
            query["user_id"] = ObjectId(user_id)

        # Camera filter
        if camera_id:
            query["camera_id"] = ObjectId(camera_id)

        # Calculate pagination
        skip = (page - 1) * per_page

        # Get DB collection
        from database import MongoDBManager

        db = MongoDBManager(collection_name="detected_logs")

        # Get total count for pagination
        total_count = len(db.find_all(query))
        total_pages = (total_count + per_page - 1) // per_page

        # Sort params
        sort_direction = -1 if sort_order == "desc" else 1

        # Get paginated records
        records = (
            db.get_collection()
            .find(query)
            .skip(skip)
            .limit(per_page)
            .sort(sort_by, sort_direction)
        )

        # Format response
        detections = []
        for record in records:
            # Get user info
            user_db = MongoDBManager(collection_name="users")
            user = user_db.find_one({"_id": record["user_id"]})
            username = user["username"] if user else "Unknown"

            # Get camera info if exists
            camera_name = "Unknown"
            if "camera_id" in record:
                camera_db = MongoDBManager(collection_name="cameras")
                # Ensure the camera_id is a valid ObjectId
                if isinstance(record["camera_id"], str):
                    camera_id_obj = ObjectId(record["camera_id"])
                else:
                    camera_id_obj = record["camera_id"]

                # Fetch camera details using the ObjectId
                camera = camera_db.find_one({"_id": camera_id_obj})

                if camera:
                    camera_name = camera["name"]

            detection = {
                "id": str(record["_id"]),
                "user_id": str(record["user_id"]),
                "user_name": username,
                "camera_name": camera_name,
                "time_stamp": record["time_stamp"],
                "distance": record["distance"],
                "origin_image_path": record.get("origin_image_path"),
                "face_image_path": record.get("face_image_path"),
                "detect_image_path": record.get("detect_image_path"),
                "truth_image_path": record.get("truth_image_path"),
            }
            detections.append(detection)

        return {
            "total_records": total_count,
            "total_pages": total_pages,
            "current_page": page,
            "per_page": per_page,
            "detections": detections,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/{detection_id}")
async def get_detection(detection_id: str):
    """Get detection details by ID"""
    try:
        service = DetectedService()
        detection = await service.get_detection(detection_id)

        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")

        return detection
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
