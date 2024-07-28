import typing as t

from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import JSONResponse
from service.detected import DetectedService

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
