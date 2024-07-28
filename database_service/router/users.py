import typing as t

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from service.detected import UserService

router = APIRouter()


@router.post("/user_update")
async def tracking(
    user_data: dict = {
        "identifier": "",
        "user_name": "",
        "face_images": [],
    },
    user_service: UserService = Depends(UserService),
):
    # init
    identifier = user_data["identifier"]
    user_name = user_data["user_name"]
    face_images = user_data["face_images"]

    # traking
    user_id, face_images = user_service.user_update(user_name, identifier, face_images)

    if user_id:
        return JSONResponse(
            status_code=200,
            content={"user_id": user_id, "face_image_path": face_images[0]},
        )

    return Response(status_code=500)
