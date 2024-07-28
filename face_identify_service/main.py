import uvicorn

import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException

from service import FaceIdentifyService

app = FastAPI()

# env get
# face_identify_service = os.getenv("FACE_IDENTIFY_SERVICE")

database_service = {"name": "database_service", "port": 8003}
embedding_service = {"name": "face_embedding_service", "port": 8001}
vector_db_service = {"name": "chroma_db", "port": 8000}
model_path = "weights/yolo8n_face_detect.pt"

face_identify_service = FaceIdentifyService(
    database_service=database_service,
    embedding_service=embedding_service,
    vector_db_service=vector_db_service,
    model_path=model_path,
)


@app.post("/face_identification")
async def face_identification(
    origin_image: UploadFile = File(...),
    detect_image: UploadFile = File(...),
    detect_id: int = -1,
):
    if detect_id == -1:
        raise HTTPException(status_code=400, detail="Detect id is required.")

    origin_image = Image.open(BytesIO(await origin_image.read()))
    detect_image = Image.open(BytesIO(await detect_image.read()))

    # numpy array
    origin_image = np.array(origin_image)
    detect_image = np.array(detect_image)

    face_identify_service.process_detect_queue(detect_id, origin_image, detect_image)

    return {"success": True}


@app.post("/face_upload")
async def face_upload(
    file: UploadFile = File(...),
    identifier: str = "",
    user_name: str = "",
):
    # try:
    face_identify_service.process_face_image_upload(
        {"identifier": identifier, "user_name": user_name}, await file.read()
    )
    return {"success": True}
    # except Exception as e:
    #     print(e)
    #     return Response(status_code=500, content=str(e))


# run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, workers=1)
