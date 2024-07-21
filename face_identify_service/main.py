import chromadb
import requests


import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()

# connect to chromadb
chroma_client = chromadb.HttpClient(host="vector_db_service", port=8000)
collection = chroma_client.get_collection("face_detect_demo")

# init detect model
model_path = "yolo8x_facedetect.pt"
model = YOLO(model_path)

# init
conf_threshold = 0.7

# face cache
face_detect_cache = {}


def is_better_quality(new_img, old_img):
    # Chuyển đổi ảnh sang ảnh xám
    gray1 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)

    laplacian_var1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
    # laplacian_var2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

    return laplacian_var1 > 200


async def face_detect(human_image, detect_id):
    last_face = face_detect_cache.get(detect_id)

    # check face id is exit and compare quality
    if last_face is not None and not is_better_quality(human_image, last_face):
        return None

    face_detect_cache[detect_id] = human_image
    results = model(human_image)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls == 0 and conf > conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_face = human_image[y1:y2, x1:x2]
                return cropped_face

    return None


async def _face_identification(human_frame, detect_id, threshold=0.8):
    # get face in frame
    face_image = await face_detect(human_frame, detect_id)
    if face_image is not None:
        # Convert face_image to bytes
        image_bytes = BytesIO()
        Image.fromarray(face_image).save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()

        # Perform embedding
        response = requests.post(
            "http://face_embedding_service:8001/embed",
            files={"file": ("face.jpg", image_bytes, "image/jpeg")},
        )
        if response.status_code != 200:
            return "Error when connecting to embedding server"
        embedding_data = response.json()
        embedding = embedding_data["embedding"]

        # vector sreach
        sreach_results = collection.query(query_embeddings=[embedding], n_results=1)

        distances = sreach_results["distances"][0][0]
        document = sreach_results["documents"][0][0]  # must get from metadata

        # return who you are
        return (
            " ".join(document.split(".")[0].split("_")[:-1])
            if distances < threshold
            else distances
        ), image_bytes

    return None, None


@app.post("/face_identification")
async def face_identification(file: UploadFile = File(...), detect_id: int = -1):
    print(detect_id)
    if detect_id == -1:
        raise HTTPException(status_code=400, detail="Detect id is required.")

    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    identify_name, image_bytes = await _face_identification(image, detect_id)
    print(identify_name)
    if image_bytes:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    else:
        image_base64 = None
    return {"detect_name": identify_name, "face_image": image_base64}
