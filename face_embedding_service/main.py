import sys

sys.path.append(".")

import numpy as np

from PIL import Image
from io import BytesIO
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    # Perform embedding
    embedding = DeepFace.represent(image, model_name="VGG-Face")

    return {"embedding": embedding[0]["embedding"]}
