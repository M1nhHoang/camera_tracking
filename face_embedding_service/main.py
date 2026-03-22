import sys

sys.path.append(".")

import numpy as np
import uvicorn

from PIL import Image
from io import BytesIO
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Model configuration
MODEL_NAME = "Facenet"          # 128-dim, 88MB (was VGG-Face: 4096-dim, 580MB)
NORMALIZATION = "Facenet"       # Facenet-specific normalization


@app.on_event("startup")
async def startup_event():
    """Pre-load model at startup to avoid first-request latency."""
    DeepFace.build_model(MODEL_NAME)


@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    embedding = DeepFace.represent(
        image, model_name=MODEL_NAME, normalization=NORMALIZATION,
    )

    return {"embedding": embedding[0]["embedding"]}


# run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, workers=1)
