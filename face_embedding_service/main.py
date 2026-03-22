import sys
sys.path.append(".")

import threading
import numpy as np
import uvicorn

from PIL import Image
from io import BytesIO
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile
from grpc_server import start_grpc_server

app = FastAPI()

MODEL_NAME = "Facenet"
NORMALIZATION = "Facenet"


@app.on_event("startup")
async def startup_event():
    """Pre-load model at startup."""
    DeepFace.build_model(MODEL_NAME)


@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    """HTTP fallback for embedding (primary path is gRPC)."""
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    embedding = DeepFace.represent(
        image, model_name=MODEL_NAME, normalization=NORMALIZATION,
    )

    return {"embedding": embedding[0]["embedding"]}


# Start gRPC server alongside FastAPI (port 50052)
grpc_thread = threading.Thread(target=start_grpc_server, daemon=True)
grpc_thread.start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, workers=1)
