import sys

sys.path.append(".")

import numpy as np
import uvicorn

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


# run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, workers=2)
