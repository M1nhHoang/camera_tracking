import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from router import detected, users, cameras
import os

# init app
app = FastAPI()

# Mount static files
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "static_files"
)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(detected.router, prefix="/detected", tags=["detected"])
app.include_router(cameras.router, prefix="/cameras", tags=["cameras"])

# run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, workers=1)
