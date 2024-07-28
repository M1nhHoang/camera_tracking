import uvicorn
from fastapi import FastAPI
from router import detected, users

# init app
app = FastAPI()

# Include router
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(detected.router, prefix="/detected", tags=["detected"])

# run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, workers=1)
