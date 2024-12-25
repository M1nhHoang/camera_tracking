from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

router = APIRouter()

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set up templates directory
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/cameras", response_class=HTMLResponse)
async def cameras(request: Request):
    return templates.TemplateResponse("cameras.html", {"request": request})


@router.get("/users", response_class=HTMLResponse)
async def users(request: Request):
    return templates.TemplateResponse("users.html", {"request": request})


@router.get("/detect", response_class=HTMLResponse)
async def detect(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})


@router.get("/camera/{camera_id}", response_class=HTMLResponse)
async def camera_detail(request: Request, camera_id: str):
    return templates.TemplateResponse(
        "camera_detail.html", {"request": request, "camera_id": camera_id}
    )
