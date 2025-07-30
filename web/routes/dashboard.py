from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

# Get templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_alt(request: Request):
    """Alternative route for dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request}) 