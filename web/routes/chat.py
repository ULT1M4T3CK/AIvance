from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

# Get templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    """Serve the chat interface page"""
    return templates.TemplateResponse("chat.html", {"request": request}) 