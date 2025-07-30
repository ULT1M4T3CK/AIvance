"""
AIvance Web Application

This module creates the main web application with dashboard and UI components.
"""

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pathlib import Path
from config import settings
from auth.security import get_current_active_user
from monitoring.metrics import get_metrics_collector

# Import route modules
from .routes import dashboard, chat

logger = logging.getLogger(__name__)


def create_web_app() -> FastAPI:
    """Create and configure the FastAPI web application"""
    
    app = FastAPI(
        title="AIvance Dashboard",
        description="Advanced AI System Dashboard",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.security.allowed_hosts
        )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Set up templates
    templates_dir = Path(__file__).parent / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    # Include routers
    app.include_router(dashboard.router, tags=["dashboard"])
    app.include_router(chat.router, tags=["chat"])

    @app.get("/")
    async def root(request: Request):
        """Root endpoint - redirect to dashboard"""
        return {"message": "AIvance AI System", "version": "1.0.0"}

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            metrics_collector = get_metrics_collector()
            return {
                "status": "healthy",
                "version": "1.0.0",
                "metrics_collector": "active" if metrics_collector else "inactive"
            }
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        """Custom 404 handler"""
        return {
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": request.url.path
        }

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: HTTPException):
        """Custom 500 handler"""
        logging.error(f"Internal server error: {exc}")
        return {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }

    return app

def create_dashboard_app() -> FastAPI:
    """Create a minimal dashboard app for embedded use"""
    app = FastAPI(
        title="AIvance Dashboard",
        description="Embedded AI System Dashboard",
        version="1.0.0",
        docs_url=None,
        redoc_url=None
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Set up templates
    templates_dir = Path(__file__).parent / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    # Include only dashboard routes
    app.include_router(dashboard.router, tags=["dashboard"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "version": "1.0.0"}

    return app 