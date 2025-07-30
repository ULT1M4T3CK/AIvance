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

logger = logging.getLogger(__name__)


def create_web_app() -> FastAPI:
    """Create the main web application."""
    
    # Create FastAPI app
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
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
    )
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Setup templates
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    templates = Jinja2Templates(directory=str(templates_dir))
    
    # Store templates in app state
    app.state.templates = templates
    
    # Add routes
    from .routes import dashboard, chat, settings, analytics
    
    app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(settings.router, prefix="/settings", tags=["settings"])
    app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
    
    # Root route
    @app.get("/")
    async def root(request: Request):
        """Root route - redirect to dashboard."""
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "title": "AIvance - Advanced AI System"}
        )
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            # Get metrics collector
            metrics = get_metrics_collector()
            
            return {
                "status": "healthy",
                "version": "1.0.0",
                "metrics": metrics.get_metrics_summary()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")
    
    # Error handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        """Handle 404 errors."""
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "title": "Page Not Found"},
            status_code=404
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: HTTPException):
        """Handle 500 errors."""
        return templates.TemplateResponse(
            "500.html",
            {"request": request, "title": "Internal Server Error"},
            status_code=500
        )
    
    logger.info("Web application created successfully")
    return app


def create_dashboard_app() -> FastAPI:
    """Create a simplified dashboard app for embedded use."""
    
    app = FastAPI(
        title="AIvance Dashboard",
        description="Embedded AI Dashboard",
        version="1.0.0",
        docs_url=None,
        redoc_url=None
    )
    
    # Add CORS for embedded use
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Setup templates
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates
    
    # Dashboard routes
    from .routes import dashboard
    
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    
    # Dashboard UI
    @app.get("/")
    async def dashboard_ui(request: Request):
        """Dashboard UI."""
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "title": "AIvance Dashboard"}
        )
    
    return app 