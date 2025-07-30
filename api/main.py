"""
AIvance API Main Application

FastAPI application with all routes, middleware, and configuration.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from config import settings
from core.engine import get_engine, shutdown_engine
from .routes import chat, models, sessions, memory, learning, health


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.monitoring.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AIvance API server...")
    
    # Initialize AI engine
    try:
        engine = get_engine()
        logger.info("AI Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AIvance API server...")
    try:
        await shutdown_engine()
        logger.info("AI Engine shutdown complete")
    except Exception as e:
        logger.error(f"Error during AI Engine shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="AIvance API",
    description="Advanced AI System API for natural language processing, reasoning, and learning",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)


# Add middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"in {process_time:.3f}s for {request.method} {request.url.path}"
        )
        
        return response
    
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error: {str(e)} in {process_time:.3f}s "
            f"for {request.method} {request.url.path}"
        )
        raise


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception in request {request_id}: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(learning.router, prefix="/learning", tags=["learning"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AIvance API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.debug else None
    }


@app.get("/info")
async def info():
    """Get API information."""
    engine = get_engine()
    status = await engine.get_engine_status()
    
    return {
        "name": "AIvance API",
        "version": "1.0.0",
        "environment": settings.environment,
        "ai_engine_status": status,
        "features": [
            "Natural Language Processing",
            "Multi-model Support",
            "Context Management",
            "Memory and Learning",
            "Reasoning Engine",
            "Session Management",
            "Safety and Ethics"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers
    ) 