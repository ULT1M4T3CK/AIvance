"""
Health Check API Routes

Routes for system health monitoring and status checks.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from core.engine import get_engine
from core.session import session_manager
from core.memory import memory_manager
from core.learning import learning_engine
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = "healthy"
    version: str = "1.0.0"
    environment: str = "development"
    components: Dict[str, Any] = {}
    timestamp: str = ""


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns the overall system health status.
    """
    try:
        # Get AI engine status
        engine = get_engine()
        engine_status = await engine.get_engine_status()
        
        # Get session statistics
        session_stats = await session_manager.get_session_statistics()
        
        # Get memory statistics
        memory_stats = await memory_manager.get_memory_statistics()
        
        # Get learning statistics
        learning_stats = await learning_engine.get_learning_statistics()
        
        # Determine overall status
        overall_status = "healthy"
        if engine_status.get("models_available", 0) == 0:
            overall_status = "degraded"
        
        response = HealthResponse(
            status=overall_status,
            version="1.0.0",
            environment=settings.environment,
            components={
                "ai_engine": engine_status,
                "sessions": session_stats,
                "memory": memory_stats,
                "learning": learning_stats
            },
            timestamp=__import__("datetime").datetime.utcnow().isoformat()
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Checks if the system is ready to handle requests.
    """
    try:
        # Check AI engine
        engine = get_engine()
        engine_status = await engine.get_engine_status()
        
        if engine_status.get("models_available", 0) == 0:
            raise HTTPException(status_code=503, detail="No AI models available")
        
        # Check session manager
        session_stats = await session_manager.get_session_statistics()
        
        # Check memory manager
        memory_stats = await memory_manager.get_memory_statistics()
        
        return {
            "status": "ready",
            "components": {
                "ai_engine": "ready",
                "session_manager": "ready",
                "memory_manager": "ready"
            },
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint.
    
    Checks if the system is alive and responding.
    """
    return {
        "status": "alive",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint.
    
    Provides comprehensive system health information.
    """
    try:
        # Get all component statuses
        engine = get_engine()
        engine_status = await engine.get_engine_status()
        session_stats = await session_manager.get_session_statistics()
        memory_stats = await memory_manager.get_memory_statistics()
        learning_stats = await learning_engine.get_learning_statistics()
        
        # Check configuration
        config_status = {
            "database_configured": bool(settings.database.url),
            "ai_models_configured": bool(settings.ai.openai_api_key or settings.ai.anthropic_api_key),
            "security_configured": bool(settings.security.secret_key),
            "environment": settings.environment,
            "debug_mode": settings.debug
        }
        
        # Determine overall health
        health_issues = []
        
        if engine_status.get("models_available", 0) == 0:
            health_issues.append("No AI models available")
        
        if not config_status["database_configured"]:
            health_issues.append("Database not configured")
        
        if not config_status["ai_models_configured"]:
            health_issues.append("AI models not configured")
        
        overall_status = "healthy" if not health_issues else "degraded"
        
        return {
            "status": overall_status,
            "version": "1.0.0",
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "issues": health_issues,
            "components": {
                "ai_engine": engine_status,
                "session_manager": session_stats,
                "memory_manager": memory_stats,
                "learning_engine": learning_stats,
                "configuration": config_status
            },
            "metrics": {
                "total_sessions": session_stats.get("total_sessions", 0),
                "active_sessions": session_stats.get("active_sessions", 0),
                "total_memories": memory_stats.get("total_memories", 0),
                "total_users": memory_stats.get("total_users", 0),
                "available_models": engine_status.get("models_available", 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")


@router.get("/metrics")
async def get_metrics():
    """
    Get system metrics for monitoring.
    
    Returns key performance indicators and system metrics.
    """
    try:
        # Get component statistics
        engine = get_engine()
        engine_status = await engine.get_engine_status()
        session_stats = await session_manager.get_session_statistics()
        memory_stats = await memory_manager.get_memory_statistics()
        learning_stats = await learning_engine.get_learning_statistics()
        
        # Calculate metrics
        metrics = {
            "system": {
                "uptime": "N/A",  # TODO: Implement uptime tracking
                "version": "1.0.0",
                "environment": settings.environment
            },
            "ai_engine": {
                "available_models": engine_status.get("models_available", 0),
                "total_models": engine_status.get("total_models", 0),
                "memory_enabled": engine_status.get("memory_enabled", False),
                "learning_enabled": engine_status.get("learning_enabled", False),
                "reasoning_enabled": engine_status.get("reasoning_enabled", False),
                "safety_enabled": engine_status.get("safety_enabled", False)
            },
            "sessions": {
                "total_sessions": session_stats.get("total_sessions", 0),
                "active_sessions": session_stats.get("active_sessions", 0),
                "average_interactions": session_stats.get("average_interactions", 0),
                "average_duration": session_stats.get("average_duration_seconds", 0)
            },
            "memory": {
                "total_memories": memory_stats.get("total_memories", 0),
                "total_users": memory_stats.get("total_users", 0),
                "average_importance": memory_stats.get("average_importance", 0),
                "embedding_model_available": memory_stats.get("embedding_model_available", False)
            },
            "learning": {
                "total_events": learning_stats.get("total_events", 0),
                "total_patterns": learning_stats.get("total_patterns", 0),
                "total_insights": learning_stats.get("total_insights", 0),
                "learning_enabled": learning_stats.get("learning_enabled", True)
            }
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics") 