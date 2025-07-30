"""
Sessions API Routes

Routes for session management and information.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from core.session import session_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionInfo(BaseModel):
    """Session information response model."""
    
    session_id: str
    user_id: Optional[str]
    created_at: str
    last_activity: str
    status: str
    interaction_count: int


@router.get("/", response_model=List[SessionInfo])
async def list_sessions(user_id: Optional[str] = None):
    """List all sessions, optionally filtered by user."""
    try:
        if user_id:
            sessions = await session_manager.get_user_sessions(user_id)
        else:
            # Get all sessions (for admin purposes)
            sessions = session_manager.store.get_all_sessions()
        
        session_list = []
        for session in sessions:
            session_list.append(SessionInfo(
                session_id=session.session_id,
                user_id=session.user_id,
                created_at=session.created_at.isoformat(),
                last_activity=session.last_activity.isoformat(),
                status=session.status.value,
                interaction_count=session.get_interaction_count()
            ))
        
        return session_list
    
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_session_info(session_id: str):
    """Get detailed information about a specific session."""
    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "status": session.status.value,
            "interaction_count": session.get_interaction_count(),
            "session_duration": session.get_session_duration().total_seconds(),
            "idle_time": session.get_idle_time().total_seconds(),
            "is_expired": session.is_expired()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def close_session(session_id: str):
    """Close a specific session."""
    try:
        success = await session_manager.close_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session closed successfully", "session_id": session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/overview")
async def get_sessions_statistics():
    """Get overview statistics for all sessions."""
    try:
        stats = await session_manager.get_session_statistics()
        return stats
    
    except Exception as e:
        logger.error(f"Error getting session statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_expired_sessions(max_age_hours: int = 24):
    """Clean up expired sessions."""
    try:
        deleted_count = await session_manager.cleanup_expired_sessions(max_age_hours)
        return {
            "message": f"Cleaned up {deleted_count} expired sessions",
            "deleted_count": deleted_count,
            "max_age_hours": max_age_hours
        }
    
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 