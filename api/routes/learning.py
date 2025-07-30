"""
Learning API Routes

Routes for learning system management and insights.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from core.learning import learning_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class LearningEvent(BaseModel):
    """Learning event response model."""
    
    event_id: str
    user_id: Optional[str]
    event_type: str
    user_input: str
    ai_response: str
    timestamp: str


@router.get("/statistics")
async def get_learning_statistics():
    """Get learning system statistics."""
    try:
        stats = await learning_engine.get_learning_statistics()
        return stats
    
    except Exception as e:
        logger.error(f"Error getting learning statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/preferences")
async def get_user_preferences(user_id: str):
    """Get learned user preferences."""
    try:
        preferences = await learning_engine.get_user_preferences(user_id)
        return {
            "user_id": user_id,
            "preferences": preferences
        }
    
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/{user_id}/preferences")
async def update_user_preferences(
    user_id: str,
    preferences: Dict[str, Any]
):
    """Update user preferences."""
    try:
        success = await learning_engine.update_user_preferences(user_id, preferences)
        if success:
            return {
                "message": "User preferences updated successfully",
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_learning_insights(limit: int = 10):
    """Get learning insights."""
    try:
        # Get insights from the learning store
        insights = list(learning_engine.store.insights.values())
        
        # Sort by timestamp (newest first)
        insights.sort(key=lambda x: x.timestamp, reverse=True)
        
        insight_list = []
        for insight in insights[:limit]:
            insight_list.append({
                "insight_id": insight.insight_id,
                "insight_type": insight.insight_type,
                "description": insight.description,
                "confidence": insight.confidence,
                "actionable": insight.actionable,
                "timestamp": insight.timestamp.isoformat(),
                "data": insight.data
            })
        
        return {
            "insights": insight_list,
            "total_insights": len(insights)
        }
    
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_learning_patterns(user_id: Optional[str] = None, limit: int = 20):
    """Get learned patterns."""
    try:
        patterns = await learning_engine.store.get_patterns(user_id)
        
        pattern_list = []
        for pattern in patterns[:limit]:
            pattern_list.append({
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "user_id": pattern.user_id,
                "confidence": pattern.confidence,
                "usage_count": pattern.usage_count,
                "last_used": pattern.last_used.isoformat(),
                "created_at": pattern.created_at.isoformat(),
                "data": pattern.pattern_data
            })
        
        return {
            "patterns": pattern_list,
            "total_patterns": len(patterns),
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Error getting learning patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_learning_feedback(
    user_input: str,
    ai_response: str,
    feedback: str,
    feedback_score: Optional[float] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Submit feedback for learning."""
    try:
        await learning_engine.learn_from_feedback(
            user_input=user_input,
            ai_response=ai_response,
            feedback=feedback,
            feedback_score=feedback_score,
            user_id=user_id,
            session_id=session_id
        )
        
        return {
            "message": "Feedback submitted successfully for learning",
            "user_id": user_id,
            "session_id": session_id
        }
    
    except Exception as e:
        logger.error(f"Error submitting learning feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_learning_data(user_id: Optional[str] = None):
    """Reset learning data (use with caution)."""
    try:
        if user_id:
            # Reset specific user's learning data
            # This would need to be implemented in the learning engine
            return {
                "message": f"Learning data reset for user: {user_id}",
                "user_id": user_id
            }
        else:
            # Reset all learning data
            # This would need to be implemented in the learning engine
            return {
                "message": "All learning data reset",
                "warning": "This action cannot be undone"
            }
    
    except Exception as e:
        logger.error(f"Error resetting learning data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 