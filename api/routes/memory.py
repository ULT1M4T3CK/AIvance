"""
Memory API Routes

Routes for memory management and retrieval.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from core.memory import memory_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class MemoryItem(BaseModel):
    """Memory item response model."""
    
    id: str
    content: str
    user_id: Optional[str]
    memory_type: str
    importance: float
    timestamp: str
    tags: List[str] = []


@router.get("/user/{user_id}")
async def get_user_memories(user_id: str, limit: int = 50):
    """Get memories for a specific user."""
    try:
        memories = await memory_manager.get_user_memories(user_id, limit)
        
        memory_list = []
        for memory in memories:
            memory_list.append(MemoryItem(
                id=memory.id,
                content=memory.content,
                user_id=memory.user_id,
                memory_type=memory.memory_type,
                importance=memory.importance,
                timestamp=memory.timestamp.isoformat(),
                tags=memory.tags
            ))
        
        return {
            "user_id": user_id,
            "memories": memory_list,
            "total_count": len(memory_list)
        }
    
    except Exception as e:
        logger.error(f"Error getting user memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_memories(
    query: str,
    user_id: Optional[str] = None,
    limit: int = 10
):
    """Search memories based on a query."""
    try:
        memories = await memory_manager.get_relevant_memories(query, user_id, limit)
        
        memory_list = []
        for memory in memories:
            memory_list.append(MemoryItem(
                id=memory.id,
                content=memory.content,
                user_id=memory.user_id,
                memory_type=memory.memory_type,
                importance=memory.importance,
                timestamp=memory.timestamp.isoformat(),
                tags=memory.tags
            ))
        
        return {
            "query": query,
            "user_id": user_id,
            "memories": memory_list,
            "total_found": len(memory_list)
        }
    
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store")
async def store_memory(
    content: str,
    user_id: Optional[str] = None,
    memory_type: str = "fact",
    importance: float = 0.5,
    tags: List[str] = []
):
    """Store a new memory."""
    try:
        memory_id = await memory_manager.store_fact(
            content=content,
            user_id=user_id,
            importance=importance,
            tags=tags
        )
        
        if memory_id:
            return {
                "message": "Memory stored successfully",
                "memory_id": memory_id,
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store memory")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory."""
    try:
        success = await memory_manager.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"message": "Memory deleted successfully", "memory_id": memory_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_memory_statistics():
    """Get memory system statistics."""
    try:
        stats = await memory_manager.get_memory_statistics()
        return stats
    
    except Exception as e:
        logger.error(f"Error getting memory statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_memories(days_old: int = 30, min_importance: float = 0.3):
    """Clean up old, low-importance memories."""
    try:
        deleted_count = await memory_manager.cleanup_old_memories(days_old, min_importance)
        return {
            "message": f"Cleaned up {deleted_count} old memories",
            "deleted_count": deleted_count,
            "days_old": days_old,
            "min_importance": min_importance
        }
    
    except Exception as e:
        logger.error(f"Error cleaning up memories: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 