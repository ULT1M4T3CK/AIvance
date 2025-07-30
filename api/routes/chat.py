"""
Chat API Routes

Routes for AI chat interactions and conversations.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from core.engine import get_engine, EngineRequest, EngineResponse
from core.session import session_manager

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    
    message: str = Field(..., description="User message")
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    model: Optional[str] = Field(None, description="AI model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Response temperature")
    max_tokens: Optional[int] = Field(4096, ge=1, le=8192, description="Maximum tokens")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    model_used: str = Field(..., description="Model used for response")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    reasoning_steps: Optional[list] = Field(None, description="Reasoning steps")
    safety_checks: Optional[Dict[str, Any]] = Field(None, description="Safety check results")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")


class ChatHistoryRequest(BaseModel):
    """Request model for chat history."""
    
    session_id: str = Field(..., description="Session ID")
    limit: Optional[int] = Field(50, ge=1, le=100, description="Number of messages to retrieve")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    
    session_id: str = Field(..., description="Session ID")
    messages: list = Field(..., description="Chat messages")
    total_messages: int = Field(..., description="Total number of messages")


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request
):
    """
    Send a message to the AI and get a response.
    
    This endpoint handles:
    - Message processing through the AI engine
    - Session management
    - Context building
    - Safety checks
    - Response generation
    """
    try:
        engine = get_engine()
        
        # Create engine request
        engine_request = EngineRequest(
            prompt=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            metadata=request.metadata or {}
        )
        
        # Process request through AI engine
        engine_response = await engine.process_request(engine_request)
        
        # Build response
        response = ChatResponse(
            response=engine_response.content,
            session_id=engine_response.session_id,
            model_used=engine_response.model_used,
            confidence_score=engine_response.confidence_score,
            reasoning_steps=engine_response.reasoning_steps,
            safety_checks=engine_response.safety_checks,
            usage=engine_response.usage,
            metadata=engine_response.metadata
        )
        
        logger.info(f"Chat response generated for user: {request.user_id}")
        return response
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response (for real-time interactions).
    
    This endpoint provides streaming responses for better user experience.
    """
    try:
        engine = get_engine()
        
        # Create engine request
        engine_request = EngineRequest(
            prompt=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            metadata=request.metadata or {}
        )
        
        # For now, return a simple response
        # TODO: Implement actual streaming
        engine_response = await engine.process_request(engine_request)
        
        return {
            "response": engine_response.content,
            "session_id": engine_response.session_id,
            "model_used": engine_response.model_used,
            "streaming": True
        }
    
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(request: ChatHistoryRequest):
    """
    Get chat history for a session.
    
    Returns the conversation history for the specified session.
    """
    try:
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get interactions from session
        interactions = session.interactions[-request.limit:] if session.interactions else []
        
        # Format messages
        messages = []
        for interaction in interactions:
            messages.append({
                "timestamp": interaction.timestamp.isoformat(),
                "user_message": interaction.user_message,
                "assistant_response": interaction.assistant_response,
                "model_used": interaction.model_used,
                "processing_time": interaction.processing_time
            })
        
        response = ChatHistoryResponse(
            session_id=request.session_id,
            messages=messages,
            total_messages=len(session.interactions)
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def close_chat_session(session_id: str):
    """
    Close a chat session.
    
    This will terminate the session and clean up resources.
    """
    try:
        success = await session_manager.close_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session closed successfully", "session_id": session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    session_id: str,
    message_id: str,
    feedback: str,
    score: Optional[float] = Field(None, ge=0.0, le=1.0),
    user_id: Optional[str] = None
):
    """
    Submit feedback for a chat response.
    
    This helps the AI system learn and improve over time.
    """
    try:
        from core.learning import learning_engine
        
        # Get session to find the interaction
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Find the specific interaction
        interaction = None
        for inter in session.interactions:
            if inter.id == message_id:
                interaction = inter
                break
        
        if not interaction:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Learn from feedback
        await learning_engine.learn_from_feedback(
            user_input=interaction.user_message,
            ai_response=interaction.assistant_response,
            feedback=feedback,
            feedback_score=score,
            user_id=user_id,
            session_id=session_id
        )
        
        return {
            "message": "Feedback submitted successfully",
            "session_id": session_id,
            "message_id": message_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
async def get_chat_suggestions(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context: Optional[str] = None
):
    """
    Get chat suggestions based on context and user history.
    
    Returns suggested messages or topics for the user.
    """
    try:
        suggestions = []
        
        # Get user preferences if available
        if user_id:
            from core.memory import memory_manager
            preferences = await memory_manager.get_user_preferences(user_id)
            
            if preferences and preferences.preferred_topics:
                suggestions.extend([
                    f"Tell me more about {topic}" for topic in preferences.preferred_topics[:3]
                ])
        
        # Add general suggestions
        general_suggestions = [
            "What can you help me with?",
            "Can you explain how you work?",
            "What are your capabilities?",
            "How can I get the best results from you?"
        ]
        
        suggestions.extend(general_suggestions)
        
        return {
            "suggestions": suggestions[:10],  # Limit to 10 suggestions
            "user_id": user_id,
            "session_id": session_id
        }
    
    except Exception as e:
        logger.error(f"Error getting chat suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 