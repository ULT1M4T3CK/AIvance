"""
Session Management System

This module handles user sessions, conversation history, and session state
for the AI system.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status enumeration."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class SessionInteraction:
    """A single interaction within a session."""
    
    id: str
    user_message: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInteraction':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AISession:
    """An AI session representing a conversation with a user."""
    
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    status: SessionStatus = SessionStatus.ACTIVE
    interactions: List[SessionInteraction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate session ID if not provided."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
    
    def add_interaction(
        self, 
        user_message: str, 
        assistant_response: str,
        model_used: Optional[str] = None,
        processing_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an interaction to the session."""
        interaction = SessionInteraction(
            id=str(uuid.uuid4()),
            user_message=user_message,
            assistant_response=assistant_response,
            model_used=model_used,
            processing_time=processing_time,
            metadata=metadata or {}
        )
        
        self.interactions.append(interaction)
        self.last_activity = datetime.utcnow()
        
        # Keep only last 50 interactions to prevent memory overflow
        if len(self.interactions) > 50:
            self.interactions = self.interactions[-50:]
    
    def get_interaction_count(self) -> int:
        """Get the number of interactions in the session."""
        return len(self.interactions)
    
    def get_session_duration(self) -> timedelta:
        """Get the duration of the session."""
        return datetime.utcnow() - self.created_at
    
    def get_idle_time(self) -> timedelta:
        """Get the time since last activity."""
        return datetime.utcnow() - self.last_activity
    
    def is_expired(self, max_idle_hours: int = 24) -> bool:
        """Check if the session has expired due to inactivity."""
        return self.get_idle_time().total_seconds() > (max_idle_hours * 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        data['status'] = self.status.value
        data['interactions'] = [interaction.to_dict() for interaction in self.interactions]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AISession':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        data['status'] = SessionStatus(data['status'])
        data['interactions'] = [
            SessionInteraction.from_dict(interaction_data)
            for interaction_data in data['interactions']
        ]
        return cls(**data)


class SessionStore:
    """Base class for session storage backends."""
    
    async def store_session(self, session: AISession) -> bool:
        """Store a session."""
        raise NotImplementedError
    
    async def get_session(self, session_id: str) -> Optional[AISession]:
        """Get a session by ID."""
        raise NotImplementedError
    
    async def update_session(self, session: AISession) -> bool:
        """Update a session."""
        raise NotImplementedError
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        raise NotImplementedError
    
    async def get_user_sessions(self, user_id: str) -> List[AISession]:
        """Get all sessions for a user."""
        raise NotImplementedError
    
    async def cleanup_expired_sessions(self, max_idle_hours: int = 24) -> int:
        """Clean up expired sessions."""
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    """In-memory session storage implementation."""
    
    def __init__(self):
        self.sessions: Dict[str, AISession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self.logger = logging.getLogger(f"{__name__}.InMemorySessionStore")
    
    async def store_session(self, session: AISession) -> bool:
        """Store a session."""
        try:
            self.sessions[session.session_id] = session
            
            # Track user sessions
            if session.user_id:
                if session.user_id not in self.user_sessions:
                    self.user_sessions[session.user_id] = []
                if session.session_id not in self.user_sessions[session.user_id]:
                    self.user_sessions[session.user_id].append(session.session_id)
            
            self.logger.debug(f"Stored session: {session.session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[AISession]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        if session:
            # Update last activity
            session.last_activity = datetime.utcnow()
        return session
    
    async def update_session(self, session: AISession) -> bool:
        """Update a session."""
        try:
            if session.session_id in self.sessions:
                self.sessions[session.session_id] = session
                self.logger.debug(f"Updated session: {session.session_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating session: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Remove from user sessions tracking
                if session.user_id and session.user_id in self.user_sessions:
                    if session_id in self.user_sessions[session.user_id]:
                        self.user_sessions[session.user_id].remove(session_id)
                
                del self.sessions[session_id]
                self.logger.debug(f"Deleted session: {session_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting session: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[AISession]:
        """Get all sessions for a user."""
        session_ids = self.user_sessions.get(user_id, [])
        sessions = []
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session:
                sessions.append(session)
        
        # Sort by last activity (newest first)
        sessions.sort(key=lambda x: x.last_activity, reverse=True)
        return sessions
    
    async def cleanup_expired_sessions(self, max_idle_hours: int = 24) -> int:
        """Clean up expired sessions."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(max_idle_hours):
                expired_sessions.append(session_id)
        
        deleted_count = 0
        for session_id in expired_sessions:
            if await self.delete_session(session_id):
                deleted_count += 1
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} expired sessions")
        
        return deleted_count
    
    def get_all_sessions(self) -> List[AISession]:
        """Get all sessions (for debugging/monitoring)."""
        return list(self.sessions.values())


class SessionManager:
    """
    Main session manager that handles all session operations.
    
    This class manages:
    - Session creation and lifecycle
    - Session storage and retrieval
    - Session cleanup and maintenance
    - Session statistics and monitoring
    """
    
    def __init__(self, store: Optional[SessionStore] = None):
        self.store = store or InMemorySessionStore()
        self.active_sessions: Dict[str, AISession] = {}
        self.logger = logging.getLogger(f"{__name__}.SessionManager")
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self.cleanup_expired_sessions()
                except Exception as e:
                    self.logger.error(f"Error in cleanup loop: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info("Started session cleanup task")
    
    async def create_session(self, user_id: Optional[str] = None) -> AISession:
        """Create a new session."""
        try:
            session = AISession(
                session_id=str(uuid.uuid4()),
                user_id=user_id
            )
            
            # Store session
            success = await self.store.store_session(session)
            if success:
                self.active_sessions[session.session_id] = session
                self.logger.info(f"Created new session: {session.session_id} for user: {user_id}")
                return session
            else:
                raise RuntimeError("Failed to store session")
        
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[AISession]:
        """Get a session by ID."""
        try:
            # Check active sessions first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.utcnow()
                return session
            
            # Try to get from store
            session = await self.store.get_session(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                # Add to active sessions
                self.active_sessions[session_id] = session
                return session
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting session: {e}")
            return None
    
    async def add_interaction(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str,
        model_used: Optional[str] = None,
        processing_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an interaction to a session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # Add interaction
            session.add_interaction(
                user_message=user_message,
                assistant_response=assistant_response,
                model_used=model_used,
                processing_time=processing_time,
                metadata=metadata
            )
            
            # Update session in store
            await self.store.update_session(session)
            
            self.logger.debug(f"Added interaction to session: {session_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding interaction: {e}")
            return False
    
    async def update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.status = status
            session.last_activity = datetime.utcnow()
            
            # Update in store
            await self.store.update_session(session)
            
            # Remove from active sessions if terminated
            if status == SessionStatus.TERMINATED:
                self.active_sessions.pop(session_id, None)
            
            self.logger.info(f"Updated session {session_id} status to: {status.value}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error updating session status: {e}")
            return False
    
    async def close_session(self, session_id: str) -> bool:
        """Close a session."""
        return await self.update_session_status(session_id, SessionStatus.TERMINATED)
    
    async def get_user_sessions(self, user_id: str) -> List[AISession]:
        """Get all sessions for a user."""
        try:
            return await self.store.get_user_sessions(user_id)
        except Exception as e:
            self.logger.error(f"Error getting user sessions: {e}")
            return []
    
    async def cleanup_expired_sessions(self, max_idle_hours: int = 24) -> int:
        """Clean up expired sessions."""
        try:
            # Clean up from store
            deleted_count = await self.store.cleanup_expired_sessions(max_idle_hours)
            
            # Clean up active sessions
            expired_active = []
            for session_id, session in self.active_sessions.items():
                if session.is_expired(max_idle_hours):
                    expired_active.append(session_id)
            
            for session_id in expired_active:
                session = self.active_sessions[session_id]
                session.status = SessionStatus.EXPIRED
                await self.store.update_session(session)
                del self.active_sessions[session_id]
            
            if expired_active:
                self.logger.info(f"Cleaned up {len(expired_active)} expired active sessions")
            
            return deleted_count + len(expired_active)
        
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    async def close_all_sessions(self):
        """Close all active sessions."""
        try:
            for session_id in list(self.active_sessions.keys()):
                await self.close_session(session_id)
            
            self.logger.info("Closed all active sessions")
        
        except Exception as e:
            self.logger.error(f"Error closing all sessions: {e}")
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            all_sessions = self.store.get_all_sessions()
            
            total_sessions = len(all_sessions)
            active_sessions = len(self.active_sessions)
            
            # Count by status
            status_counts = {}
            for session in all_sessions:
                status = session.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Average interactions per session
            total_interactions = sum(session.get_interaction_count() for session in all_sessions)
            avg_interactions = total_interactions / total_sessions if total_sessions > 0 else 0
            
            # Average session duration
            total_duration = sum(session.get_session_duration().total_seconds() for session in all_sessions)
            avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "status_distribution": status_counts,
                "average_interactions": avg_interactions,
                "average_duration_seconds": avg_duration,
                "total_interactions": total_interactions
            }
        
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the session manager."""
        try:
            # Stop cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Close all sessions
            await self.close_all_sessions()
            
            self.logger.info("Session manager shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global session manager instance
session_manager = SessionManager() 