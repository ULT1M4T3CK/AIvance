"""
Context Management System

This module handles conversation context, user preferences, session information,
and context building for the AI system.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context information."""
    
    SESSION = "session"
    USER_PREFERENCES = "user_preferences"
    MEMORY = "memory"
    SYSTEM = "system"
    CONVERSATION = "conversation"
    ENVIRONMENTAL = "environmental"


@dataclass
class ContextItem:
    """A single piece of context information."""
    
    type: ContextType
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority items are more important
    
    def __post_init__(self):
        """Validate context item after initialization."""
        if not isinstance(self.type, ContextType):
            raise ValueError("type must be a ContextType enum")
        if not self.key:
            raise ValueError("key cannot be empty")


@dataclass
class Memory:
    """A memory item that can be stored and retrieved."""
    
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Context:
    """
    Main context class that manages all context information for a conversation.
    
    This class handles:
    - Session information
    - User preferences
    - Conversation history
    - System context
    - Memory integration
    """
    
    def __init__(self):
        self.items: Dict[str, ContextItem] = {}
        self.memories: List[Memory] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.logger = logging.getLogger(f"{__name__}.Context")
    
    def add_item(self, context_type: ContextType, key: str, value: Any, **kwargs):
        """Add a context item."""
        item = ContextItem(
            type=context_type,
            key=key,
            value=value,
            **kwargs
        )
        self.items[f"{context_type.value}_{key}"] = item
        self.logger.debug(f"Added context item: {context_type.value}_{key}")
    
    def get_item(self, context_type: ContextType, key: str) -> Optional[Any]:
        """Get a context item by type and key."""
        item_key = f"{context_type.value}_{key}"
        item = self.items.get(item_key)
        return item.value if item else None
    
    def get_items_by_type(self, context_type: ContextType) -> List[ContextItem]:
        """Get all context items of a specific type."""
        return [
            item for item in self.items.values()
            if item.type == context_type
        ]
    
    def add_session_context(self, session):
        """Add session-related context."""
        if hasattr(session, 'session_id'):
            self.add_item(ContextType.SESSION, "session_id", session.session_id)
        if hasattr(session, 'user_id'):
            self.add_item(ContextType.SESSION, "user_id", session.user_id)
        if hasattr(session, 'created_at'):
            self.add_item(ContextType.SESSION, "created_at", session.created_at)
        if hasattr(session, 'last_activity'):
            self.add_item(ContextType.SESSION, "last_activity", session.last_activity)
    
    def add_user_preferences(self, preferences: Dict[str, Any]):
        """Add user preferences to context."""
        for key, value in preferences.items():
            self.add_item(ContextType.USER_PREFERENCES, key, value)
    
    def add_system_context(self, system_info: Dict[str, Any]):
        """Add system-related context."""
        for key, value in system_info.items():
            self.add_item(ContextType.SYSTEM, key, value)
    
    def add_memories(self, memories: List[Memory]):
        """Add memory items to context."""
        self.memories.extend(memories)
        # Also add as context items for easy access
        for i, memory in enumerate(memories):
            self.add_item(
                ContextType.MEMORY,
                f"memory_{i}",
                memory.content,
                metadata={"importance": memory.importance, "tags": memory.tags}
            )
    
    def add_conversation_turn(self, user_message: str, assistant_message: str):
        """Add a conversation turn to the history."""
        turn = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.utcnow()
        }
        self.conversation_history.append(turn)
        
        # Keep only last 10 turns to prevent context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_current_topic(self) -> Optional[str]:
        """Extract the current conversation topic."""
        if not self.conversation_history:
            return None
        
        # Simple topic extraction from the last user message
        last_message = self.conversation_history[-1]["user"]
        # TODO: Implement more sophisticated topic extraction
        return last_message[:100]  # First 100 characters as topic
    
    def get_session_context(self) -> Dict[str, Any]:
        """Get all session-related context."""
        session_items = self.get_items_by_type(ContextType.SESSION)
        return {item.key: item.value for item in session_items}
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get all user preferences."""
        pref_items = self.get_items_by_type(ContextType.USER_PREFERENCES)
        return {item.key: item.value for item in pref_items}
    
    def get_system_context(self) -> Dict[str, Any]:
        """Get all system context."""
        system_items = self.get_items_by_type(ContextType.SYSTEM)
        return {item.key: item.value for item in system_items}
    
    def get_memories(self) -> List[Memory]:
        """Get all memories in context."""
        return self.memories.copy()
    
    def get_session_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def get_relevant_memories(self, topic: str, limit: int = 5) -> List[Memory]:
        """Get memories relevant to the current topic."""
        if not topic or not self.memories:
            return []
        
        # Simple relevance scoring based on keyword matching
        scored_memories = []
        topic_words = set(topic.lower().split())
        
        for memory in self.memories:
            memory_words = set(memory.content.lower().split())
            relevance = len(topic_words.intersection(memory_words)) / len(topic_words)
            scored_memories.append((memory, relevance))
        
        # Sort by relevance and importance
        scored_memories.sort(
            key=lambda x: (x[1], x[0].importance),
            reverse=True
        )
        
        return [memory for memory, _ in scored_memories[:limit]]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of all context information."""
        return {
            "session": self.get_session_context(),
            "user_preferences": self.get_user_preferences(),
            "system": self.get_system_context(),
            "memories_count": len(self.memories),
            "conversation_turns": len(self.conversation_history),
            "total_context_items": len(self.items)
        }
    
    def clear_context(self, context_type: Optional[ContextType] = None):
        """Clear context items of a specific type or all if none specified."""
        if context_type:
            keys_to_remove = [
                key for key, item in self.items.items()
                if item.type == context_type
            ]
            for key in keys_to_remove:
                del self.items[key]
        else:
            self.items.clear()
            self.memories.clear()
            self.conversation_history.clear()
        
        self.logger.info(f"Cleared context: {context_type.value if context_type else 'all'}")
    
    def merge_context(self, other_context: 'Context'):
        """Merge another context into this one."""
        # Merge context items
        for key, item in other_context.items.items():
            if key not in self.items or item.priority > self.items[key].priority:
                self.items[key] = item
        
        # Merge memories
        self.memories.extend(other_context.memories)
        
        # Merge conversation history
        self.conversation_history.extend(other_context.conversation_history)
        
        # Keep conversation history within limits
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        self.logger.info("Merged context from another context object")


class ContextManager:
    """
    Manager for multiple context objects.
    
    This class handles:
    - Context creation and management
    - Context persistence
    - Context sharing between sessions
    """
    
    def __init__(self):
        self.contexts: Dict[str, Context] = {}
        self.logger = logging.getLogger(f"{__name__}.ContextManager")
    
    def create_context(self, context_id: str) -> Context:
        """Create a new context with the given ID."""
        context = Context()
        self.contexts[context_id] = context
        self.logger.info(f"Created new context: {context_id}")
        return context
    
    def get_context(self, context_id: str) -> Optional[Context]:
        """Get a context by ID."""
        return self.contexts.get(context_id)
    
    def update_context(self, context_id: str, context: Context):
        """Update or create a context."""
        self.contexts[context_id] = context
        self.logger.debug(f"Updated context: {context_id}")
    
    def delete_context(self, context_id: str):
        """Delete a context."""
        if context_id in self.contexts:
            del self.contexts[context_id]
            self.logger.info(f"Deleted context: {context_id}")
    
    def list_contexts(self) -> List[str]:
        """List all context IDs."""
        return list(self.contexts.keys())
    
    def get_context_summary(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a specific context."""
        context = self.get_context(context_id)
        if context:
            return context.get_context_summary()
        return None
    
    def cleanup_old_contexts(self, max_age_hours: int = 24):
        """Clean up contexts older than the specified age."""
        current_time = datetime.utcnow()
        contexts_to_remove = []
        
        for context_id, context in self.contexts.items():
            # Check if context has session information
            session_context = context.get_session_context()
            if "last_activity" in session_context:
                last_activity = session_context["last_activity"]
                if isinstance(last_activity, str):
                    # Parse string timestamp
                    try:
                        last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                
                age_hours = (current_time - last_activity).total_seconds() / 3600
                if age_hours > max_age_hours:
                    contexts_to_remove.append(context_id)
        
        for context_id in contexts_to_remove:
            self.delete_context(context_id)
        
        if contexts_to_remove:
            self.logger.info(f"Cleaned up {len(contexts_to_remove)} old contexts")
    
    def get_all_contexts_summary(self) -> Dict[str, Any]:
        """Get a summary of all contexts."""
        return {
            "total_contexts": len(self.contexts),
            "context_ids": list(self.contexts.keys()),
            "contexts": {
                context_id: context.get_context_summary()
                for context_id, context in self.contexts.items()
            }
        }


# Global context manager instance
context_manager = ContextManager() 