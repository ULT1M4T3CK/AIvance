"""
Memory Management System

This module handles long-term memory storage, retrieval, and management
for the AI system, including user preferences and interaction history.
"""

import asyncio
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A memory item with metadata and embeddings."""
    
    id: str
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    memory_type: str = "interaction"  # interaction, preference, fact, etc.
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()
            timestamp_str = self.timestamp.isoformat()
            self.id = f"{content_hash}_{timestamp_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


@dataclass
class UserPreferences:
    """User preferences and settings."""
    
    user_id: str
    language: str = "en"
    response_style: str = "balanced"  # concise, detailed, balanced
    preferred_topics: List[str] = field(default_factory=list)
    communication_style: str = "friendly"  # formal, casual, friendly
    expertise_level: str = "general"  # beginner, intermediate, expert
    timezone: str = "UTC"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create from dictionary."""
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


class MemoryStore:
    """Base class for memory storage backends."""
    
    async def store(self, memory: MemoryItem) -> bool:
        """Store a memory item."""
        raise NotImplementedError
    
    async def retrieve(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """Retrieve relevant memories based on query."""
        raise NotImplementedError
    
    async def get_user_memories(self, user_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get all memories for a specific user."""
        raise NotImplementedError
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        raise NotImplementedError
    
    async def update_memory(self, memory: MemoryItem) -> bool:
        """Update a memory item."""
        raise NotImplementedError


class FileMemoryStore(MemoryStore):
    """File-based memory storage implementation."""
    
    def __init__(self, storage_path: str = "./data/memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories_file = self.storage_path / "memories.json"
        self.preferences_file = self.storage_path / "preferences.json"
        self.memories: Dict[str, MemoryItem] = {}
        self.preferences: Dict[str, UserPreferences] = {}
        self.logger = logging.getLogger(f"{__name__}.FileMemoryStore")
        self._load_data()
    
    def _load_data(self):
        """Load memories and preferences from files."""
        try:
            if self.memories_file.exists():
                with open(self.memories_file, 'r') as f:
                    memories_data = json.load(f)
                    self.memories = {
                        mid: MemoryItem.from_dict(mdata)
                        for mid, mdata in memories_data.items()
                    }
            
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r') as f:
                    preferences_data = json.load(f)
                    self.preferences = {
                        uid: UserPreferences.from_dict(pdata)
                        for uid, pdata in preferences_data.items()
                    }
            
            self.logger.info(f"Loaded {len(self.memories)} memories and {len(self.preferences)} user preferences")
        
        except Exception as e:
            self.logger.error(f"Error loading memory data: {e}")
    
    async def _save_data(self):
        """Save memories and preferences to files."""
        try:
            # Save memories
            memories_data = {
                mid: memory.to_dict()
                for mid, memory in self.memories.items()
            }
            with open(self.memories_file, 'w') as f:
                json.dump(memories_data, f, indent=2)
            
            # Save preferences
            preferences_data = {
                uid: prefs.to_dict()
                for uid, prefs in self.preferences.items()
            }
            with open(self.preferences_file, 'w') as f:
                json.dump(preferences_data, f, indent=2)
            
            self.logger.debug("Memory data saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving memory data: {e}")
    
    async def store(self, memory: MemoryItem) -> bool:
        """Store a memory item."""
        try:
            self.memories[memory.id] = memory
            await self._save_data()
            self.logger.debug(f"Stored memory: {memory.id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return False
    
    async def retrieve(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[MemoryItem]:
        """Retrieve relevant memories based on query."""
        # Simple keyword-based retrieval for now
        # TODO: Implement semantic search with embeddings
        relevant_memories = []
        query_lower = query.lower()
        
        for memory in self.memories.values():
            if user_id and memory.user_id != user_id:
                continue
            
            # Check if query words appear in memory content
            content_lower = memory.content.lower()
            if any(word in content_lower for word in query_lower.split()):
                relevant_memories.append(memory)
        
        # Sort by relevance (simple scoring) and importance
        scored_memories = []
        for memory in relevant_memories:
            score = 0
            for word in query_lower.split():
                if word in memory.content.lower():
                    score += 1
            score = score / len(query_lower.split())  # Normalize
            scored_memories.append((memory, score))
        
        scored_memories.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        
        return [memory for memory, _ in scored_memories[:limit]]
    
    async def get_user_memories(self, user_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get all memories for a specific user."""
        user_memories = [
            memory for memory in self.memories.values()
            if memory.user_id == user_id
        ]
        
        # Sort by timestamp (newest first)
        user_memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        return user_memories[:limit]
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        try:
            if memory_id in self.memories:
                del self.memories[memory_id]
                await self._save_data()
                self.logger.debug(f"Deleted memory: {memory_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False
    
    async def update_memory(self, memory: MemoryItem) -> bool:
        """Update a memory item."""
        try:
            if memory.id in self.memories:
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
                self.memories[memory.id] = memory
                await self._save_data()
                self.logger.debug(f"Updated memory: {memory.id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
            return False
    
    async def store_user_preferences(self, preferences: UserPreferences) -> bool:
        """Store user preferences."""
        try:
            self.preferences[preferences.user_id] = preferences
            await self._save_data()
            self.logger.debug(f"Stored preferences for user: {preferences.user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error storing preferences: {e}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences."""
        return self.preferences.get(user_id)


class MemoryManager:
    """
    Main memory manager that handles all memory operations.
    
    This class manages:
    - Memory storage and retrieval
    - User preferences
    - Memory embeddings and similarity search
    - Memory cleanup and maintenance
    """
    
    def __init__(self, storage_path: str = "./data/memory"):
        self.storage = FileMemoryStore(storage_path)
        self.embedding_model = None
        self.logger = logging.getLogger(f"{__name__}.MemoryManager")
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model for semantic search."""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize embedding model: {e}")
            self.embedding_model = None
    
    async def store_interaction(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_response: str,
        session_id: Optional[str] = None
    ) -> str:
        """Store an interaction as a memory."""
        try:
            # Create memory from interaction
            content = f"User: {user_message}\nAssistant: {assistant_response}"
            memory = MemoryItem(
                content=content,
                user_id=user_id,
                session_id=session_id,
                memory_type="interaction",
                importance=0.6,  # Default importance for interactions
                tags=["interaction", "conversation"]
            )
            
            # Generate embedding if model is available
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(content).tolist()
                    memory.embedding = embedding
                except Exception as e:
                    self.logger.warning(f"Could not generate embedding: {e}")
            
            # Store memory
            success = await self.storage.store(memory)
            if success:
                self.logger.debug(f"Stored interaction memory: {memory.id}")
                return memory.id
            else:
                self.logger.error("Failed to store interaction memory")
                return None
        
        except Exception as e:
            self.logger.error(f"Error storing interaction: {e}")
            return None
    
    async def store_fact(
        self, 
        content: str, 
        user_id: Optional[str] = None,
        importance: float = 0.7,
        tags: List[str] = None
    ) -> str:
        """Store a factual memory."""
        try:
            memory = MemoryItem(
                content=content,
                user_id=user_id,
                memory_type="fact",
                importance=importance,
                tags=tags or ["fact"]
            )
            
            # Generate embedding
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(content).tolist()
                    memory.embedding = embedding
                except Exception as e:
                    self.logger.warning(f"Could not generate embedding: {e}")
            
            success = await self.storage.store(memory)
            if success:
                self.logger.debug(f"Stored fact memory: {memory.id}")
                return memory.id
            else:
                self.logger.error("Failed to store fact memory")
                return None
        
        except Exception as e:
            self.logger.error(f"Error storing fact: {e}")
            return None
    
    async def get_relevant_memories(
        self, 
        query: str, 
        user_id: Optional[str] = None, 
        limit: int = 5
    ) -> List[MemoryItem]:
        """Get memories relevant to the query."""
        try:
            memories = await self.storage.retrieve(query, user_id, limit * 2)
            
            # If we have embeddings, use semantic similarity
            if self.embedding_model and memories:
                try:
                    query_embedding = self.embedding_model.encode(query)
                    
                    # Calculate similarities
                    scored_memories = []
                    for memory in memories:
                        if memory.embedding:
                            similarity = np.dot(query_embedding, memory.embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
                            )
                            # Combine similarity with importance
                            score = (similarity * 0.7) + (memory.importance * 0.3)
                            scored_memories.append((memory, score))
                        else:
                            # Fallback to importance only
                            scored_memories.append((memory, memory.importance))
                    
                    # Sort by score
                    scored_memories.sort(key=lambda x: x[1], reverse=True)
                    memories = [memory for memory, _ in scored_memories[:limit]]
                
                except Exception as e:
                    self.logger.warning(f"Error in semantic search: {e}")
                    memories = memories[:limit]
            else:
                memories = memories[:limit]
            
            # Update access statistics
            for memory in memories:
                await self.storage.update_memory(memory)
            
            return memories
        
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences."""
        try:
            return await self.storage.get_user_preferences(user_id)
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {e}")
            return None
    
    async def update_user_preferences(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update user preferences."""
        try:
            # Get existing preferences or create new ones
            preferences = await self.storage.get_user_preferences(user_id)
            if not preferences:
                preferences = UserPreferences(user_id=user_id)
            
            # Update fields
            for key, value in updates.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)
            
            preferences.last_updated = datetime.utcnow()
            
            # Store updated preferences
            success = await self.storage.store_user_preferences(preferences)
            if success:
                self.logger.debug(f"Updated preferences for user: {user_id}")
                return True
            else:
                self.logger.error(f"Failed to update preferences for user: {user_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {e}")
            return False
    
    async def get_user_memories(self, user_id: str, limit: int = 100) -> List[MemoryItem]:
        """Get all memories for a user."""
        try:
            return await self.storage.get_user_memories(user_id, limit)
        except Exception as e:
            self.logger.error(f"Error getting user memories: {e}")
            return []
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            return await self.storage.delete_memory(memory_id)
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False
    
    async def cleanup_old_memories(self, days_old: int = 30, min_importance: float = 0.3):
        """Clean up old, low-importance memories."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            memories_to_delete = []
            
            for memory in self.storage.memories.values():
                if (memory.timestamp < cutoff_date and 
                    memory.importance < min_importance and
                    memory.memory_type == "interaction"):
                    memories_to_delete.append(memory.id)
            
            deleted_count = 0
            for memory_id in memories_to_delete:
                if await self.storage.delete_memory(memory_id):
                    deleted_count += 1
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old memories")
            
            return deleted_count
        
        except Exception as e:
            self.logger.error(f"Error cleaning up memories: {e}")
            return 0
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            total_memories = len(self.storage.memories)
            total_users = len(self.storage.preferences)
            
            # Count by type
            type_counts = {}
            for memory in self.storage.memories.values():
                type_counts[memory.memory_type] = type_counts.get(memory.memory_type, 0) + 1
            
            # Average importance
            if total_memories > 0:
                avg_importance = sum(m.importance for m in self.storage.memories.values()) / total_memories
            else:
                avg_importance = 0
            
            return {
                "total_memories": total_memories,
                "total_users": total_users,
                "memory_types": type_counts,
                "average_importance": avg_importance,
                "embedding_model_available": self.embedding_model is not None
            }
        
        except Exception as e:
            self.logger.error(f"Error getting memory statistics: {e}")
            return {}
    
    async def save_all(self):
        """Save all memory data."""
        try:
            await self.storage._save_data()
            self.logger.info("All memory data saved")
        except Exception as e:
            self.logger.error(f"Error saving memory data: {e}")


# Global memory manager instance
memory_manager = MemoryManager() 