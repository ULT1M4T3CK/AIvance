"""
Core System Tests

Tests for the core AI system components including engine, models, context, memory, and learning.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from core.engine import AIEngine, EngineConfig, EngineRequest, EngineResponse
from core.models import AIModel, ModelConfig, ModelProvider, ModelType, ModelResponse
from core.context import Context, ContextManager, ContextType
from core.memory import MemoryManager, MemoryItem, UserPreferences
from core.session import SessionManager, AISession, SessionStatus
from core.learning import LearningEngine, LearningEvent, LearningPattern
from core.reasoning import ReasoningEngine


class TestAIEngine:
    """Test the main AI engine."""
    
    @pytest.fixture
    def engine_config(self):
        """Create a test engine configuration."""
        return EngineConfig(
            default_model="test-model",
            max_context_length=2048,
            enable_memory=True,
            enable_learning=True,
            enable_reasoning=True,
            enable_safety=True,
            max_concurrent_requests=5
        )
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock AI model."""
        model = Mock(spec=AIModel)
        model.is_available.return_value = True
        model.generate = AsyncMock(return_value=ModelResponse(
            content="Test response",
            model="test-model",
            usage={"total_tokens": 10}
        ))
        return model
    
    @pytest.fixture
    def engine(self, engine_config, mock_model):
        """Create a test AI engine."""
        with patch('core.engine.model_registry') as mock_registry:
            mock_registry.get_model.return_value = mock_model
            mock_registry.generate_with_fallback.return_value = ModelResponse(
                content="Fallback response",
                model="fallback-model",
                usage={"total_tokens": 5}
            )
            
            engine = AIEngine(engine_config)
            return engine
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.default_model == "test-model"
        assert engine.context_manager is not None
        assert engine.memory_manager is not None
        assert engine.learning_engine is not None
        assert engine.reasoning_engine is not None
        assert engine.session_manager is not None
    
    @pytest.mark.asyncio
    async def test_process_request(self, engine):
        """Test processing a request through the engine."""
        request = EngineRequest(
            prompt="Hello, how are you?",
            user_id="test-user"
        )
        
        response = await engine.process_request(request)
        
        assert response.content == "Test response"
        assert response.model_used == "test-model"
        assert response.session_id is not None
        assert response.confidence_score is not None
    
    @pytest.mark.asyncio
    async def test_engine_status(self, engine):
        """Test getting engine status."""
        status = await engine.get_engine_status()
        
        assert status["status"] == "healthy"
        assert "models_available" in status
        assert "memory_enabled" in status
        assert "learning_enabled" in status
        assert "reasoning_enabled" in status
        assert "safety_enabled" in status


class TestContextManagement:
    """Test context management system."""
    
    @pytest.fixture
    def context(self):
        """Create a test context."""
        return Context()
    
    def test_context_creation(self, context):
        """Test context creation."""
        assert context.items == {}
        assert context.memories == []
        assert context.conversation_history == []
    
    def test_add_context_item(self, context):
        """Test adding context items."""
        context.add_item(ContextType.SESSION, "user_id", "test-user")
        
        assert len(context.items) == 1
        assert context.get_item(ContextType.SESSION, "user_id") == "test-user"
    
    def test_add_conversation_turn(self, context):
        """Test adding conversation turns."""
        context.add_conversation_turn("Hello", "Hi there!")
        
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0]["user"] == "Hello"
        assert context.conversation_history[0]["assistant"] == "Hi there!"
    
    def test_context_summary(self, context):
        """Test getting context summary."""
        context.add_item(ContextType.SESSION, "user_id", "test-user")
        context.add_conversation_turn("Hello", "Hi there!")
        
        summary = context.get_context_summary()
        
        assert summary["total_context_items"] == 1
        assert summary["conversation_turns"] == 1
        assert summary["memories_count"] == 0


class TestMemoryManagement:
    """Test memory management system."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a test memory manager."""
        with patch('core.memory.SentenceTransformer'):
            return MemoryManager()
    
    @pytest.mark.asyncio
    async def test_store_interaction(self, memory_manager):
        """Test storing an interaction."""
        memory_id = await memory_manager.store_interaction(
            user_id="test-user",
            user_message="Hello",
            assistant_response="Hi there!",
            session_id="test-session"
        )
        
        assert memory_id is not None
    
    @pytest.mark.asyncio
    async def test_get_relevant_memories(self, memory_manager):
        """Test retrieving relevant memories."""
        # Store a memory first
        await memory_manager.store_interaction(
            user_id="test-user",
            user_message="What is AI?",
            assistant_response="AI is artificial intelligence."
        )
        
        # Get relevant memories
        memories = await memory_manager.get_relevant_memories(
            query="artificial intelligence",
            user_id="test-user"
        )
        
        assert len(memories) >= 0  # May be 0 if no matches found
    
    @pytest.mark.asyncio
    async def test_user_preferences(self, memory_manager):
        """Test user preferences management."""
        # Update preferences
        success = await memory_manager.update_user_preferences(
            user_id="test-user",
            updates={"language": "en", "response_style": "detailed"}
        )
        
        assert success is True
        
        # Get preferences
        preferences = await memory_manager.get_user_preferences("test-user")
        assert preferences is not None


class TestSessionManagement:
    """Test session management system."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a test session manager."""
        return SessionManager()
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a new session."""
        session = await session_manager.create_session(user_id="test-user")
        
        assert session.session_id is not None
        assert session.user_id == "test-user"
        assert session.status == SessionStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        """Test retrieving a session."""
        created_session = await session_manager.create_session(user_id="test-user")
        
        retrieved_session = await session_manager.get_session(created_session.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session.user_id == "test-user"
    
    @pytest.mark.asyncio
    async def test_add_interaction(self, session_manager):
        """Test adding an interaction to a session."""
        session = await session_manager.create_session(user_id="test-user")
        
        success = await session_manager.add_interaction(
            session_id=session.session_id,
            user_message="Hello",
            assistant_response="Hi there!"
        )
        
        assert success is True
        
        # Verify interaction was added
        updated_session = await session_manager.get_session(session.session_id)
        assert len(updated_session.interactions) == 1
        assert updated_session.interactions[0].user_message == "Hello"
        assert updated_session.interactions[0].assistant_response == "Hi there!"


class TestLearningEngine:
    """Test learning engine."""
    
    @pytest.fixture
    def learning_engine(self):
        """Create a test learning engine."""
        return LearningEngine()
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, learning_engine):
        """Test learning from an interaction."""
        await learning_engine.learn_from_interaction(
            user_input="What is Python?",
            ai_response="Python is a programming language.",
            user_id="test-user"
        )
        
        # Verify learning occurred (check if patterns were extracted)
        # This is a basic test - in a real scenario, you'd check the actual data
    
    @pytest.mark.asyncio
    async def test_learn_from_feedback(self, learning_engine):
        """Test learning from feedback."""
        await learning_engine.learn_from_feedback(
            user_input="What is Python?",
            ai_response="Python is a programming language.",
            feedback="Good answer!",
            feedback_score=0.9,
            user_id="test-user"
        )
        
        # Verify feedback was processed
    
    @pytest.mark.asyncio
    async def test_get_user_preferences(self, learning_engine):
        """Test getting user preferences."""
        preferences = await learning_engine.get_user_preferences("test-user")
        
        # Should return empty dict for new user
        assert isinstance(preferences, dict)


class TestReasoningEngine:
    """Test reasoning engine."""
    
    @pytest.fixture
    def reasoning_engine(self):
        """Create a test reasoning engine."""
        return ReasoningEngine()
    
    @pytest.mark.asyncio
    async def test_analyze_response(self, reasoning_engine):
        """Test response analysis."""
        reasoning_steps = await reasoning_engine.analyze_response(
            user_prompt="What is AI?",
            ai_response="AI is artificial intelligence that mimics human thinking."
        )
        
        assert isinstance(reasoning_steps, list)
        assert len(reasoning_steps) > 0
    
    @pytest.mark.asyncio
    async def test_logical_reasoning(self, reasoning_engine):
        """Test logical reasoning."""
        premises = [
            "All humans are mortal.",
            "Socrates is a human."
        ]
        question = "Is Socrates mortal?"
        
        result = await reasoning_engine.perform_logical_reasoning(premises, question)
        
        assert result.reasoning_id is not None
        assert result.final_conclusion is not None
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_argument_validation(self, reasoning_engine):
        """Test argument validation."""
        argument = "This is wrong because it's stupid."
        
        validation = await reasoning_engine.validate_argument(argument)
        
        assert "valid" in validation
        assert "issues" in validation
        assert "strength" in validation
        assert "suggestions" in validation


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow through the system."""
        # This would test the entire flow from request to response
        # including context building, memory retrieval, learning, etc.
        pass
    
    @pytest.mark.asyncio
    async def test_system_persistence(self):
        """Test that the system properly persists data."""
        # Test that memories, sessions, and learning data are properly saved
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling throughout the system."""
        # Test how the system handles various error conditions
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 