"""
AIvance Core Engine

The main AI engine that orchestrates all AI operations, manages context,
memory, reasoning, and learning capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

from .models import AIModel, ModelConfig, ModelProvider, ModelType, ModelResponse, model_registry
from .context import Context, ContextManager
from .memory import Memory, MemoryManager
from .reasoning import ReasoningEngine
from .learning import LearningEngine
from .session import AISession, SessionManager
from config import settings


logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the AI engine."""
    
    default_model: str = "gpt-4"
    max_context_length: int = 8192
    enable_memory: bool = True
    enable_learning: bool = True
    enable_reasoning: bool = True
    enable_safety: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineRequest:
    """Request structure for the AI engine."""
    
    prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Context] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResponse:
    """Response structure from the AI engine."""
    
    content: str
    model_used: str
    session_id: Optional[str] = None
    context: Optional[Context] = None
    reasoning_steps: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    safety_checks: Optional[Dict[str, Any]] = None
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AIEngine:
    """
    Main AI engine that orchestrates all AI operations.
    
    This engine manages:
    - Model selection and fallback
    - Context management
    - Memory and learning
    - Reasoning and safety
    - Session management
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AIEngine")
        
        # Initialize components
        self.context_manager = ContextManager()
        self.memory_manager = MemoryManager() if config.enable_memory else None
        self.learning_engine = LearningEngine() if config.enable_learning else None
        self.reasoning_engine = ReasoningEngine() if config.enable_reasoning else None
        self.session_manager = SessionManager()
        
        # Request semaphore for concurrency control
        self.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info("AI Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize AI models based on configuration."""
        try:
            api_keys = {
                "openai": settings.ai.openai_api_key,
                "anthropic": settings.ai.anthropic_api_key,
                "huggingface": settings.ai.huggingface_token
            }
            
            # Register OpenAI models
            if api_keys["openai"]:
                openai_config = ModelConfig(
                    name="gpt-4",
                    provider=ModelProvider.OPENAI,
                    model_type=ModelType.LANGUAGE,
                    max_tokens=settings.ai.max_tokens,
                    temperature=settings.ai.temperature
                )
                openai_model = AIModel(openai_config, api_keys)
                model_registry.register_model("gpt-4", openai_model)
                
                # Register GPT-3.5 as fallback
                gpt35_config = ModelConfig(
                    name="gpt-3.5-turbo",
                    provider=ModelProvider.OPENAI,
                    model_type=ModelType.LANGUAGE,
                    max_tokens=settings.ai.max_tokens,
                    temperature=settings.ai.temperature
                )
                gpt35_model = AIModel(gpt35_config, api_keys)
                model_registry.register_model("gpt-3.5-turbo", gpt35_model)
            
            # Register Anthropic models
            if api_keys["anthropic"]:
                claude_config = ModelConfig(
                    name="claude-3-sonnet-20240229",
                    provider=ModelProvider.ANTHROPIC,
                    model_type=ModelType.LANGUAGE,
                    max_tokens=settings.ai.max_tokens,
                    temperature=settings.ai.temperature
                )
                claude_model = AIModel(claude_config, api_keys)
                model_registry.register_model("claude-3-sonnet", claude_model)
            
            # Register embedding models
            embedding_config = ModelConfig(
                name="text-embedding-ada-002",
                provider=ModelProvider.OPENAI,
                model_type=ModelType.EMBEDDING
            )
            embedding_model = AIModel(embedding_config, api_keys)
            model_registry.register_model("text-embedding-ada-002", embedding_model)
            
            self.logger.info(f"Initialized {len(model_registry.models)} models")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def process_request(self, request: EngineRequest) -> EngineResponse:
        """
        Process an AI request through the complete pipeline.
        
        This method handles:
        1. Session management
        2. Context retrieval and management
        3. Memory integration
        4. Safety checks
        5. Model selection and generation
        6. Reasoning and learning
        7. Response formatting
        """
        async with self.request_semaphore:
            try:
                self.logger.info(f"Processing request for user: {request.user_id}")
                
                # 1. Session management
                session = await self._get_or_create_session(request)
                
                # 2. Context management
                context = await self._build_context(request, session)
                
                # 3. Memory integration
                if self.memory_manager:
                    context = await self._integrate_memory(context, request.user_id)
                
                # 4. Safety checks
                safety_result = await self._perform_safety_checks(request.prompt)
                if not safety_result["safe"]:
                    return EngineResponse(
                        content=safety_result["message"],
                        model_used="safety-filter",
                        session_id=session.session_id,
                        safety_checks=safety_result
                    )
                
                # 5. Model selection and generation
                model_response = await self._generate_response(request, context)
                
                # 6. Reasoning and analysis
                reasoning_steps = None
                if self.reasoning_engine:
                    reasoning_steps = await self.reasoning_engine.analyze_response(
                        request.prompt, model_response.content
                    )
                
                # 7. Learning integration
                if self.learning_engine:
                    await self.learning_engine.learn_from_interaction(
                        request.prompt, model_response.content, request.user_id
                    )
                
                # 8. Update session and memory
                await self._update_session(session, request, model_response)
                if self.memory_manager:
                    await self.memory_manager.store_interaction(
                        request.user_id, request.prompt, model_response.content
                    )
                
                # 9. Build response
                response = EngineResponse(
                    content=model_response.content,
                    model_used=model_response.model,
                    session_id=session.session_id,
                    context=context,
                    reasoning_steps=reasoning_steps,
                    confidence_score=self._calculate_confidence(model_response),
                    safety_checks=safety_result,
                    usage=model_response.usage,
                    metadata=model_response.metadata
                )
                
                self.logger.info(f"Request processed successfully for user: {request.user_id}")
                return response
            
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                raise
    
    async def _get_or_create_session(self, request: EngineRequest) -> AISession:
        """Get existing session or create a new one."""
        if request.session_id:
            session = await self.session_manager.get_session(request.session_id)
            if session:
                return session
        
        return await self.session_manager.create_session(request.user_id)
    
    async def _build_context(self, request: EngineRequest, session: AISession) -> Context:
        """Build context for the request."""
        context = request.context or Context()
        
        # Add session context
        context.add_session_context(session)
        
        # Add user preferences if available
        if request.user_id and self.memory_manager:
            user_preferences = await self.memory_manager.get_user_preferences(request.user_id)
            if user_preferences:
                context.add_user_preferences(user_preferences)
        
        # Add system context
        context.add_system_context({
            "timestamp": datetime.utcnow(),
            "engine_config": self.config.metadata,
            "available_models": model_registry.get_available_models()
        })
        
        return context
    
    async def _integrate_memory(self, context: Context, user_id: str) -> Context:
        """Integrate relevant memory into context."""
        if not user_id:
            return context
        
        # Get relevant memories
        memories = await self.memory_manager.get_relevant_memories(
            context.get_current_topic(), user_id, limit=5
        )
        
        if memories:
            context.add_memories(memories)
        
        return context
    
    async def _perform_safety_checks(self, prompt: str) -> Dict[str, Any]:
        """Perform safety checks on the prompt."""
        if not self.config.enable_safety:
            return {"safe": True, "message": "Safety checks disabled"}
        
        # Basic content filtering
        blocked_words = ["harmful", "dangerous", "illegal"]
        for word in blocked_words:
            if word.lower() in prompt.lower():
                return {
                    "safe": False,
                    "message": f"Content contains blocked term: {word}",
                    "blocked_term": word
                }
        
        # TODO: Implement more sophisticated safety checks
        # - Toxicity detection
        # - Bias detection
        # - Harmful content classification
        
        return {"safe": True, "message": "Content passed safety checks"}
    
    async def _generate_response(
        self, 
        request: EngineRequest, 
        context: Context
    ) -> ModelResponse:
        """Generate response using the appropriate model."""
        # Select model
        model_name = request.model or self.config.default_model
        model = model_registry.get_model(model_name)
        
        if not model or not model.is_available():
            # Fallback to available models
            self.logger.warning(f"Model {model_name} not available, using fallback")
            return await model_registry.generate_with_fallback(
                request.prompt,
                preferred_model=model_name,
                temperature=request.temperature or self.config.temperature,
                max_tokens=request.max_tokens or self.config.max_context_length
            )
        
        # Build full prompt with context
        full_prompt = self._build_full_prompt(request.prompt, context)
        
        # Generate response
        return await model.generate(
            full_prompt,
            temperature=request.temperature or self.config.temperature,
            max_tokens=request.max_tokens or self.config.max_context_length,
            system_prompt=request.system_prompt
        )
    
    def _build_full_prompt(self, prompt: str, context: Context) -> str:
        """Build a full prompt incorporating context."""
        context_parts = []
        
        # Add system context
        system_context = context.get_system_context()
        if system_context:
            context_parts.append(f"System Context: {system_context}")
        
        # Add user preferences
        user_preferences = context.get_user_preferences()
        if user_preferences:
            context_parts.append(f"User Preferences: {user_preferences}")
        
        # Add relevant memories
        memories = context.get_memories()
        if memories:
            memory_text = "\n".join([f"- {memory.content}" for memory in memories])
            context_parts.append(f"Relevant Context:\n{memory_text}")
        
        # Add session history
        session_history = context.get_session_history()
        if session_history:
            history_text = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in session_history[-3:]  # Last 3 exchanges
            ])
            context_parts.append(f"Recent Conversation:\n{history_text}")
        
        # Build final prompt
        if context_parts:
            context_text = "\n\n".join(context_parts)
            return f"{context_text}\n\nCurrent Request: {prompt}"
        else:
            return prompt
    
    def _calculate_confidence(self, model_response: ModelResponse) -> float:
        """Calculate confidence score for the response."""
        # Simple confidence calculation based on response length and finish reason
        base_confidence = 0.8
        
        # Adjust based on finish reason
        if model_response.finish_reason == "stop":
            base_confidence += 0.1
        elif model_response.finish_reason == "length":
            base_confidence -= 0.2
        
        # Adjust based on response length
        if len(model_response.content) < 50:
            base_confidence -= 0.1
        elif len(model_response.content) > 500:
            base_confidence += 0.05
        
        return min(max(base_confidence, 0.0), 1.0)
    
    async def _update_session(
        self, 
        session: AISession, 
        request: EngineRequest, 
        response: ModelResponse
    ):
        """Update session with new interaction."""
        await self.session_manager.add_interaction(
            session.session_id,
            request.prompt,
            response.content
        )
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get information about available models."""
        models = []
        for name, model in model_registry.models.items():
            models.append({
                "name": name,
                "info": model.model_info,
                "available": model.is_available()
            })
        return models
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and health information."""
        return {
            "status": "healthy",
            "models_available": len(model_registry.get_available_models()),
            "total_models": len(model_registry.models),
            "memory_enabled": self.memory_manager is not None,
            "learning_enabled": self.learning_engine is not None,
            "reasoning_enabled": self.reasoning_engine is not None,
            "safety_enabled": self.config.enable_safety,
            "active_sessions": len(self.session_manager.active_sessions),
            "config": self.config.metadata
        }
    
    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        self.logger.info("Shutting down AI Engine")
        
        # Close sessions
        await self.session_manager.close_all_sessions()
        
        # Save memory and learning data
        if self.memory_manager:
            await self.memory_manager.save_all()
        
        if self.learning_engine:
            await self.learning_engine.save_learned_data()
        
        self.logger.info("AI Engine shutdown complete")


# Global engine instance
_engine: Optional[AIEngine] = None


def get_engine() -> AIEngine:
    """Get the global AI engine instance."""
    global _engine
    if _engine is None:
        config = EngineConfig(
            default_model=settings.ai.default_language_model,
            max_context_length=settings.ai.max_tokens,
            enable_memory=True,
            enable_learning=True,
            enable_reasoning=True,
            enable_safety=settings.safety.content_filtering_enabled,
            max_concurrent_requests=settings.ai.max_concurrent_requests
        )
        _engine = AIEngine(config)
    return _engine


async def shutdown_engine():
    """Shutdown the global engine instance."""
    global _engine
    if _engine:
        await _engine.shutdown()
        _engine = None 