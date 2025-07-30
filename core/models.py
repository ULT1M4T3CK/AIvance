"""
AI Model Management

This module handles different AI model providers, types, and configurations
for the AIvance system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import openai
import anthropic
from transformers import AutoTokenizer, AutoModel
import torch


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models supported by the system."""
    
    LANGUAGE = "language"
    EMBEDDING = "embedding"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"
    REASONING = "reasoning"


class ModelProvider(Enum):
    """Supported AI model providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    
    name: str
    provider: ModelProvider
    model_type: ModelType
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("Top_p must be between 0 and 1")


@dataclass
class ModelResponse:
    """Response from an AI model."""
    
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BaseModel(ABC):
    """Base class for all AI models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""
        pass


class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using OpenAI API."""
        try:
            # Merge config with kwargs
            params = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                **kwargs
            }
            
            if self.config.system_prompt:
                params["messages"].insert(0, {
                    "role": "system", 
                    "content": self.config.system_prompt
                })
            
            if self.config.stop_sequences:
                params["stop"] = self.config.stop_sequences
            
            response = await self.client.chat.completions.create(**params)
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )
        
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return hasattr(self, 'client') and self.client is not None


class AnthropicModel(BaseModel):
    """Anthropic Claude model implementation."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Anthropic API."""
        try:
            params = {
                "model": self.config.name,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                **kwargs
            }
            
            # Handle system prompt
            if self.config.system_prompt:
                params["system"] = self.config.system_prompt
                params["messages"] = [{"role": "user", "content": prompt}]
            else:
                params["prompt"] = f"\n\nHuman: {prompt}\n\nAssistant:"
            
            response = await self.client.messages.create(**params)
            
            return ModelResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                finish_reason=response.stop_reason
            )
        
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Anthropic API."""
        try:
            response = await self.client.messages.embed(
                model="claude-3-sonnet-20240229",
                input=text
            )
            return response.embedding
        except Exception as e:
            self.logger.error(f"Anthropic embedding error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return hasattr(self, 'client') and self.client is not None


class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation."""
    
    def __init__(self, config: ModelConfig, token: Optional[str] = None):
        super().__init__(config)
        self.token = token
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model."""
        try:
            if self.config.model_type == ModelType.LANGUAGE:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.name, 
                    token=self.token
                )
                self.model = AutoModel.from_pretrained(
                    self.config.name, 
                    token=self.token
                )
            elif self.config.model_type == ModelType.EMBEDDING:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.config.name)
            
            self.logger.info(f"Loaded HuggingFace model: {self.config.name}")
        
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using HuggingFace model."""
        try:
            if not self.tokenizer or not self.model:
                raise RuntimeError("Model not loaded")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=self.config.max_tokens,
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return ModelResponse(
                content=response_text,
                model=self.config.name,
                usage={"tokens": len(outputs[0])}
            )
        
        except Exception as e:
            self.logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using HuggingFace model."""
        try:
            if not self.model:
                raise RuntimeError("Model not loaded")
            
            if self.config.model_type == ModelType.EMBEDDING:
                embeddings = self.model.encode(text)
                return embeddings.tolist()
            else:
                # Use the language model for embeddings
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use the last hidden state as embeddings
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    return embeddings.squeeze().tolist()
        
        except Exception as e:
            self.logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if HuggingFace model is available."""
        return self.model is not None


class AIModel:
    """Main AI model interface that manages different providers."""
    
    def __init__(self, config: ModelConfig, api_keys: Dict[str, str]):
        self.config = config
        self.api_keys = api_keys
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate model based on provider."""
        try:
            if self.config.provider == ModelProvider.OPENAI:
                api_key = self.api_keys.get("openai")
                if not api_key:
                    raise ValueError("OpenAI API key required")
                self._model = OpenAIModel(self.config, api_key)
            
            elif self.config.provider == ModelProvider.ANTHROPIC:
                api_key = self.api_keys.get("anthropic")
                if not api_key:
                    raise ValueError("Anthropic API key required")
                self._model = AnthropicModel(self.config, api_key)
            
            elif self.config.provider == ModelProvider.HUGGINGFACE:
                token = self.api_keys.get("huggingface")
                self._model = HuggingFaceModel(self.config, token)
            
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            self.logger = logging.getLogger(f"{__name__}.AIModel")
            self.logger.info(f"Initialized {self.config.provider.value} model: {self.config.name}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model."""
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        return await self._model.generate(prompt, **kwargs)
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self._model:
            raise RuntimeError("Model not initialized")
        
        return await self._model.embed(text)
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        return self._model is not None and self._model.is_available()
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.config.name,
            "provider": self.config.provider.value,
            "type": self.config.model_type.value,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "available": self.is_available()
        }


class ModelRegistry:
    """Registry for managing multiple AI models."""
    
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelRegistry")
    
    def register_model(self, name: str, model: AIModel):
        """Register a model in the registry."""
        self.models[name] = model
        self.logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[AIModel]:
        """Get a model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [name for name, model in self.models.items() if model.is_available()]
    
    async def generate_with_fallback(
        self, 
        prompt: str, 
        preferred_model: str = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response with fallback to available models."""
        # Try preferred model first
        if preferred_model and preferred_model in self.models:
            model = self.models[preferred_model]
            if model.is_available():
                try:
                    return await model.generate(prompt, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Preferred model failed: {e}")
        
        # Try other available models
        for name, model in self.models.items():
            if model.is_available():
                try:
                    return await model.generate(prompt, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Model {name} failed: {e}")
                    continue
        
        raise RuntimeError("No available models found")


# Global model registry
model_registry = ModelRegistry() 