"""
Models API Routes

Routes for AI model management and information.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from core.engine import get_engine
from core.models import model_registry

logger = logging.getLogger(__name__)
router = APIRouter()


class ModelInfo(BaseModel):
    """Model information response model."""
    
    name: str
    provider: str
    type: str
    available: bool
    max_tokens: int
    temperature: float
    metadata: Dict[str, Any] = {}


@router.get("/", response_model=List[ModelInfo])
async def list_models():
    """
    List all available AI models.
    
    Returns information about all registered models.
    """
    try:
        engine = get_engine()
        models = await engine.get_available_models()
        
        model_list = []
        for model_data in models:
            model_list.append(ModelInfo(
                name=model_data["name"],
                provider=model_data["info"]["provider"],
                type=model_data["info"]["type"],
                available=model_data["available"],
                max_tokens=model_data["info"]["max_tokens"],
                temperature=model_data["info"]["temperature"],
                metadata=model_data["info"]
            ))
        
        return model_list
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    
    Returns comprehensive information about the specified model.
    """
    try:
        model = model_registry.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = model.model_info
        
        return {
            "name": model_name,
            "info": model_info,
            "available": model.is_available(),
            "capabilities": {
                "text_generation": model_info["type"] == "language",
                "embeddings": model_info["type"] == "embedding",
                "multimodal": model_info["type"] == "multimodal"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/overview")
async def get_models_status():
    """
    Get overview of all models status.
    
    Returns a summary of model availability and health.
    """
    try:
        engine = get_engine()
        models = await engine.get_available_models()
        
        total_models = len(models)
        available_models = sum(1 for m in models if m["available"])
        
        # Group by provider
        providers = {}
        for model in models:
            provider = model["info"]["provider"]
            if provider not in providers:
                providers[provider] = {"total": 0, "available": 0}
            
            providers[provider]["total"] += 1
            if model["available"]:
                providers[provider]["available"] += 1
        
        # Group by type
        types = {}
        for model in models:
            model_type = model["info"]["type"]
            if model_type not in types:
                types[model_type] = {"total": 0, "available": 0}
            
            types[model_type]["total"] += 1
            if model["available"]:
                types[model_type]["available"] += 1
        
        return {
            "summary": {
                "total_models": total_models,
                "available_models": available_models,
                "health_percentage": (available_models / total_models * 100) if total_models > 0 else 0
            },
            "by_provider": providers,
            "by_type": types,
            "models": [
                {
                    "name": m["name"],
                    "provider": m["info"]["provider"],
                    "type": m["info"]["type"],
                    "available": m["available"]
                }
                for m in models
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/{model_name}")
async def test_model(model_name: str, prompt: str = "Hello, how are you?"):
    """
    Test a specific model with a simple prompt.
    
    Returns the model's response to verify it's working correctly.
    """
    try:
        model = model_registry.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not model.is_available():
            raise HTTPException(status_code=503, detail="Model is not available")
        
        # Test the model
        response = await model.generate(prompt, max_tokens=100)
        
        return {
            "model": model_name,
            "prompt": prompt,
            "response": response.content,
            "usage": response.usage,
            "model_used": response.model,
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing model {model_name}: {e}")
        return {
            "model": model_name,
            "prompt": prompt,
            "response": None,
            "error": str(e),
            "success": False
        }


@router.get("/capabilities")
async def get_model_capabilities():
    """
    Get information about model capabilities.
    
    Returns what types of tasks each model can perform.
    """
    try:
        engine = get_engine()
        models = await engine.get_available_models()
        
        capabilities = {
            "text_generation": [],
            "embeddings": [],
            "multimodal": [],
            "reasoning": [],
            "code_generation": []
        }
        
        for model in models:
            model_type = model["info"]["type"]
            model_name = model["name"]
            
            if model_type == "language":
                capabilities["text_generation"].append(model_name)
            elif model_type == "embedding":
                capabilities["embeddings"].append(model_name)
            elif model_type == "multimodal":
                capabilities["multimodal"].append(model_name)
            elif model_type == "reasoning":
                capabilities["reasoning"].append(model_name)
            elif model_type == "code":
                capabilities["code_generation"].append(model_name)
        
        return {
            "capabilities": capabilities,
            "total_models": len(models),
            "available_models": sum(1 for m in models if m["available"])
        }
    
    except Exception as e:
        logger.error(f"Error getting model capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 