"""
Plugin Base Classes

This module defines the base classes and interfaces for the plugin system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import inspect

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a plugin."""
    
    name: str
    version: str
    description: str
    author: str
    homepage: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)


@dataclass
class PluginConfig:
    """Configuration for a plugin."""
    
    enabled: bool = True
    auto_load: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """
    Base class for all AIvance plugins.
    
    Plugins can extend the AI system with additional capabilities
    such as new models, tools, integrations, or processing steps.
    """
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.info = self.get_info()
        self.logger = logging.getLogger(f"plugin.{self.info.name}")
        self._is_loaded = False
        self._is_enabled = False
        
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Get plugin information."""
        pass
    
    async def load(self) -> bool:
        """Load the plugin and initialize resources."""
        try:
            if self._is_loaded:
                self.logger.warning("Plugin already loaded")
                return True
            
            self.logger.info(f"Loading plugin: {self.info.name}")
            
            # Initialize plugin
            await self.on_load()
            
            self._is_loaded = True
            self.logger.info(f"Plugin loaded successfully: {self.info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {self.info.name}: {e}")
            return False
    
    async def unload(self) -> bool:
        """Unload the plugin and cleanup resources."""
        try:
            if not self._is_loaded:
                self.logger.warning("Plugin not loaded")
                return True
            
            self.logger.info(f"Unloading plugin: {self.info.name}")
            
            # Cleanup plugin
            await self.on_unload()
            
            self._is_loaded = False
            self._is_enabled = False
            self.logger.info(f"Plugin unloaded successfully: {self.info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {self.info.name}: {e}")
            return False
    
    async def enable(self) -> bool:
        """Enable the plugin."""
        try:
            if not self._is_loaded:
                self.logger.error("Cannot enable unloaded plugin")
                return False
            
            if self._is_enabled:
                self.logger.warning("Plugin already enabled")
                return True
            
            self.logger.info(f"Enabling plugin: {self.info.name}")
            
            # Enable plugin
            await self.on_enable()
            
            self._is_enabled = True
            self.logger.info(f"Plugin enabled successfully: {self.info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable plugin {self.info.name}: {e}")
            return False
    
    async def disable(self) -> bool:
        """Disable the plugin."""
        try:
            if not self._is_enabled:
                self.logger.warning("Plugin already disabled")
                return True
            
            self.logger.info(f"Disabling plugin: {self.info.name}")
            
            # Disable plugin
            await self.on_disable()
            
            self._is_enabled = False
            self.logger.info(f"Plugin disabled successfully: {self.info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to disable plugin {self.info.name}: {e}")
            return False
    
    async def on_load(self):
        """Called when the plugin is loaded."""
        pass
    
    async def on_unload(self):
        """Called when the plugin is unloaded."""
        pass
    
    async def on_enable(self):
        """Called when the plugin is enabled."""
        pass
    
    async def on_disable(self):
        """Called when the plugin is disabled."""
        pass
    
    def is_loaded(self) -> bool:
        """Check if the plugin is loaded."""
        return self._is_loaded
    
    def is_enabled(self) -> bool:
        """Check if the plugin is enabled."""
        return self._is_enabled
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "name": self.info.name,
            "version": self.info.version,
            "loaded": self._is_loaded,
            "enabled": self._is_enabled,
            "config": self.config.config,
            "metadata": self.config.metadata
        }


class ModelPlugin(BasePlugin):
    """Base class for model plugins that provide new AI models."""
    
    @abstractmethod
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of models provided by this plugin."""
        pass
    
    @abstractmethod
    async def generate(
        self, 
        model_name: str, 
        prompt: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using the specified model."""
        pass


class ToolPlugin(BasePlugin):
    """Base class for tool plugins that provide new capabilities."""
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of tools provided by this plugin."""
        pass
    
    @abstractmethod
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        pass


class IntegrationPlugin(BasePlugin):
    """Base class for integration plugins that connect to external services."""
    
    @abstractmethod
    def get_integrations(self) -> List[Dict[str, Any]]:
        """Get list of integrations provided by this plugin."""
        pass
    
    @abstractmethod
    async def test_connection(self, integration_name: str) -> bool:
        """Test connection to an integration."""
        pass


class ProcessingPlugin(BasePlugin):
    """Base class for processing plugins that modify AI processing."""
    
    @abstractmethod
    async def process_input(self, input_data: Any) -> Any:
        """Process input data before AI processing."""
        pass
    
    @abstractmethod
    async def process_output(self, output_data: Any) -> Any:
        """Process output data after AI processing."""
        pass


def plugin_info(
    name: str,
    version: str,
    description: str,
    author: str,
    homepage: Optional[str] = None,
    license: Optional[str] = None,
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None
):
    """Decorator to set plugin information."""
    def decorator(cls):
        if not issubclass(cls, BasePlugin):
            raise ValueError("Plugin info decorator can only be used on BasePlugin subclasses")
        
        # Store plugin info in the class
        cls._plugin_info = PluginInfo(
            name=name,
            version=version,
            description=description,
            author=author,
            homepage=homepage,
            license=license,
            tags=tags or [],
            dependencies=dependencies or [],
            requirements=requirements or []
        )
        
        return cls
    
    return decorator


def load_plugin_from_file(file_path: Path) -> Optional[BasePlugin]:
    """Load a plugin from a Python file."""
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("plugin_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        plugin_classes = []
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BasePlugin) and 
                obj != BasePlugin):
                plugin_classes.append(obj)
        
        if not plugin_classes:
            logger.warning(f"No plugin classes found in {file_path}")
            return None
        
        if len(plugin_classes) > 1:
            logger.warning(f"Multiple plugin classes found in {file_path}, using first one")
        
        plugin_class = plugin_classes[0]
        
        # Create plugin instance
        config = PluginConfig()
        plugin = plugin_class(config)
        
        return plugin
        
    except Exception as e:
        logger.error(f"Failed to load plugin from {file_path}: {e}")
        return None 