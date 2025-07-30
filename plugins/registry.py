"""
Plugin Registry

This module provides a registry for managing and discovering plugins.
"""

import logging
from typing import Dict, List, Optional, Type, Any
from .base import BasePlugin, ModelPlugin, ToolPlugin, IntegrationPlugin, ProcessingPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Registry for managing plugins and their capabilities.
    
    This registry provides:
    - Plugin registration and discovery
    - Capability-based plugin lookup
    - Plugin type categorization
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.model_plugins: Dict[str, ModelPlugin] = {}
        self.tool_plugins: Dict[str, ToolPlugin] = {}
        self.integration_plugins: Dict[str, IntegrationPlugin] = {}
        self.processing_plugins: Dict[str, ProcessingPlugin] = {}
        
        logger.info("Plugin registry initialized")
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin in the registry."""
        try:
            plugin_name = plugin.info.name
            
            if plugin_name in self.plugins:
                logger.warning(f"Plugin {plugin_name} already registered")
                return False
            
            # Register in main plugins dict
            self.plugins[plugin_name] = plugin
            
            # Register by type
            if isinstance(plugin, ModelPlugin):
                self.model_plugins[plugin_name] = plugin
            elif isinstance(plugin, ToolPlugin):
                self.tool_plugins[plugin_name] = plugin
            elif isinstance(plugin, IntegrationPlugin):
                self.integration_plugins[plugin_name] = plugin
            elif isinstance(plugin, ProcessingPlugin):
                self.processing_plugins[plugin_name] = plugin
            
            logger.info(f"Registered plugin: {plugin_name} ({type(plugin).__name__})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin.info.name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin from the registry."""
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} not registered")
                return False
            
            plugin = self.plugins[plugin_name]
            
            # Remove from type-specific registries
            if isinstance(plugin, ModelPlugin):
                self.model_plugins.pop(plugin_name, None)
            elif isinstance(plugin, ToolPlugin):
                self.tool_plugins.pop(plugin_name, None)
            elif isinstance(plugin, IntegrationPlugin):
                self.integration_plugins.pop(plugin_name, None)
            elif isinstance(plugin, ProcessingPlugin):
                self.processing_plugins.pop(plugin_name, None)
            
            # Remove from main registry
            del self.plugins[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins."""
        return self.plugins.copy()
    
    def get_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> Dict[str, BasePlugin]:
        """Get plugins of a specific type."""
        if plugin_type == ModelPlugin:
            return self.model_plugins.copy()
        elif plugin_type == ToolPlugin:
            return self.tool_plugins.copy()
        elif plugin_type == IntegrationPlugin:
            return self.integration_plugins.copy()
        elif plugin_type == ProcessingPlugin:
            return self.processing_plugins.copy()
        else:
            # Return plugins that are instances of the specified type
            return {
                name: plugin for name, plugin in self.plugins.items()
                if isinstance(plugin, plugin_type)
            }
    
    def get_enabled_plugins(self) -> Dict[str, BasePlugin]:
        """Get all enabled plugins."""
        return {
            name: plugin for name, plugin in self.plugins.items()
            if plugin.is_enabled()
        }
    
    def get_enabled_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> Dict[str, BasePlugin]:
        """Get enabled plugins of a specific type."""
        plugins = self.get_plugins_by_type(plugin_type)
        return {
            name: plugin for name, plugin in plugins.items()
            if plugin.is_enabled()
        }
    
    async def get_available_models(self) -> List[Dict]:
        """Get all available models from model plugins."""
        models = []
        
        for plugin_name, plugin in self.model_plugins.items():
            if plugin.is_enabled():
                try:
                    plugin_models = await plugin.get_models()
                    for model in plugin_models:
                        model["plugin"] = plugin_name
                        models.append(model)
                except Exception as e:
                    logger.error(f"Failed to get models from plugin {plugin_name}: {e}")
        
        return models
    
    async def generate_with_model(
        self, 
        model_name: str, 
        prompt: str, 
        **kwargs
    ) -> Optional[Dict]:
        """Generate a response using a specific model."""
        for plugin_name, plugin in self.model_plugins.items():
            if not plugin.is_enabled():
                continue
            
            try:
                models = await plugin.get_models()
                model_names = [model.get("name") for model in models]
                
                if model_name in model_names:
                    return await plugin.generate(model_name, prompt, **kwargs)
                    
            except Exception as e:
                logger.error(f"Failed to generate with model {model_name} from plugin {plugin_name}: {e}")
        
        logger.warning(f"Model {model_name} not found in any enabled model plugins")
        return None
    
    def get_available_tools(self) -> List[Dict]:
        """Get all available tools from tool plugins."""
        tools = []
        
        for plugin_name, plugin in self.tool_plugins.items():
            if plugin.is_enabled():
                try:
                    plugin_tools = plugin.get_tools()
                    for tool in plugin_tools:
                        tool["plugin"] = plugin_name
                        tools.append(tool)
                except Exception as e:
                    logger.error(f"Failed to get tools from plugin {plugin_name}: {e}")
        
        return tools
    
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict
    ) -> Optional[Dict]:
        """Execute a tool with the given parameters."""
        for plugin_name, plugin in self.tool_plugins.items():
            if not plugin.is_enabled():
                continue
            
            try:
                tools = plugin.get_tools()
                tool_names = [tool.get("name") for tool in tools]
                
                if tool_name in tool_names:
                    return await plugin.execute_tool(tool_name, parameters)
                    
            except Exception as e:
                logger.error(f"Failed to execute tool {tool_name} from plugin {plugin_name}: {e}")
        
        logger.warning(f"Tool {tool_name} not found in any enabled tool plugins")
        return None
    
    def get_available_integrations(self) -> List[Dict]:
        """Get all available integrations from integration plugins."""
        integrations = []
        
        for plugin_name, plugin in self.integration_plugins.items():
            if plugin.is_enabled():
                try:
                    plugin_integrations = plugin.get_integrations()
                    for integration in plugin_integrations:
                        integration["plugin"] = plugin_name
                        integrations.append(integration)
                except Exception as e:
                    logger.error(f"Failed to get integrations from plugin {plugin_name}: {e}")
        
        return integrations
    
    async def test_integration_connection(self, integration_name: str) -> bool:
        """Test connection to an integration."""
        for plugin_name, plugin in self.integration_plugins.items():
            if not plugin.is_enabled():
                continue
            
            try:
                integrations = plugin.get_integrations()
                integration_names = [integration.get("name") for integration in integrations]
                
                if integration_name in integration_names:
                    return await plugin.test_connection(integration_name)
                    
            except Exception as e:
                logger.error(f"Failed to test integration {integration_name} from plugin {plugin_name}: {e}")
        
        logger.warning(f"Integration {integration_name} not found in any enabled integration plugins")
        return False
    
    async def process_input_with_plugins(self, input_data: Any) -> Any:
        """Process input data through all enabled processing plugins."""
        processed_data = input_data
        
        for plugin_name, plugin in self.processing_plugins.items():
            if not plugin.is_enabled():
                continue
            
            try:
                processed_data = await plugin.process_input(processed_data)
            except Exception as e:
                logger.error(f"Failed to process input with plugin {plugin_name}: {e}")
        
        return processed_data
    
    async def process_output_with_plugins(self, output_data: Any) -> Any:
        """Process output data through all enabled processing plugins."""
        processed_data = output_data
        
        for plugin_name, plugin in self.processing_plugins.items():
            if not plugin.is_enabled():
                continue
            
            try:
                processed_data = await plugin.process_output(processed_data)
            except Exception as e:
                logger.error(f"Failed to process output with plugin {plugin_name}: {e}")
        
        return processed_data
    
    def get_plugin_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all registered plugins."""
        capabilities = {}
        
        for plugin_name, plugin in self.plugins.items():
            plugin_capabilities = []
            
            if isinstance(plugin, ModelPlugin):
                plugin_capabilities.append("model_provider")
            if isinstance(plugin, ToolPlugin):
                plugin_capabilities.append("tool_provider")
            if isinstance(plugin, IntegrationPlugin):
                plugin_capabilities.append("integration_provider")
            if isinstance(plugin, ProcessingPlugin):
                plugin_capabilities.append("processing_provider")
            
            capabilities[plugin_name] = plugin_capabilities
        
        return capabilities
    
    def get_registry_status(self) -> Dict:
        """Get status information about the registry."""
        return {
            "total_plugins": len(self.plugins),
            "enabled_plugins": len(self.get_enabled_plugins()),
            "model_plugins": len(self.model_plugins),
            "tool_plugins": len(self.tool_plugins),
            "integration_plugins": len(self.integration_plugins),
            "processing_plugins": len(self.processing_plugins),
            "plugin_types": {
                "model": list(self.model_plugins.keys()),
                "tool": list(self.tool_plugins.keys()),
                "integration": list(self.integration_plugins.keys()),
                "processing": list(self.processing_plugins.keys())
            }
        }


# Global plugin registry instance
plugin_registry = PluginRegistry() 