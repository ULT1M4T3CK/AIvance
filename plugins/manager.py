"""
Plugin Manager

This module manages the loading, unloading, and lifecycle of plugins.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type
from pathlib import Path
import json
import yaml

from config import settings
from .base import BasePlugin, PluginConfig, PluginInfo, load_plugin_from_file
from .registry import plugin_registry

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages the lifecycle of plugins in the AIvance system.
    
    This class handles:
    - Loading plugins from directories
    - Managing plugin states (loaded, enabled, disabled)
    - Plugin configuration management
    - Plugin dependency resolution
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.plugins_dir = Path(settings.plugins.plugins_dir)
        self.config_file = self.plugins_dir / "plugins_config.yaml"
        
        # Ensure plugins directory exists
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Plugin manager initialized with plugins directory: {self.plugins_dir}")
    
    async def load_plugins(self) -> bool:
        """Load all available plugins."""
        try:
            logger.info("Loading plugins...")
            
            # Load plugin configurations
            await self._load_plugin_configs()
            
            # Discover and load plugins
            plugin_files = self._discover_plugin_files()
            
            for plugin_file in plugin_files:
                await self._load_plugin_from_file(plugin_file)
            
            # Auto-enable plugins if configured
            if settings.plugins.auto_load_plugins:
                await self._auto_enable_plugins()
            
            logger.info(f"Plugin loading completed. Loaded {len(self.plugins)} plugins")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load plugins: {e}")
            return False
    
    async def unload_plugins(self) -> bool:
        """Unload all plugins."""
        try:
            logger.info("Unloading plugins...")
            
            for plugin_name, plugin in self.plugins.items():
                try:
                    await plugin.unload()
                    logger.info(f"Unloaded plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            
            self.plugins.clear()
            logger.info("All plugins unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugins: {e}")
            return False
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin by name."""
        try:
            if plugin_name in self.plugins:
                logger.warning(f"Plugin {plugin_name} already loaded")
                return True
            
            plugin_file = self._find_plugin_file(plugin_name)
            if not plugin_file:
                logger.error(f"Plugin file not found for {plugin_name}")
                return False
            
            return await self._load_plugin_from_file(plugin_file)
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin by name."""
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} not loaded")
                return True
            
            plugin = self.plugins[plugin_name]
            success = await plugin.unload()
            
            if success:
                del self.plugins[plugin_name]
                logger.info(f"Unloaded plugin: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a specific plugin."""
        try:
            if plugin_name not in self.plugins:
                logger.error(f"Plugin {plugin_name} not loaded")
                return False
            
            plugin = self.plugins[plugin_name]
            success = await plugin.enable()
            
            if success:
                # Update configuration
                self.plugin_configs[plugin_name].enabled = True
                await self._save_plugin_configs()
                logger.info(f"Enabled plugin: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to enable plugin {plugin_name}: {e}")
            return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a specific plugin."""
        try:
            if plugin_name not in self.plugins:
                logger.error(f"Plugin {plugin_name} not loaded")
                return False
            
            plugin = self.plugins[plugin_name]
            success = await plugin.disable()
            
            if success:
                # Update configuration
                self.plugin_configs[plugin_name].enabled = False
                await self._save_plugin_configs()
                logger.info(f"Disabled plugin: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to disable plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins(self) -> Dict[str, BasePlugin]:
        """Get all loaded plugins."""
        return self.plugins.copy()
    
    def get_plugin_status(self) -> Dict[str, Dict]:
        """Get status of all plugins."""
        status = {}
        for name, plugin in self.plugins.items():
            status[name] = plugin.get_status()
        return status
    
    async def get_enabled_plugins(self) -> List[BasePlugin]:
        """Get all enabled plugins."""
        return [plugin for plugin in self.plugins.values() if plugin.is_enabled()]
    
    async def get_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> List[BasePlugin]:
        """Get plugins of a specific type."""
        return [
            plugin for plugin in self.plugins.values() 
            if isinstance(plugin, plugin_type) and plugin.is_enabled()
        ]
    
    def _discover_plugin_files(self) -> List[Path]:
        """Discover plugin files in the plugins directory."""
        plugin_files = []
        
        for file_path in self.plugins_dir.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip private files
            
            if file_path.name in ["__init__.py", "base.py", "manager.py", "registry.py"]:
                continue  # Skip internal files
            
            plugin_files.append(file_path)
        
        logger.info(f"Discovered {len(plugin_files)} plugin files")
        return plugin_files
    
    def _find_plugin_file(self, plugin_name: str) -> Optional[Path]:
        """Find a plugin file by name."""
        for file_path in self._discover_plugin_files():
            if file_path.stem == plugin_name:
                return file_path
        return None
    
    async def _load_plugin_from_file(self, plugin_file: Path) -> bool:
        """Load a plugin from a file."""
        try:
            plugin = load_plugin_from_file(plugin_file)
            if not plugin:
                return False
            
            plugin_name = plugin.info.name
            
            # Check if plugin already loaded
            if plugin_name in self.plugins:
                logger.warning(f"Plugin {plugin_name} already loaded, skipping")
                return True
            
            # Load plugin configuration
            config = self.plugin_configs.get(plugin_name, PluginConfig())
            plugin.config = config
            
            # Load the plugin
            success = await plugin.load()
            if success:
                self.plugins[plugin_name] = plugin
                self.plugin_configs[plugin_name] = config
                
                # Register plugin
                plugin_registry.register_plugin(plugin)
                
                logger.info(f"Loaded plugin: {plugin_name} from {plugin_file}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_file}: {e}")
            return False
    
    async def _load_plugin_configs(self):
        """Load plugin configurations from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                
                for plugin_name, config_dict in config_data.items():
                    self.plugin_configs[plugin_name] = PluginConfig(**config_dict)
                
                logger.info(f"Loaded configurations for {len(self.plugin_configs)} plugins")
            else:
                logger.info("No plugin configuration file found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load plugin configurations: {e}")
    
    async def _save_plugin_configs(self):
        """Save plugin configurations to file."""
        try:
            config_data = {}
            for plugin_name, config in self.plugin_configs.items():
                config_data[plugin_name] = {
                    "enabled": config.enabled,
                    "auto_load": config.auto_load,
                    "config": config.config,
                    "metadata": config.metadata
                }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info("Plugin configurations saved")
            
        except Exception as e:
            logger.error(f"Failed to save plugin configurations: {e}")
    
    async def _auto_enable_plugins(self):
        """Auto-enable plugins based on configuration."""
        for plugin_name, config in self.plugin_configs.items():
            if config.auto_load and config.enabled:
                if plugin_name in self.plugins:
                    await self.enable_plugin(plugin_name)
    
    async def update_plugin_config(
        self, 
        plugin_name: str, 
        config: PluginConfig
    ) -> bool:
        """Update plugin configuration."""
        try:
            if plugin_name not in self.plugins:
                logger.error(f"Plugin {plugin_name} not loaded")
                return False
            
            # Update configuration
            self.plugin_configs[plugin_name] = config
            self.plugins[plugin_name].config = config
            
            # Save configurations
            await self._save_plugin_configs()
            
            logger.info(f"Updated configuration for plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update plugin configuration for {plugin_name}: {e}")
            return False
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """Get detailed information about a plugin."""
        try:
            if plugin_name not in self.plugins:
                return None
            
            plugin = self.plugins[plugin_name]
            info = plugin.info
            
            return {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "author": info.author,
                "homepage": info.homepage,
                "license": info.license,
                "tags": info.tags,
                "dependencies": info.dependencies,
                "requirements": info.requirements,
                "status": plugin.get_status()
            }
            
        except Exception as e:
            logger.error(f"Failed to get plugin info for {plugin_name}: {e}")
            return None 