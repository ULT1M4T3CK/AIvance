"""
AIvance Plugin System

This package handles the plugin system for extending AI capabilities.
"""

from .manager import PluginManager
from .base import BasePlugin, PluginConfig, PluginInfo
from .registry import plugin_registry

__all__ = [
    "PluginManager",
    "BasePlugin", 
    "PluginConfig",
    "PluginInfo",
    "plugin_registry"
] 