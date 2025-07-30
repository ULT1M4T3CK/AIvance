"""
AIvance Web UI Package

This package provides the web-based dashboard and user interface.
"""

from .app import create_web_app
from .routes import dashboard, chat, settings, analytics

__all__ = [
    "create_web_app",
    "dashboard",
    "chat", 
    "settings",
    "analytics"
] 