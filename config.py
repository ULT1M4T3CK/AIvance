"""
AIvance Configuration Management

This module handles all configuration settings for the AI system,
including environment variables, model settings, and system parameters.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(default="postgresql://user:pass@localhost/aivance", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DB_ECHO")
    
    class Config:
        env_prefix = "DB_"


class AISettings(BaseSettings):
    """AI model and processing configuration."""
    
    # Model providers
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    
    # Model configurations
    default_language_model: str = Field(default="gpt-4", env="DEFAULT_LANGUAGE_MODEL")
    default_embedding_model: str = Field(default="text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Processing settings
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # Vector database
    vector_db_path: str = Field(default="./data/vectors", env="VECTOR_DB_PATH")
    similarity_threshold: float = Field(default=0.8, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_prefix = "AI_"


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""
    
    secret_key: str = Field(env="SECRET_KEY")
    jwt_secret: str = Field(env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION")
    
    # Encryption
    encryption_key: str = Field(env="ENCRYPTION_KEY")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # CORS
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Tracing
    tracing_enabled: bool = Field(default=False, env="TRACING_ENABLED")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    class Config:
        env_prefix = "MONITORING_"


class PluginSettings(BaseSettings):
    """Plugin system configuration."""
    
    plugins_dir: str = Field(default="./plugins", env="PLUGINS_DIR")
    auto_load_plugins: bool = Field(default=True, env="AUTO_LOAD_PLUGINS")
    plugin_timeout: int = Field(default=30, env="PLUGIN_TIMEOUT")
    
    class Config:
        env_prefix = "PLUGIN_"


class SafetySettings(BaseSettings):
    """Safety and ethics configuration."""
    
    content_filtering_enabled: bool = Field(default=True, env="CONTENT_FILTERING_ENABLED")
    toxicity_threshold: float = Field(default=0.7, env="TOXICITY_THRESHOLD")
    bias_detection_enabled: bool = Field(default=True, env="BIAS_DETECTION_ENABLED")
    
    # Harmful content categories
    blocked_categories: List[str] = Field(
        default=["violence", "hate", "self-harm", "sexual"],
        env="BLOCKED_CATEGORIES"
    )
    
    # Ethical guidelines
    ethical_guidelines_file: str = Field(default="./config/ethical_guidelines.yaml", env="ETHICAL_GUIDELINES_FILE")
    
    class Config:
        env_prefix = "SAFETY_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Data directories
    data_dir: str = Field(default="./data", env="DATA_DIR")
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    ai: AISettings = AISettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    plugins: PluginSettings = PluginSettings()
    safety: SafetySettings = SafetySettings()
    
    @validator("data_dir", "cache_dir", "logs_dir")
    def create_directories(cls, v):
        """Ensure directories exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings


# Configuration validation
def validate_configuration() -> Dict[str, List[str]]:
    """Validate the current configuration and return any issues."""
    issues = []
    
    # Check required settings
    if not settings.security.secret_key:
        issues.append("SECRET_KEY is required")
    
    if not settings.security.jwt_secret:
        issues.append("JWT_SECRET is required")
    
    if not settings.security.encryption_key:
        issues.append("ENCRYPTION_KEY is required")
    
    # Check AI API keys
    if not settings.ai.openai_api_key and not settings.ai.anthropic_api_key:
        issues.append("At least one AI API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) is required")
    
    # Check database connection
    if not settings.database.url:
        issues.append("DATABASE_URL is required")
    
    return {"errors": issues, "warnings": []}


# Export configuration for easy access
__all__ = [
    "Settings",
    "DatabaseSettings",
    "AISettings", 
    "SecuritySettings",
    "MonitoringSettings",
    "PluginSettings",
    "SafetySettings",
    "settings",
    "get_settings",
    "reload_settings",
    "validate_configuration"
] 