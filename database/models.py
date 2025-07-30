"""
Database Models

This module defines all SQLAlchemy models for the AIvance system.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    JSON, ForeignKey, Index, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

Base = declarative_base()


class User(Base):
    """User model for authentication and user management."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime(timezone=True))
    preferences = Column(JSONB, default=dict)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    learning_data = relationship("LearningData", back_populates="user", cascade="all, delete-orphan")
    model_usage = relationship("ModelUsage", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_users_username', 'username'),
        Index('idx_users_email', 'email'),
        Index('idx_users_created_at', 'created_at'),
    )


class Session(Base):
    """AI session model for tracking conversation sessions."""
    
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(255))
    model_used = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_activity = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_session_id', 'session_id'),
        Index('idx_sessions_created_at', 'created_at'),
        Index('idx_sessions_last_activity', 'last_activity'),
    )


class Memory(Base):
    """Memory model for storing user interaction memories."""
    
    __tablename__ = "memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="conversation")  # conversation, preference, fact, etc.
    importance_score = Column(Float, default=0.5)
    embedding = Column(JSONB)  # Vector embedding for similarity search
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="memories")
    
    __table_args__ = (
        Index('idx_memories_user_id', 'user_id'),
        Index('idx_memories_content_type', 'content_type'),
        Index('idx_memories_importance_score', 'importance_score'),
        Index('idx_memories_created_at', 'created_at'),
        Index('idx_memories_accessed_at', 'accessed_at'),
    )


class LearningData(Base):
    """Learning data model for storing AI learning information."""
    
    __tablename__ = "learning_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    input_data = Column(Text, nullable=False)
    output_data = Column(Text, nullable=False)
    feedback_score = Column(Float)  # User feedback score
    model_used = Column(String(100))
    learning_type = Column(String(50), default="interaction")  # interaction, feedback, correction
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="learning_data")
    
    __table_args__ = (
        Index('idx_learning_data_user_id', 'user_id'),
        Index('idx_learning_data_learning_type', 'learning_type'),
        Index('idx_learning_data_feedback_score', 'feedback_score'),
        Index('idx_learning_data_created_at', 'created_at'),
    )


class ModelUsage(Base):
    """Model usage tracking for analytics and billing."""
    
    __tablename__ = "model_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)
    tokens_used = Column(Integer, default=0)
    tokens_prompt = Column(Integer, default=0)
    tokens_completion = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    response_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="model_usage")
    
    __table_args__ = (
        Index('idx_model_usage_user_id', 'user_id'),
        Index('idx_model_usage_model_name', 'model_name'),
        Index('idx_model_usage_provider', 'provider'),
        Index('idx_model_usage_created_at', 'created_at'),
        Index('idx_model_usage_cost', 'cost_usd'),
    )


class SafetyLog(Base):
    """Safety and content filtering logs."""
    
    __tablename__ = "safety_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="prompt")  # prompt, response
    safety_score = Column(Float)
    blocked = Column(Boolean, default=False)
    blocked_reason = Column(String(255))
    filter_applied = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    metadata = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_safety_logs_user_id', 'user_id'),
        Index('idx_safety_logs_content_type', 'content_type'),
        Index('idx_safety_logs_blocked', 'blocked'),
        Index('idx_safety_logs_created_at', 'created_at'),
    )


class Plugin(Base):
    """Plugin system model for managing AI plugins."""
    
    __tablename__ = "plugins"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    version = Column(String(20), nullable=False)
    description = Column(Text)
    author = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_enabled = Column(Boolean, default=False)
    config = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSONB, default=dict)
    
    __table_args__ = (
        Index('idx_plugins_name', 'name'),
        Index('idx_plugins_is_active', 'is_active'),
        Index('idx_plugins_is_enabled', 'is_enabled'),
    )


class SystemMetrics(Base):
    """System metrics and monitoring data."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    tags = Column(JSONB, default=dict)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_system_metrics_name', 'metric_name'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
        Index('idx_system_metrics_name_timestamp', 'metric_name', 'timestamp'),
    ) 