"""
Authentication Models

This module defines Pydantic models for user authentication and management.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator
from uuid import UUID


class UserBase(BaseModel):
    """Base user model with common fields."""
    
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    """Model for creating a new user."""
    
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseModel):
    """Model for updating user information."""
    
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class UserInDB(UserBase):
    """Internal user model with database fields."""
    
    id: UUID
    hashed_password: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True


class User(UserBase):
    """Public user model for API responses."""
    
    id: UUID
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token model for authentication."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token data model for JWT payload."""
    
    username: Optional[str] = None
    user_id: Optional[UUID] = None
    scopes: list[str] = Field(default_factory=list)


class LoginRequest(BaseModel):
    """Login request model."""
    
    username: str
    password: str


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""
    
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserPreferences(BaseModel):
    """User preferences model."""
    
    language: str = "en"
    timezone: str = "UTC"
    theme: str = "light"
    ai_model_preference: Optional[str] = None
    notification_settings: Dict[str, bool] = Field(default_factory=dict)
    privacy_settings: Dict[str, bool] = Field(default_factory=dict)
    custom_settings: Dict[str, Any] = Field(default_factory=dict) 