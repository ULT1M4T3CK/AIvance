"""
AIvance Authentication Package

This package handles user authentication, authorization, and security.
"""

from .models import User, UserCreate, UserUpdate, UserInDB
from .security import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    verify_token,
    get_current_user,
    get_current_active_user
)
from .crud import (
    get_user_by_email,
    get_user_by_username,
    create_user,
    update_user,
    authenticate_user
)

__all__ = [
    "User",
    "UserCreate", 
    "UserUpdate",
    "UserInDB",
    "get_password_hash",
    "verify_password",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    "get_user_by_email",
    "get_user_by_username", 
    "create_user",
    "update_user",
    "authenticate_user"
] 