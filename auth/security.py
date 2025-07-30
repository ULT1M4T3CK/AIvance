"""
Authentication Security

This module handles password hashing, JWT tokens, and security utilities.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from .models import TokenData, UserInDB
from .crud import get_user_by_username

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None,
    scopes: Optional[list[str]] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.security.jwt_expiration
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    if scopes:
        to_encode["scopes"] = scopes
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.security.jwt_secret, 
        algorithm=settings.security.jwt_algorithm
    )
    
    return encoded_jwt


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=30)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret,
        algorithm=settings.security.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(
            token, 
            settings.security.jwt_secret, 
            algorithms=[settings.security.jwt_algorithm]
        )
        
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        scopes: list[str] = payload.get("scopes", [])
        token_type: str = payload.get("type", "access")
        
        if username is None:
            return None
        
        return TokenData(
            username=username,
            user_id=UUID(user_id) if user_id else None,
            scopes=scopes
        )
    
    except JWTError as e:
        logger.warning(f"JWT token verification failed: {e}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserInDB:
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = verify_token(credentials.credentials)
        if token_data is None:
            raise credentials_exception
        
        user = await get_user_by_username(token_data.username)
        if user is None:
            raise credentials_exception
        
        return user
    
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise credentials_exception


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_superuser(
    current_user: UserInDB = Depends(get_current_active_user)
) -> UserInDB:
    """Get the current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def check_permissions(
    required_scopes: list[str],
    current_user: UserInDB = Depends(get_current_active_user)
) -> UserInDB:
    """Check if user has required permissions."""
    # Superusers have all permissions
    if current_user.is_superuser:
        return current_user
    
    # TODO: Implement scope-based permission checking
    # For now, just return the user
    return current_user


def generate_password_reset_token(email: str) -> str:
    """Generate a password reset token."""
    delta = timedelta(hours=settings.security.jwt_expiration // 60)
    now = datetime.now(timezone.utc)
    expires = now + delta
    
    to_encode = {
        "exp": expires,
        "iat": now,
        "sub": email,
        "type": "password_reset"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret,
        algorithm=settings.security.jwt_algorithm
    )
    
    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify a password reset token and return the email."""
    try:
        payload = jwt.decode(
            token,
            settings.security.jwt_secret,
            algorithms=[settings.security.jwt_algorithm]
        )
        
        email: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if email is None or token_type != "password_reset":
            return None
        
        return email
    
    except JWTError:
        return None


def create_api_key(user_id: UUID, scopes: Optional[list[str]] = None) -> str:
    """Create an API key for a user."""
    to_encode = {
        "user_id": str(user_id),
        "type": "api_key",
        "iat": datetime.now(timezone.utc),
        "scopes": scopes or []
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret,
        algorithm=settings.security.jwt_algorithm
    )
    
    return encoded_jwt


def verify_api_key(api_key: str) -> Optional[TokenData]:
    """Verify an API key and return token data."""
    try:
        payload = jwt.decode(
            api_key,
            settings.security.jwt_secret,
            algorithms=[settings.security.jwt_algorithm]
        )
        
        user_id: str = payload.get("user_id")
        token_type: str = payload.get("type")
        scopes: list[str] = payload.get("scopes", [])
        
        if user_id is None or token_type != "api_key":
            return None
        
        return TokenData(
            user_id=UUID(user_id),
            scopes=scopes
        )
    
    except JWTError:
        return None 