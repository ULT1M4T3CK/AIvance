"""
Authentication CRUD Operations

This module handles database operations for user authentication and management.
"""

import logging
from typing import Optional, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from database.connection import get_session_context
from database.models import User as DBUser
from .models import UserCreate, UserUpdate, UserInDB
from .security import get_password_hash, verify_password

logger = logging.getLogger(__name__)


async def get_user_by_id(user_id: UUID) -> Optional[UserInDB]:
    """Get a user by ID."""
    try:
        async with get_session_context() as session:
            result = await session.execute(
                select(DBUser).where(DBUser.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                return UserInDB.model_validate(user)
            return None
    
    except Exception as e:
        logger.error(f"Error getting user by ID {user_id}: {e}")
        return None


async def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get a user by email."""
    try:
        async with get_session_context() as session:
            result = await session.execute(
                select(DBUser).where(DBUser.email == email)
            )
            user = result.scalar_one_or_none()
            
            if user:
                return UserInDB.model_validate(user)
            return None
    
    except Exception as e:
        logger.error(f"Error getting user by email {email}: {e}")
        return None


async def get_user_by_username(username: str) -> Optional[UserInDB]:
    """Get a user by username."""
    try:
        async with get_session_context() as session:
            result = await session.execute(
                select(DBUser).where(DBUser.username == username)
            )
            user = result.scalar_one_or_none()
            
            if user:
                return UserInDB.model_validate(user)
            return None
    
    except Exception as e:
        logger.error(f"Error getting user by username {username}: {e}")
        return None


async def get_users(
    skip: int = 0, 
    limit: int = 100,
    active_only: bool = True
) -> List[UserInDB]:
    """Get a list of users."""
    try:
        async with get_session_context() as session:
            query = select(DBUser)
            
            if active_only:
                query = query.where(DBUser.is_active == True)
            
            query = query.offset(skip).limit(limit)
            
            result = await session.execute(query)
            users = result.scalars().all()
            
            return [UserInDB.model_validate(user) for user in users]
    
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return []


async def create_user(user_create: UserCreate) -> Optional[UserInDB]:
    """Create a new user."""
    try:
        # Check if user already exists
        existing_user = await get_user_by_email(user_create.email)
        if existing_user:
            logger.warning(f"User with email {user_create.email} already exists")
            return None
        
        existing_user = await get_user_by_username(user_create.username)
        if existing_user:
            logger.warning(f"User with username {user_create.username} already exists")
            return None
        
        # Hash the password
        hashed_password = get_password_hash(user_create.password)
        
        # Create user data
        user_data = user_create.model_dump(exclude={"password"})
        user_data["hashed_password"] = hashed_password
        
        async with get_session_context() as session:
            db_user = DBUser(**user_data)
            session.add(db_user)
            await session.commit()
            await session.refresh(db_user)
            
            logger.info(f"Created new user: {user_create.username}")
            return UserInDB.model_validate(db_user)
    
    except Exception as e:
        logger.error(f"Error creating user {user_create.username}: {e}")
        return None


async def update_user(
    user_id: UUID, 
    user_update: UserUpdate
) -> Optional[UserInDB]:
    """Update a user."""
    try:
        async with get_session_context() as session:
            # Get current user
            result = await session.execute(
                select(DBUser).where(DBUser.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"User with ID {user_id} not found")
                return None
            
            # Update fields
            update_data = user_update.model_dump(exclude_unset=True)
            
            if update_data:
                await session.execute(
                    update(DBUser)
                    .where(DBUser.id == user_id)
                    .values(**update_data)
                )
                await session.commit()
                
                # Get updated user
                result = await session.execute(
                    select(DBUser).where(DBUser.id == user_id)
                )
                updated_user = result.scalar_one_or_none()
                
                logger.info(f"Updated user: {user.username}")
                return UserInDB.model_validate(updated_user)
            
            return UserInDB.model_validate(user)
    
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        return None


async def delete_user(user_id: UUID) -> bool:
    """Delete a user."""
    try:
        async with get_session_context() as session:
            # Get user first
            result = await session.execute(
                select(DBUser).where(DBUser.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"User with ID {user_id} not found")
                return False
            
            # Delete user
            await session.execute(
                delete(DBUser).where(DBUser.id == user_id)
            )
            await session.commit()
            
            logger.info(f"Deleted user: {user.username}")
            return True
    
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return False


async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password."""
    try:
        user = await get_user_by_username(username)
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {username}")
            return None
        
        # Update last login
        await update_user(
            user.id, 
            UserUpdate(last_login=datetime.utcnow())
        )
        
        logger.info(f"User authenticated successfully: {username}")
        return user
    
    except Exception as e:
        logger.error(f"Error authenticating user {username}: {e}")
        return None


async def change_password(
    user_id: UUID, 
    current_password: str, 
    new_password: str
) -> bool:
    """Change a user's password."""
    try:
        user = await get_user_by_id(user_id)
        if not user:
            return False
        
        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            return False
        
        # Hash new password
        hashed_new_password = get_password_hash(new_password)
        
        # Update password
        success = await update_user(
            user_id,
            UserUpdate(hashed_password=hashed_new_password)
        )
        
        if success:
            logger.info(f"Password changed for user: {user.username}")
        
        return success is not None
    
    except Exception as e:
        logger.error(f"Error changing password for user {user_id}: {e}")
        return False


async def reset_password(email: str, new_password: str) -> bool:
    """Reset a user's password (for password reset flow)."""
    try:
        user = await get_user_by_email(email)
        if not user:
            return False
        
        # Hash new password
        hashed_new_password = get_password_hash(new_password)
        
        # Update password
        success = await update_user(
            user.id,
            UserUpdate(hashed_password=hashed_new_password)
        )
        
        if success:
            logger.info(f"Password reset for user: {user.username}")
        
        return success is not None
    
    except Exception as e:
        logger.error(f"Error resetting password for email {email}: {e}")
        return False


async def update_user_preferences(
    user_id: UUID, 
    preferences: dict
) -> Optional[UserInDB]:
    """Update user preferences."""
    try:
        return await update_user(
            user_id,
            UserUpdate(preferences=preferences)
        )
    
    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id}: {e}")
        return None


async def get_user_stats(user_id: UUID) -> dict:
    """Get user statistics."""
    try:
        async with get_session_context() as session:
            # Get user with relationships
            result = await session.execute(
                select(DBUser)
                .options(
                    selectinload(DBUser.sessions),
                    selectinload(DBUser.memories),
                    selectinload(DBUser.learning_data),
                    selectinload(DBUser.model_usage)
                )
                .where(DBUser.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return {}
            
            return {
                "total_sessions": len(user.sessions),
                "active_sessions": len([s for s in user.sessions if s.is_active]),
                "total_memories": len(user.memories),
                "total_learning_data": len(user.learning_data),
                "total_model_usage": len(user.model_usage),
                "total_cost": sum(usage.cost_usd for usage in user.model_usage),
                "total_tokens": sum(usage.tokens_used for usage in user.model_usage)
            }
    
    except Exception as e:
        logger.error(f"Error getting stats for user {user_id}: {e}")
        return {} 