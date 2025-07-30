"""
Database Connection Management

This module handles database connections, session management, and connection pooling.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from config import settings

logger = logging.getLogger(__name__)

# Global engine and session factory instances
_async_engine: Optional[AsyncSession] = None
_async_session_factory: Optional[async_sessionmaker] = None
_sync_engine = None
_sync_session_factory = None


def get_database_url() -> str:
    """Get the database URL from settings."""
    return settings.database.url


def create_engine():
    """Create and configure the database engine."""
    global _async_engine, _async_session_factory, _sync_engine, _sync_session_factory
    
    database_url = get_database_url()
    
    # Create async engine
    _async_engine = create_async_engine(
        database_url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600,
        poolclass=QueuePool
    )
    
    # Create async session factory
    _async_session_factory = async_sessionmaker(
        bind=_async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Create sync engine for migrations
    _sync_engine = create_sync_engine(
        database_url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow
    )
    
    # Create sync session factory
    _sync_session_factory = sessionmaker(
        bind=_sync_engine,
        expire_on_commit=False
    )
    
    logger.info("Database engines created successfully")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    if not _async_session_factory:
        create_engine()
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context():
    """Get a database session context manager."""
    if not _async_session_factory:
        create_engine()
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """Get a synchronous database session for migrations."""
    if not _sync_session_factory:
        create_engine()
    
    return _sync_session_factory()


async def close_engines():
    """Close all database engines."""
    global _async_engine, _sync_engine
    
    if _async_engine:
        await _async_engine.dispose()
        logger.info("Async database engine closed")
    
    if _sync_engine:
        _sync_engine.dispose()
        logger.info("Sync database engine closed")


async def test_connection():
    """Test the database connection."""
    try:
        async with get_session_context() as session:
            await session.execute("SELECT 1")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_engine_info():
    """Get information about the database engines."""
    return {
        "async_engine_created": _async_engine is not None,
        "sync_engine_created": _sync_engine is not None,
        "database_url": get_database_url(),
        "pool_size": settings.database.pool_size,
        "max_overflow": settings.database.max_overflow
    } 