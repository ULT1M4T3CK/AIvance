"""
Database Migrations

This module handles database migrations using Alembic.
"""

import os
import logging
from pathlib import Path
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from config import settings
from .connection import get_sync_session
from .models import Base

logger = logging.getLogger(__name__)


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Create Alembic config
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database.url)
    alembic_cfg.set_main_option("file_template", "%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s")
    
    return alembic_cfg


def init_migrations():
    """Initialize Alembic migrations if not already initialized."""
    project_root = Path(__file__).parent.parent
    alembic_dir = project_root / "alembic"
    
    if not alembic_dir.exists():
        logger.info("Initializing Alembic migrations...")
        
        # Create alembic config
        alembic_cfg = get_alembic_config()
        
        # Initialize alembic
        command.init(alembic_cfg, str(alembic_dir))
        
        # Update env.py to import our models
        env_py_path = alembic_dir / "env.py"
        if env_py_path.exists():
            with open(env_py_path, 'r') as f:
                content = f.read()
            
            # Add model imports
            model_import = """
# Import your model's MetaData object here
from database.models import Base
target_metadata = Base.metadata
"""
            
            # Replace the target_metadata line
            content = content.replace(
                "target_metadata = None",
                model_import
            )
            
            with open(env_py_path, 'w') as f:
                f.write(content)
        
        logger.info("Alembic migrations initialized successfully")
    else:
        logger.info("Alembic migrations already initialized")


def create_migration(message: str):
    """Create a new migration."""
    try:
        alembic_cfg = get_alembic_config()
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info(f"Migration created: {message}")
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        raise


def run_migrations():
    """Run all pending migrations."""
    try:
        alembic_cfg = get_alembic_config()
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise


def rollback_migration(revision: str):
    """Rollback to a specific migration revision."""
    try:
        alembic_cfg = get_alembic_config()
        command.downgrade(alembic_cfg, revision)
        logger.info(f"Rolled back to revision: {revision}")
    except Exception as e:
        logger.error(f"Failed to rollback migration: {e}")
        raise


def get_migration_history():
    """Get migration history."""
    try:
        alembic_cfg = get_alembic_config()
        script_dir = ScriptDirectory.from_config(alembic_cfg)
        
        history = []
        for revision in script_dir.walk_revisions():
            history.append({
                "revision": revision.revision,
                "down_revision": revision.down_revision,
                "message": revision.message,
                "date": revision.date
            })
        
        return history
    except Exception as e:
        logger.error(f"Failed to get migration history: {e}")
        return []


def get_current_revision():
    """Get current database revision."""
    try:
        with get_sync_session() as session:
            result = session.execute(text("SELECT version_num FROM alembic_version"))
            current_revision = result.scalar()
            return current_revision
    except Exception as e:
        logger.error(f"Failed to get current revision: {e}")
        return None


def check_migrations_status():
    """Check the status of migrations."""
    try:
        alembic_cfg = get_alembic_config()
        current_revision = get_current_revision()
        
        # Get the latest revision
        script_dir = ScriptDirectory.from_config(alembic_cfg)
        head_revision = script_dir.get_current_head()
        
        return {
            "current_revision": current_revision,
            "head_revision": head_revision,
            "is_up_to_date": current_revision == head_revision,
            "pending_migrations": current_revision != head_revision
        }
    except Exception as e:
        logger.error(f"Failed to check migration status: {e}")
        return {
            "current_revision": None,
            "head_revision": None,
            "is_up_to_date": False,
            "pending_migrations": False,
            "error": str(e)
        }


def create_initial_migration():
    """Create the initial migration with all models."""
    try:
        # Initialize migrations if needed
        init_migrations()
        
        # Create initial migration
        create_migration("Initial migration - create all tables")
        
        logger.info("Initial migration created successfully")
    except Exception as e:
        logger.error(f"Failed to create initial migration: {e}")
        raise


def setup_database():
    """Complete database setup including migrations."""
    try:
        logger.info("Setting up database...")
        
        # Initialize migrations
        init_migrations()
        
        # Check if we need to create initial migration
        status = check_migrations_status()
        
        if not status["current_revision"]:
            # No migrations have been run, create initial migration
            create_initial_migration()
        
        # Run any pending migrations
        if status["pending_migrations"]:
            run_migrations()
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise 