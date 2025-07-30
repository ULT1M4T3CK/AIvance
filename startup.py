#!/usr/bin/env python3
"""
AIvance Startup Script

This script initializes and starts all components of the AIvance system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings, validate_configuration
from database.connection import create_engine, test_connection, close_engines
from database.migrations import setup_database
from core.engine import get_engine, shutdown_engine
from auth.crud import create_user
from auth.models import UserCreate
from plugins.manager import PluginManager
from monitoring.metrics import start_metrics_collection, stop_metrics_collection
from tasks.celery_app import celery_app
from web.app import create_web_app
import uvicorn


async def initialize_database():
    """Initialize database and run migrations."""
    logging.info("Initializing database...")
    
    try:
        # Create database engine
        create_engine()
        
        # Test connection
        if not await test_connection():
            raise Exception("Database connection failed")
        
        # Setup database and run migrations
        await setup_database()
        
        logging.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        return False


async def initialize_auth():
    """Initialize authentication system."""
    logging.info("Initializing authentication system...")
    
    try:
        # Create default admin user if it doesn't exist
        admin_user = await create_user(UserCreate(
            username="admin",
            email="admin@aivance.local",
            password="Admin123!",
            is_superuser=True
        ))
        
        if admin_user:
            logging.info("Default admin user created")
        else:
            logging.info("Admin user already exists")
        
        logging.info("Authentication system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Authentication initialization failed: {e}")
        return False


async def initialize_plugins():
    """Initialize plugin system."""
    logging.info("Initializing plugin system...")
    
    try:
        plugin_manager = PluginManager()
        await plugin_manager.load_plugins()
        
        # Store plugin manager in global context
        import core.engine
        core.engine.plugin_manager = plugin_manager
        
        logging.info("Plugin system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Plugin system initialization failed: {e}")
        return False


async def initialize_monitoring():
    """Initialize monitoring and metrics."""
    logging.info("Initializing monitoring system...")
    
    try:
        await start_metrics_collection()
        logging.info("Monitoring system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Monitoring initialization failed: {e}")
        return False


async def initialize_ai_engine():
    """Initialize the AI engine."""
    logging.info("Initializing AI engine...")
    
    try:
        engine = get_engine()
        status = await engine.get_engine_status()
        
        logging.info(f"AI Engine initialized successfully")
        logging.info(f"  - Available models: {status['models_available']}")
        logging.info(f"  - Memory enabled: {status['memory_enabled']}")
        logging.info(f"  - Learning enabled: {status['learning_enabled']}")
        logging.info(f"  - Reasoning enabled: {status['reasoning_enabled']}")
        logging.info(f"  - Safety enabled: {status['safety_enabled']}")
        
        return True
        
    except Exception as e:
        logging.error(f"AI Engine initialization failed: {e}")
        return False


async def initialize_task_queue():
    """Initialize the task queue system."""
    logging.info("Initializing task queue system...")
    
    try:
        # Test Celery connection
        from celery import current_app
        current_app.control.inspect().active()
        
        logging.info("Task queue system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Task queue initialization failed: {e}")
        return False


async def initialize_web_app():
    """Initialize the web application."""
    logging.info("Initializing web application...")
    
    try:
        app = create_web_app()
        
        # Test web app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        
        if response.status_code == 200:
            logging.info("Web application initialized successfully")
            return True
        else:
            raise Exception(f"Web app health check failed: {response.status_code}")
        
    except Exception as e:
        logging.error(f"Web application initialization failed: {e}")
        return False


async def run_system_checks():
    """Run comprehensive system checks."""
    logging.info("Running system checks...")
    
    checks = [
        ("Database", initialize_database),
        ("Authentication", initialize_auth),
        ("Plugin System", initialize_plugins),
        ("Monitoring", initialize_monitoring),
        ("AI Engine", initialize_ai_engine),
        ("Task Queue", initialize_task_queue),
        ("Web Application", initialize_web_app),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            success = await check_func()
            results[check_name] = success
            status = "✓" if success else "✗"
            logging.info(f"{status} {check_name}")
        except Exception as e:
            results[check_name] = False
            logging.error(f"✗ {check_name}: {e}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logging.info(f"\nSystem Check Summary: {passed}/{total} components ready")
    
    for check_name, success in results.items():
        status = "READY" if success else "FAILED"
        logging.info(f"  {check_name}: {status}")
    
    return all(results.values())


async def start_services():
    """Start all background services."""
    logging.info("Starting background services...")
    
    try:
        # Start Celery worker (in background)
        import subprocess
        import sys
        
        # Start Celery worker
        worker_process = subprocess.Popen([
            sys.executable, "-m", "celery", "-A", "tasks.celery_app", "worker",
            "--loglevel=info", "--concurrency=2"
        ])
        
        # Start Celery beat scheduler
        beat_process = subprocess.Popen([
            sys.executable, "-m", "celery", "-A", "tasks.celery_app", "beat",
            "--loglevel=info"
        ])
        
        logging.info("Background services started successfully")
        return worker_process, beat_process
        
    except Exception as e:
        logging.error(f"Failed to start background services: {e}")
        return None, None


async def shutdown_services():
    """Shutdown all services gracefully."""
    logging.info("Shutting down services...")
    
    try:
        # Stop monitoring
        await stop_metrics_collection()
        
        # Shutdown AI engine
        await shutdown_engine()
        
        # Close database connections
        await close_engines()
        
        logging.info("Services shutdown complete")
        
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.monitoring.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.logs_dir + "/aivance.log")
        ]
    )


async def main():
    """Main startup function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AIvance Advanced AI System...")
    
    # Validate configuration
    config_validation = validate_configuration()
    if config_validation["errors"]:
        logger.error("Configuration validation failed:")
        for error in config_validation["errors"]:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    if config_validation["warnings"]:
        logger.warning("Configuration warnings:")
        for warning in config_validation["warnings"]:
            logger.warning(f"  - {warning}")
    
    try:
        # Run system checks
        if not await run_system_checks():
            logger.error("System checks failed. Exiting.")
            sys.exit(1)
        
        # Start background services
        worker_process, beat_process = await start_services()
        
        # Start web server
        logger.info("Starting web server...")
        
        app = create_web_app()
        
        config = uvicorn.Config(
            app=app,
            host=settings.host,
            port=settings.port,
            log_level=settings.monitoring.log_level.lower(),
            access_log=True,
            reload=settings.debug
        )
        
        server = uvicorn.Server(config)
        
        # Run the server
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AIvance system...")
        try:
            await shutdown_services()
            logger.info("AIvance system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 