#!/usr/bin/env python3
"""
AIvance - Advanced AI System

Main entry point for the AIvance system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings, validate_configuration
from core.engine import get_engine, shutdown_engine


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
    """Main application entry point."""
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
        # Initialize AI engine
        logger.info("Initializing AI Engine...")
        engine = get_engine()
        
        # Get engine status
        status = await engine.get_engine_status()
        logger.info(f"AI Engine initialized successfully")
        logger.info(f"  - Available models: {status['models_available']}")
        logger.info(f"  - Memory enabled: {status['memory_enabled']}")
        logger.info(f"  - Learning enabled: {status['learning_enabled']}")
        logger.info(f"  - Reasoning enabled: {status['reasoning_enabled']}")
        logger.info(f"  - Safety enabled: {status['safety_enabled']}")
        
        # Keep the system running
        logger.info("AIvance system is ready and running...")
        logger.info("Press Ctrl+C to stop the system")
        
        # Run indefinitely
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AIvance system...")
        try:
            await shutdown_engine()
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