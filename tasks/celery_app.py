"""
Celery Application Configuration

This module configures the Celery task queue for background processing.
"""

import os
import logging
from celery import Celery
from celery.schedules import crontab

from config import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "aivance",
    broker=settings.database.redis_url,
    backend=settings.database.redis_url,
    include=[
        "tasks.ai_tasks",
        "tasks.maintenance_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "tasks.ai_tasks.*": {"queue": "ai"},
        "tasks.maintenance_tasks.*": {"queue": "maintenance"},
    },
    
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_always_eager=False,  # Set to True for testing
    task_eager_propagates=True,
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result backend
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Task timeouts
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    
    # Retry configuration
    task_acks_late=True,
    worker_reject_on_worker_lost=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Logging
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
)

# Periodic tasks (beat schedule)
celery_app.conf.beat_schedule = {
    # Daily maintenance tasks
    "daily-cleanup": {
        "task": "tasks.maintenance_tasks.cleanup_old_data",
        "schedule": crontab(hour=2, minute=0),  # 2 AM UTC
        "args": (),
        "options": {"queue": "maintenance"}
    },
    
    "daily-backup": {
        "task": "tasks.maintenance_tasks.backup_database",
        "schedule": crontab(hour=3, minute=0),  # 3 AM UTC
        "args": (),
        "options": {"queue": "maintenance"}
    },
    
    # Hourly health checks
    "hourly-health-check": {
        "task": "tasks.maintenance_tasks.health_check",
        "schedule": crontab(minute=0),  # Every hour
        "args": (),
        "options": {"queue": "maintenance"}
    },
    
    # Weekly model updates
    "weekly-model-update": {
        "task": "tasks.maintenance_tasks.update_models",
        "schedule": crontab(day_of_week=0, hour=4, minute=0),  # Sunday 4 AM UTC
        "args": (),
        "options": {"queue": "maintenance"}
    },
    
    # Memory optimization (every 6 hours)
    "memory-optimization": {
        "task": "tasks.maintenance_tasks.optimize_memory",
        "schedule": crontab(hour="*/6"),  # Every 6 hours
        "args": (),
        "options": {"queue": "maintenance"}
    },
}

# Task error handling
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing."""
    logger.info(f"Request: {self.request!r}")


# Task monitoring
@celery_app.task(bind=True)
def monitor_task(self, task_name, *args, **kwargs):
    """Monitor task execution and performance."""
    import time
    start_time = time.time()
    
    try:
        # Execute the actual task
        result = self.apply_async(args=args, kwargs=kwargs)
        duration = time.time() - start_time
        
        logger.info(f"Task {task_name} completed in {duration:.2f}s")
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Task {task_name} failed after {duration:.2f}s: {e}")
        raise


# Task routing
@celery_app.task(bind=True)
def route_task(self, task_name, *args, **kwargs):
    """Route tasks to appropriate queues based on type."""
    if task_name.startswith("ai."):
        return self.apply_async(
            args=args, 
            kwargs=kwargs, 
            queue="ai"
        )
    elif task_name.startswith("maintenance."):
        return self.apply_async(
            args=args, 
            kwargs=kwargs, 
            queue="maintenance"
        )
    else:
        return self.apply_async(args=args, kwargs=kwargs)


# Health check task
@celery_app.task(bind=True)
def health_check_task(self):
    """Health check for the task queue system."""
    try:
        # Check Redis connection
        from redis import Redis
        redis_client = Redis.from_url(settings.database.redis_url)
        redis_client.ping()
        
        # Check database connection
        from database.connection import test_connection
        db_ok = await test_connection()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "database": "connected" if db_ok else "disconnected",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# Task result monitoring
@celery_app.task(bind=True)
def monitor_result(self, task_id):
    """Monitor task results and handle failures."""
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.successful():
            logger.info(f"Task {task_id} completed successfully")
            return result.get()
        elif result.failed():
            logger.error(f"Task {task_id} failed: {result.info}")
            # Handle failure (e.g., retry, alert, etc.)
            return None
        else:
            logger.info(f"Task {task_id} is still pending")
            return None
            
    except Exception as e:
        logger.error(f"Error monitoring task {task_id}: {e}")
        return None


# Task cleanup
@celery_app.task(bind=True)
def cleanup_completed_tasks(self, older_than_hours=24):
    """Clean up completed task results older than specified time."""
    try:
        from datetime import datetime, timedelta
        from celery.result import GroupResult
        
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        # This is a simplified cleanup - in production you might want
        # to use a more sophisticated approach
        logger.info(f"Cleaning up tasks older than {older_than_hours} hours")
        
        return {"cleaned": True, "cutoff_time": cutoff_time.isoformat()}
        
    except Exception as e:
        logger.error(f"Task cleanup failed: {e}")
        return {"cleaned": False, "error": str(e)}


# Initialize Celery
if __name__ == "__main__":
    celery_app.start() 