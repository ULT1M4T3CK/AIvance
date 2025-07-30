"""
AIvance Task Queue Package

This package handles background task processing using Celery.
"""

from .celery_app import celery_app
from .ai_tasks import (
    process_ai_request,
    train_model,
    generate_embeddings,
    analyze_content,
    process_batch
)
from .maintenance_tasks import (
    cleanup_old_data,
    backup_database,
    update_models,
    health_check
)

__all__ = [
    "celery_app",
    "process_ai_request",
    "train_model", 
    "generate_embeddings",
    "analyze_content",
    "process_batch",
    "cleanup_old_data",
    "backup_database",
    "update_models",
    "health_check"
] 