"""
Metrics Collection

This module handles collection and management of system metrics.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

from config import settings
from database.connection import get_session_context
from database.models import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class MetricsCollector:
    """
    Collects and manages system metrics.
    
    This class handles:
    - Metric collection and storage
    - Real-time metric aggregation
    - Metric querying and reporting
    - Performance monitoring
    """
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background task for metric persistence
        self._persistence_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Metrics collector initialized")
    
    async def start(self):
        """Start the metrics collector."""
        if self._running:
            return
        
        self._running = True
        self._persistence_task = asyncio.create_task(self._persistence_loop())
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector."""
        if not self._running:
            return
        
        self._running = False
        
        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
        
        # Persist final metrics
        await self._persist_metrics()
        logger.info("Metrics collector stopped")
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            
            # Store metric point
            metric = MetricPoint(
                name=f"{name}_counter",
                value=float(self.counters[name]),
                timestamp=datetime.utcnow(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            
            # Store metric point
            metric = MetricPoint(
                name=f"{name}_gauge",
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self._lock:
            self.histograms[name].append(value)
            
            # Keep only last 1000 values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            
            # Store metric point
            metric = MetricPoint(
                name=f"{name}_histogram",
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        with self._lock:
            self.timers[name].append(duration)
            
            # Keep only last 1000 values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            
            # Store metric point
            metric = MetricPoint(
                name=f"{name}_timer",
                value=duration,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                unit="seconds"
            )
            self.metrics[name].append(metric)
    
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags)
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        with self._lock:
            return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self.histograms.get(name, [])
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        with self._lock:
            values = self.timers.get(name, [])
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: self.get_histogram_stats(name) 
                    for name in self.histograms.keys()
                },
                "timers": {
                    name: self.get_timer_stats(name) 
                    for name in self.timers.keys()
                },
                "total_metric_points": sum(len(metrics) for metrics in self.metrics.values())
            }
            return summary
    
    def get_metric_history(
        self, 
        name: str, 
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MetricPoint]:
        """Get metric history for a specific metric."""
        with self._lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
            
            # Filter by time if specified
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            # Limit results
            return metrics[-limit:]
    
    async def _persistence_loop(self):
        """Background loop for persisting metrics to database."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Persist every minute
                await self._persist_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics persistence loop: {e}")
    
    async def _persist_metrics(self):
        """Persist current metrics to database."""
        try:
            with self._lock:
                metrics_to_persist = []
                
                # Collect all current metric points
                for metric_name, metric_queue in self.metrics.items():
                    for metric in metric_queue:
                        metrics_to_persist.append({
                            "metric_name": metric.name,
                            "metric_value": metric.value,
                            "metric_unit": metric.unit,
                            "tags": metric.tags,
                            "timestamp": metric.timestamp
                        })
                
                # Clear the queues after persisting
                for queue in self.metrics.values():
                    queue.clear()
            
            # Persist to database
            if metrics_to_persist:
                async with get_session_context() as session:
                    for metric_data in metrics_to_persist:
                        db_metric = SystemMetrics(**metric_data)
                        session.add(db_metric)
                    
                    await session.commit()
                
                logger.debug(f"Persisted {len(metrics_to_persist)} metric points")
                
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    # AI-specific metrics
    def record_model_usage(
        self, 
        model_name: str, 
        tokens_used: int, 
        response_time: float,
        cost: float,
        success: bool = True
    ):
        """Record AI model usage metrics."""
        tags = {"model": model_name, "success": str(success)}
        
        self.increment_counter("model_requests", 1, tags)
        self.increment_counter("model_tokens", tokens_used, tags)
        self.record_timer("model_response_time", response_time, tags)
        self.increment_counter("model_cost", int(cost * 1000), tags)  # Store as millicents
    
    def record_user_interaction(
        self, 
        user_id: str, 
        interaction_type: str,
        duration: float
    ):
        """Record user interaction metrics."""
        tags = {"user_id": user_id, "type": interaction_type}
        
        self.increment_counter("user_interactions", 1, tags)
        self.record_timer("interaction_duration", duration, tags)
    
    def record_system_performance(
        self, 
        component: str, 
        operation: str,
        duration: float,
        success: bool = True
    ):
        """Record system performance metrics."""
        tags = {"component": component, "operation": operation, "success": str(success)}
        
        self.record_timer(f"{component}_{operation}_duration", duration, tags)
        self.increment_counter(f"{component}_{operation}_count", 1, tags)


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.tags)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


async def start_metrics_collection():
    """Start the global metrics collection."""
    collector = get_metrics_collector()
    await collector.start()


async def stop_metrics_collection():
    """Stop the global metrics collection."""
    collector = get_metrics_collector()
    await collector.stop() 