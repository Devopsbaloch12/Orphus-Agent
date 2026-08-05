"""Observability: structured logging, metrics, tracing, and health probes."""

from __future__ import annotations

from orphus.observability.health import (
    ComponentHealth,
    HealthRegistry,
    HealthStatus,
    SystemHealth,
    check_cuda,
    check_disk,
    check_memory,
    make_callable_check,
)
from orphus.observability.logging import (
    bind_session,
    bind_turn,
    configure_logging,
    current_session_id,
    get_logger,
)
from orphus.observability.metrics import (
    CONTENT_TYPE_LATEST,
    GpuMetricsCollector,
    Metrics,
    PipelineStage,
    get_metrics,
    reset_metrics,
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "ComponentHealth",
    "GpuMetricsCollector",
    "HealthRegistry",
    "HealthStatus",
    "Metrics",
    "PipelineStage",
    "SystemHealth",
    "bind_session",
    "bind_turn",
    "check_cuda",
    "check_disk",
    "check_memory",
    "configure_logging",
    "current_session_id",
    "get_logger",
    "get_metrics",
    "make_callable_check",
    "reset_metrics",
]
