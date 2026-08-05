"""Prometheus metrics for the voice pipeline.

Two decisions worth explaining:

**Custom histogram buckets.** Prometheus' defaults top out at 10s with coarse
resolution below 100ms, which is useless here: every latency that matters in a
voice conversation lives between 20ms and 3s, and the difference between 180ms
and 320ms of first-audio latency is the difference between a natural exchange
and an awkward one. The buckets below are chosen so the p50/p95/p99 of each
stage land on distinct boundaries.

**A private registry, not the global default.** ``prometheus_client``'s default
registry is process-global and silently accumulates duplicate collectors when a
module is imported twice or a test re-imports it, which surfaces as a
``Duplicated timeseries`` crash at startup. Owning the registry makes the
lifecycle explicit and lets tests build a clean one per case.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Final, Self

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CollectorRegistry as _Registry

from orphus.observability.logging import get_logger

__all__ = [
    "CONTENT_TYPE_LATEST",
    "GpuMetricsCollector",
    "Metrics",
    "PipelineStage",
    "get_metrics",
    "reset_metrics",
]

logger = get_logger(__name__)

CONTENT_TYPE_LATEST: Final[str] = "text/plain; version=0.0.4; charset=utf-8"

# Sub-second resolution where conversational latency actually lives, thinning
# out past 2s where the only question is "how bad", not "how much worse".
_LATENCY_BUCKETS: Final[Sequence[float]] = (
    0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.100, 0.150, 0.200, 0.300,
    0.400, 0.500, 0.750, 1.000, 1.500, 2.000, 3.000, 5.000, 10.000,
)

# End-to-end (speech-end to first audio out) is a sum of stages, so it needs a
# wider range than any single stage.
_E2E_BUCKETS: Final[Sequence[float]] = (
    0.100, 0.200, 0.300, 0.400, 0.500, 0.750, 1.000, 1.250, 1.500, 2.000,
    2.500, 3.000, 4.000, 5.000, 7.500, 10.000, 15.000, 30.000,
)

_QUEUE_DEPTH_BUCKETS: Final[Sequence[float]] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


class PipelineStage:
    """Canonical stage labels. String constants, not an enum.

    Prometheus label values are strings and these are used in hot-path metric
    calls; a plain constant avoids an ``.value`` lookup per observation.
    """

    VAD: Final[str] = "vad"
    ASR: Final[str] = "asr"
    LLM: Final[str] = "llm"
    TTS: Final[str] = "tts"
    PLAYBACK: Final[str] = "playback"


@dataclass(slots=True)
class Metrics:
    """Every metric the platform exports, bound to one registry."""

    registry: CollectorRegistry

    # --- Sessions ---
    sessions_active: Gauge = None  # type: ignore[assignment]
    sessions_started: Counter = None  # type: ignore[assignment]
    sessions_ended: Counter = None  # type: ignore[assignment]
    sessions_rejected: Counter = None  # type: ignore[assignment]
    session_duration: Histogram = None  # type: ignore[assignment]

    # --- Pipeline latency ---
    stage_latency: Histogram = None  # type: ignore[assignment]
    llm_first_token: Histogram = None  # type: ignore[assignment]
    tts_first_audio: Histogram = None  # type: ignore[assignment]
    end_to_end_latency: Histogram = None  # type: ignore[assignment]

    # --- Throughput ---
    audio_seconds_processed: Counter = None  # type: ignore[assignment]
    audio_seconds_synthesized: Counter = None  # type: ignore[assignment]
    turns_completed: Counter = None  # type: ignore[assignment]
    llm_tokens: Counter = None  # type: ignore[assignment]

    # --- Queues and backpressure ---
    queue_depth: Gauge = None  # type: ignore[assignment]
    queue_depth_observed: Histogram = None  # type: ignore[assignment]
    queue_dropped: Counter = None  # type: ignore[assignment]
    backpressure_events: Counter = None  # type: ignore[assignment]

    # --- Workers ---
    workers_busy: Gauge = None  # type: ignore[assignment]
    workers_total: Gauge = None  # type: ignore[assignment]
    worker_restarts: Counter = None  # type: ignore[assignment]

    # --- Model batching ---
    asr_batch_size: Histogram = None  # type: ignore[assignment]
    model_inference_latency: Histogram = None  # type: ignore[assignment]

    # --- Conversation behaviour ---
    barge_ins: Counter = None  # type: ignore[assignment]
    cancellations: Counter = None  # type: ignore[assignment]

    # --- Errors ---
    errors: Counter = None  # type: ignore[assignment]
    provider_requests: Counter = None  # type: ignore[assignment]
    circuit_breaker_state: Gauge = None  # type: ignore[assignment]

    # --- GPU ---
    gpu_utilization: Gauge = None  # type: ignore[assignment]
    gpu_memory_used: Gauge = None  # type: ignore[assignment]
    gpu_memory_total: Gauge = None  # type: ignore[assignment]
    gpu_temperature: Gauge = None  # type: ignore[assignment]

    # --- Health ---
    component_healthy: Gauge = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        reg = self.registry

        self.sessions_active = Gauge(
            "orphus_sessions_active", "Currently active voice sessions", registry=reg
        )
        self.sessions_started = Counter(
            "orphus_sessions_started_total", "Sessions opened", registry=reg
        )
        self.sessions_ended = Counter(
            "orphus_sessions_ended_total", "Sessions closed", ["reason"], registry=reg
        )
        self.sessions_rejected = Counter(
            "orphus_sessions_rejected_total",
            "Sessions refused admission",
            ["reason"],
            registry=reg,
        )
        self.session_duration = Histogram(
            "orphus_session_duration_seconds",
            "Wall duration of a completed session",
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600),
            registry=reg,
        )

        self.stage_latency = Histogram(
            "orphus_stage_latency_seconds",
            "Processing latency of one pipeline stage",
            ["stage"],
            buckets=_LATENCY_BUCKETS,
            registry=reg,
        )
        self.llm_first_token = Histogram(
            "orphus_llm_first_token_seconds",
            "Time from prompt dispatch to first LLM token",
            ["provider", "model"],
            buckets=_LATENCY_BUCKETS,
            registry=reg,
        )
        self.tts_first_audio = Histogram(
            "orphus_tts_first_audio_seconds",
            "Time from first text delta to first synthesised audio chunk",
            buckets=_LATENCY_BUCKETS,
            registry=reg,
        )
        self.end_to_end_latency = Histogram(
            "orphus_end_to_end_latency_seconds",
            "User speech-end to first audio out -- the number users actually feel",
            buckets=_E2E_BUCKETS,
            registry=reg,
        )

        self.audio_seconds_processed = Counter(
            "orphus_audio_seconds_processed_total", "Inbound audio transcribed", registry=reg
        )
        self.audio_seconds_synthesized = Counter(
            "orphus_audio_seconds_synthesized_total", "Outbound audio generated", registry=reg
        )
        self.turns_completed = Counter(
            "orphus_turns_completed_total", "Completed conversation turns", registry=reg
        )
        self.llm_tokens = Counter(
            "orphus_llm_tokens_total", "LLM tokens", ["provider", "kind"], registry=reg
        )

        self.queue_depth = Gauge(
            "orphus_queue_depth", "Current queue occupancy", ["queue"], registry=reg
        )
        self.queue_depth_observed = Histogram(
            "orphus_queue_depth_observed",
            "Queue occupancy sampled at enqueue -- reveals bursts a gauge misses",
            ["queue"],
            buckets=_QUEUE_DEPTH_BUCKETS,
            registry=reg,
        )
        self.queue_dropped = Counter(
            "orphus_queue_dropped_total",
            "Items shed on overflow",
            ["queue", "policy"],
            registry=reg,
        )
        self.backpressure_events = Counter(
            "orphus_backpressure_events_total",
            "Times admission control shed or deferred load",
            ["stage"],
            registry=reg,
        )

        self.workers_busy = Gauge(
            "orphus_workers_busy", "Workers currently serving", ["pool"], registry=reg
        )
        self.workers_total = Gauge(
            "orphus_workers_total", "Workers in the pool", ["pool"], registry=reg
        )
        self.worker_restarts = Counter(
            "orphus_worker_restarts_total", "Workers restarted after failure",
            ["pool", "reason"], registry=reg,
        )

        self.asr_batch_size = Histogram(
            "orphus_asr_batch_size",
            "Sessions coalesced into one ASR forward pass -- the key throughput lever",
            buckets=(1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32),
            registry=reg,
        )
        self.model_inference_latency = Histogram(
            "orphus_model_inference_seconds",
            "Raw model forward-pass latency, excluding queueing",
            ["model"],
            buckets=_LATENCY_BUCKETS,
            registry=reg,
        )

        self.barge_ins = Counter(
            "orphus_barge_ins_total", "User interruptions of assistant speech", registry=reg
        )
        self.cancellations = Counter(
            "orphus_cancellations_total", "Cancelled in-flight requests", ["stage"], registry=reg
        )

        self.errors = Counter(
            "orphus_errors_total", "Errors by component and type",
            ["component", "type"], registry=reg,
        )
        self.provider_requests = Counter(
            "orphus_provider_requests_total", "LLM provider calls",
            ["provider", "outcome"], registry=reg,
        )
        self.circuit_breaker_state = Gauge(
            "orphus_circuit_breaker_state",
            "Circuit breaker: 0 closed, 1 half-open, 2 open",
            ["provider"],
            registry=reg,
        )

        self.gpu_utilization = Gauge(
            "orphus_gpu_utilization_percent", "GPU core utilisation", ["gpu"], registry=reg
        )
        self.gpu_memory_used = Gauge(
            "orphus_gpu_memory_used_bytes", "VRAM in use", ["gpu"], registry=reg
        )
        self.gpu_memory_total = Gauge(
            "orphus_gpu_memory_total_bytes", "VRAM installed", ["gpu"], registry=reg
        )
        self.gpu_temperature = Gauge(
            "orphus_gpu_temperature_celsius", "GPU temperature", ["gpu"], registry=reg
        )

        self.component_healthy = Gauge(
            "orphus_component_healthy",
            "Health probe result: 1 healthy, 0 unhealthy",
            ["component"],
            registry=reg,
        )

    # -- helpers ------------------------------------------------------------

    @contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        """Record the wall time of a pipeline stage.

        Uses ``perf_counter`` rather than ``time()`` so an NTP step mid-turn
        cannot produce a negative or absurd latency sample.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_latency.labels(stage=stage).observe(time.perf_counter() - started)

    def record_error(self, component: str, exc: BaseException) -> None:
        """Count an error, labelled by component and exception type."""
        self.errors.labels(component=component, type=type(exc).__name__).inc()

    def render(self) -> bytes:
        """Serialise the registry in Prometheus exposition format."""
        return generate_latest(self.registry)


# ---------------------------------------------------------------------------
# GPU collection
# ---------------------------------------------------------------------------


class GpuMetricsCollector:
    """Polls NVML for per-device utilisation, VRAM, and temperature.

    Degrades to a no-op when NVML is unavailable (no driver, no GPU, or the
    ``observability`` extra not installed). A dev box without a GPU must still
    be able to start the API and scrape ``/metrics``.
    """

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._handles: list[tuple[str, object]] = []
        self._nvml: object | None = None
        self._available = False

    def start(self) -> Self:
        """Initialise NVML and enumerate devices. Never raises."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                self._handles.append((str(index), handle))
                total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                self._metrics.gpu_memory_total.labels(gpu=str(index)).set(total)
            self._available = True
            logger.info("gpu.metrics.initialised", extra={"device_count": count})
        except Exception as exc:
            logger.info("gpu.metrics.unavailable", extra={"reason": str(exc)})
            self._available = False
        return self

    @property
    def available(self) -> bool:
        """Whether NVML initialised and at least one device was found."""
        return self._available and bool(self._handles)

    def poll(self) -> None:
        """Sample every device once. Never raises."""
        if not self.available or self._nvml is None:
            return
        pynvml = self._nvml
        for label, handle in self._handles:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
                temp = pynvml.nvmlDeviceGetTemperature(  # type: ignore[attr-defined]
                    handle, pynvml.NVML_TEMPERATURE_GPU  # type: ignore[attr-defined]
                )
            except Exception as exc:
                logger.warning("gpu.metrics.poll_failed", extra={"gpu": label, "error": str(exc)})
                continue
            self._metrics.gpu_utilization.labels(gpu=label).set(util.gpu)
            self._metrics.gpu_memory_used.labels(gpu=label).set(mem.used)
            self._metrics.gpu_memory_total.labels(gpu=label).set(mem.total)
            self._metrics.gpu_temperature.labels(gpu=label).set(temp)

    def shutdown(self) -> None:
        """Release NVML. Idempotent and never raises."""
        if self._nvml is not None and self._available:
            with suppress(Exception):
                self._nvml.nvmlShutdown()  # type: ignore[attr-defined]
        self._available = False
        self._handles.clear()


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------

_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Return the process-wide metrics, creating them on first use."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics(registry=_Registry())
    return _metrics


def reset_metrics() -> Metrics:
    """Rebuild the metrics on a fresh registry. For tests only."""
    global _metrics
    _metrics = Metrics(registry=_Registry())
    return _metrics
