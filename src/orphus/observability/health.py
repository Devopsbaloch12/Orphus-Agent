"""Health probes for every dependency the platform can lose.

Three rules the whole module obeys:

1. **A probe never raises.** A failing probe reports ``UNHEALTHY`` with a
   diagnosis. If the health endpoint could throw, the one tool you use to find
   out what is broken would break exactly when something is broken.
2. **Every probe is bounded by a timeout.** A hung TCP connect to a dead
   PostgreSQL must not wedge the readiness endpoint and get the whole pod
   killed for an unrelated reason.
3. **Probes are injected, not imported.** Checks take callables rather than
   importing the ASR/TTS/Redis modules, so this file has no dependency on the
   GPU stack and the API can report health on a machine with no CUDA.

``DEGRADED`` is a distinct state from ``UNHEALTHY`` on purpose: the platform is
built to shed quality rather than fall over, and liveness must not fail just
because the system is running hot. Only ``UNHEALTHY`` should take a pod out of
rotation.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from orphus.observability.logging import get_logger

__all__ = [
    "ComponentHealth",
    "HealthRegistry",
    "HealthStatus",
    "SystemHealth",
    "check_cuda",
    "check_disk",
    "check_memory",
    "make_callable_check",
]

logger = get_logger(__name__)

_DEFAULT_PROBE_TIMEOUT_S: Final[float] = 5.0


class HealthStatus(StrEnum):
    """Outcome of a probe.

    ``DEGRADED`` means "serving, but with reduced headroom or quality". It is
    reported and alerted on, but it must not fail a liveness check.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """Result of one probe. Satisfies the ``HealthReport`` protocol."""

    component: str
    status: HealthStatus
    detail: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """Whether the component is serving. ``DEGRADED`` still counts as serving."""
        return self.status is not HealthStatus.UNHEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Render for the health endpoint payload."""
        payload: dict[str, Any] = {"component": self.component, "status": self.status.value}
        if self.detail:
            payload["detail"] = self.detail
        if self.latency_ms is not None:
            payload["latency_ms"] = round(self.latency_ms, 2)
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True, slots=True)
class SystemHealth:
    """Aggregate of every component probe."""

    status: HealthStatus
    components: Sequence[ComponentHealth]
    checked_at: float

    @property
    def healthy(self) -> bool:
        """Whether the platform as a whole is serving."""
        return self.status is not HealthStatus.UNHEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Render for the health endpoint payload."""
        return {
            "status": self.status.value,
            "checked_at": self.checked_at,
            "components": [c.to_dict() for c in self.components],
        }


ProbeFn = Callable[[], Awaitable[ComponentHealth]]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class HealthRegistry:
    """Runs registered probes concurrently and aggregates the result."""

    def __init__(self, *, timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S) -> None:
        self._probes: dict[str, ProbeFn] = {}
        self._timeout_s = timeout_s

    def register(self, name: str, probe: ProbeFn) -> None:
        """Add a probe, replacing any existing one of the same name."""
        self._probes[name] = probe

    def unregister(self, name: str) -> None:
        """Remove a probe if present."""
        self._probes.pop(name, None)

    @property
    def registered(self) -> Sequence[str]:
        """Names of all registered probes."""
        return tuple(self._probes)

    async def _run_one(self, name: str, probe: ProbeFn) -> ComponentHealth:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_s):
                result = await probe()
        except TimeoutError:
            elapsed = (time.perf_counter() - started) * 1000
            return ComponentHealth(
                component=name,
                status=HealthStatus.UNHEALTHY,
                detail=f"probe timed out after {self._timeout_s:.1f}s",
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            logger.warning(
                "health.probe_failed",
                extra={"component": name, "error": str(exc)},
                exc_info=True,
            )
            return ComponentHealth(
                component=name,
                status=HealthStatus.UNHEALTHY,
                detail=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed,
            )

        if result.latency_ms is None:
            elapsed = (time.perf_counter() - started) * 1000
            return ComponentHealth(
                component=result.component,
                status=result.status,
                detail=result.detail,
                latency_ms=elapsed,
                metadata=result.metadata,
            )
        return result

    async def check_all(self) -> SystemHealth:
        """Run every probe concurrently and aggregate.

        Concurrent rather than sequential because probes are almost entirely
        I/O wait; serialising a dozen of them would put the endpoint's latency
        above the timeout an orchestrator typically allows.
        """
        if not self._probes:
            return SystemHealth(
                status=HealthStatus.HEALTHY, components=(), checked_at=time.time()
            )

        results = await asyncio.gather(
            *(self._run_one(name, probe) for name, probe in self._probes.items())
        )

        if any(r.status is HealthStatus.UNHEALTHY for r in results):
            overall = HealthStatus.UNHEALTHY
        elif any(r.status is HealthStatus.DEGRADED for r in results):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return SystemHealth(status=overall, components=results, checked_at=time.time())


# ---------------------------------------------------------------------------
# Generic probe adapters
# ---------------------------------------------------------------------------


def make_callable_check(
    name: str,
    probe: Callable[[], Awaitable[Any]],
    *,
    degraded_above_ms: float | None = None,
) -> ProbeFn:
    """Wrap any awaitable into a probe.

    Success is "it returned without raising". Optionally flags ``DEGRADED`` when
    the call succeeds but is slow, which is how a saturated dependency usually
    presents before it starts failing outright.

    Args:
        name: Component name in the report.
        probe: Awaitable that raises on failure.
        degraded_above_ms: Latency above which to report ``DEGRADED``.
    """

    async def _check() -> ComponentHealth:
        started = time.perf_counter()
        try:
            await probe()
        except Exception as exc:
            return ComponentHealth(
                component=name,
                status=HealthStatus.UNHEALTHY,
                detail=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if degraded_above_ms is not None and elapsed_ms > degraded_above_ms:
            return ComponentHealth(
                component=name,
                status=HealthStatus.DEGRADED,
                detail=f"slow: {elapsed_ms:.0f}ms > {degraded_above_ms:.0f}ms threshold",
                latency_ms=elapsed_ms,
            )
        return ComponentHealth(
            component=name, status=HealthStatus.HEALTHY, latency_ms=elapsed_ms
        )

    return _check


# ---------------------------------------------------------------------------
# Built-in host probes
# ---------------------------------------------------------------------------


async def check_disk(
    path: Path | str = ".",
    *,
    degraded_below_pct: float = 15.0,
    unhealthy_below_pct: float = 5.0,
) -> ComponentHealth:
    """Report free disk space.

    Disk matters here beyond the usual reasons: model weights are tens of GB and
    rotating logs grow continuously, so a full disk takes down startup and
    logging together.
    """
    try:
        usage = await asyncio.to_thread(shutil.disk_usage, str(path))
    except Exception as exc:
        return ComponentHealth("disk", HealthStatus.UNHEALTHY, f"{type(exc).__name__}: {exc}")

    free_pct = (usage.free / usage.total) * 100 if usage.total else 0.0
    meta = {
        "free_gb": round(usage.free / 1024**3, 2),
        "total_gb": round(usage.total / 1024**3, 2),
        "free_pct": round(free_pct, 1),
    }

    if free_pct < unhealthy_below_pct:
        return ComponentHealth(
            "disk", HealthStatus.UNHEALTHY, f"only {free_pct:.1f}% free", metadata=meta
        )
    if free_pct < degraded_below_pct:
        return ComponentHealth(
            "disk", HealthStatus.DEGRADED, f"{free_pct:.1f}% free", metadata=meta
        )
    return ComponentHealth("disk", HealthStatus.HEALTHY, metadata=meta)


async def check_memory(
    *, degraded_above_pct: float = 85.0, unhealthy_above_pct: float = 95.0
) -> ComponentHealth:
    """Report host RAM pressure.

    Reports ``HEALTHY`` with a note when ``psutil`` is absent rather than
    failing: a missing optional metrics dependency is not an outage, and
    conflating the two produces false pages.
    """
    try:
        import psutil
    except ImportError:
        return ComponentHealth(
            "memory", HealthStatus.HEALTHY, "psutil not installed; RAM not monitored"
        )

    try:
        vm = await asyncio.to_thread(psutil.virtual_memory)
    except Exception as exc:
        return ComponentHealth("memory", HealthStatus.UNHEALTHY, f"{type(exc).__name__}: {exc}")

    meta = {
        "used_pct": round(vm.percent, 1),
        "available_gb": round(vm.available / 1024**3, 2),
        "total_gb": round(vm.total / 1024**3, 2),
    }
    if vm.percent > unhealthy_above_pct:
        return ComponentHealth(
            "memory", HealthStatus.UNHEALTHY, f"RAM at {vm.percent:.0f}%", metadata=meta
        )
    if vm.percent > degraded_above_pct:
        return ComponentHealth(
            "memory", HealthStatus.DEGRADED, f"RAM at {vm.percent:.0f}%", metadata=meta
        )
    return ComponentHealth("memory", HealthStatus.HEALTHY, metadata=meta)


async def check_cuda(
    *, degraded_vram_pct: float = 90.0, unhealthy_vram_pct: float = 97.0
) -> ComponentHealth:
    """Report CUDA availability and VRAM headroom.

    VRAM pressure is the failure mode that actually bites this platform: the TTS
    KV cache grows with concurrency, and crossing the limit surfaces as a hard
    CUDA OOM mid-conversation rather than as graceful degradation. Catching it
    at ``DEGRADED`` gives the scheduler a chance to stop admitting sessions
    first.
    """
    try:
        import torch
    except ImportError:
        return ComponentHealth("cuda", HealthStatus.UNHEALTHY, "torch is not installed")
    except Exception as exc:
        # torch raises OSError when its native libraries fail to load (missing
        # CUDA runtime, missing VC++ redistributable on Windows). Functionally
        # identical to "no GPU here", so report it as such rather than letting
        # a DLL error escape as an unhandled probe failure.
        return ComponentHealth(
            "cuda", HealthStatus.UNHEALTHY, f"torch failed to load: {type(exc).__name__}: {exc}"
        )

    if not torch.cuda.is_available():
        return ComponentHealth("cuda", HealthStatus.UNHEALTHY, "no CUDA device visible")

    try:
        device = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(device)
        used_pct = ((total_b - free_b) / total_b) * 100 if total_b else 0.0
        meta = {
            "device": torch.cuda.get_device_name(device),
            "device_index": device,
            "cuda_version": torch.version.cuda,
            "vram_used_pct": round(used_pct, 1),
            "vram_free_gb": round(free_b / 1024**3, 2),
            "vram_total_gb": round(total_b / 1024**3, 2),
        }
    except Exception as exc:
        return ComponentHealth("cuda", HealthStatus.UNHEALTHY, f"{type(exc).__name__}: {exc}")

    if used_pct > unhealthy_vram_pct:
        return ComponentHealth(
            "cuda", HealthStatus.UNHEALTHY, f"VRAM at {used_pct:.0f}%", metadata=meta
        )
    if used_pct > degraded_vram_pct:
        return ComponentHealth(
            "cuda", HealthStatus.DEGRADED, f"VRAM at {used_pct:.0f}%", metadata=meta
        )
    return ComponentHealth("cuda", HealthStatus.HEALTHY, metadata=meta)
