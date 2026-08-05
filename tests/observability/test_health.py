"""Tests for the health probe registry.

The invariant under test throughout: a probe never takes down the endpoint. If
the health check could crash or hang, the one tool you use to diagnose an
outage would be unavailable precisely during an outage.
"""

from __future__ import annotations

import asyncio

import pytest

from orphus.observability.health import (
    ComponentHealth,
    HealthRegistry,
    HealthStatus,
    check_cuda,
    check_disk,
    check_memory,
    make_callable_check,
)


async def _healthy() -> ComponentHealth:
    return ComponentHealth("ok", HealthStatus.HEALTHY)


async def _degraded() -> ComponentHealth:
    return ComponentHealth("warm", HealthStatus.DEGRADED, "running hot")


async def _unhealthy() -> ComponentHealth:
    return ComponentHealth("down", HealthStatus.UNHEALTHY, "connection refused")


class TestAggregation:
    async def test_empty_registry_is_healthy(self) -> None:
        assert (await HealthRegistry().check_all()).status is HealthStatus.HEALTHY

    async def test_all_healthy(self) -> None:
        reg = HealthRegistry()
        reg.register("a", _healthy)
        reg.register("b", _healthy)
        assert (await reg.check_all()).status is HealthStatus.HEALTHY

    async def test_one_degraded_degrades_the_whole(self) -> None:
        reg = HealthRegistry()
        reg.register("a", _healthy)
        reg.register("b", _degraded)
        result = await reg.check_all()
        assert result.status is HealthStatus.DEGRADED
        # Degraded must still count as serving, or the platform's whole
        # graceful-degradation design would take pods out of rotation.
        assert result.healthy

    async def test_unhealthy_dominates_degraded(self) -> None:
        reg = HealthRegistry()
        reg.register("a", _degraded)
        reg.register("b", _unhealthy)
        result = await reg.check_all()
        assert result.status is HealthStatus.UNHEALTHY
        assert not result.healthy

    async def test_every_component_appears_in_the_report(self) -> None:
        reg = HealthRegistry()
        for name in ("a", "b", "c"):
            reg.register(name, _healthy)
        result = await reg.check_all()
        assert {c.component for c in result.components} == {"ok"}
        assert len(result.components) == 3


class TestFailureContainment:
    async def test_raising_probe_is_reported_not_propagated(self) -> None:
        async def boom() -> ComponentHealth:
            raise ConnectionError("redis is gone")

        reg = HealthRegistry()
        reg.register("redis", boom)
        result = await reg.check_all()

        component = result.components[0]
        assert component.status is HealthStatus.UNHEALTHY
        assert "ConnectionError" in (component.detail or "")
        assert "redis is gone" in (component.detail or "")

    async def test_hanging_probe_is_cut_off_by_timeout(self) -> None:
        async def hangs() -> ComponentHealth:
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

        reg = HealthRegistry(timeout_s=0.05)
        reg.register("wedged", hangs)

        result = await asyncio.wait_for(reg.check_all(), timeout=2.0)
        assert result.components[0].status is HealthStatus.UNHEALTHY
        assert "timed out" in (result.components[0].detail or "")

    async def test_one_bad_probe_does_not_mask_the_others(self) -> None:
        async def boom() -> ComponentHealth:
            raise RuntimeError("x")

        reg = HealthRegistry()
        reg.register("good", _healthy)
        reg.register("bad", boom)
        result = await reg.check_all()
        assert len(result.components) == 2

    async def test_probes_run_concurrently(self) -> None:
        """Serialised probes would blow past an orchestrator's timeout budget."""

        async def slow() -> ComponentHealth:
            await asyncio.sleep(0.1)
            return ComponentHealth("slow", HealthStatus.HEALTHY)

        reg = HealthRegistry(timeout_s=2.0)
        for i in range(8):
            reg.register(f"slow{i}", slow)

        loop = asyncio.get_running_loop()
        started = loop.time()
        await reg.check_all()
        elapsed = loop.time() - started

        # 8 x 100ms serially would be 800ms; concurrently it is ~100ms.
        assert elapsed < 0.4, f"probes appear serialised: {elapsed:.3f}s"


class TestRegistryManagement:
    def test_register_and_list(self) -> None:
        reg = HealthRegistry()
        reg.register("a", _healthy)
        reg.register("b", _healthy)
        assert set(reg.registered) == {"a", "b"}

    def test_re_registering_replaces(self) -> None:
        reg = HealthRegistry()
        reg.register("a", _healthy)
        reg.register("a", _unhealthy)
        assert len(reg.registered) == 1

    def test_unregister_is_forgiving(self) -> None:
        reg = HealthRegistry()
        reg.unregister("never-registered")
        assert reg.registered == ()


class TestCallableAdapter:
    async def test_success(self) -> None:
        async def ping() -> str:
            return "pong"

        result = await make_callable_check("redis", ping)()
        assert result.status is HealthStatus.HEALTHY
        assert result.latency_ms is not None

    async def test_failure_captures_the_diagnosis(self) -> None:
        async def ping() -> None:
            raise TimeoutError("no route to host")

        result = await make_callable_check("redis", ping)()
        assert result.status is HealthStatus.UNHEALTHY
        assert "no route to host" in (result.detail or "")

    async def test_slow_success_is_degraded_not_failed(self) -> None:
        async def ping() -> None:
            await asyncio.sleep(0.05)

        result = await make_callable_check("pg", ping, degraded_above_ms=10)()
        assert result.status is HealthStatus.DEGRADED
        assert "slow" in (result.detail or "")

    async def test_fast_success_is_healthy(self) -> None:
        async def ping() -> None:
            return None

        result = await make_callable_check("pg", ping, degraded_above_ms=1000)()
        assert result.status is HealthStatus.HEALTHY


class TestHostProbes:
    async def test_disk_probe_reports_capacity(self) -> None:
        result = await check_disk(".")
        assert result.component == "disk"
        assert result.metadata["total_gb"] > 0
        assert 0 <= result.metadata["free_pct"] <= 100

    async def test_disk_thresholds_trip(self) -> None:
        # Any real filesystem has <200% free, so this forces the unhealthy path.
        result = await check_disk(".", degraded_below_pct=300, unhealthy_below_pct=200)
        assert result.status is HealthStatus.UNHEALTHY

    async def test_disk_probe_on_missing_path_reports_not_raises(self) -> None:
        result = await check_disk("/definitely/not/a/real/path/xyzzy")
        assert result.status is HealthStatus.UNHEALTHY

    async def test_memory_probe_never_raises(self) -> None:
        result = await check_memory()
        assert result.component == "memory"
        assert isinstance(result.status, HealthStatus)

    async def test_cuda_probe_never_raises_without_a_gpu(self) -> None:
        """A broken or absent torch must report, not explode."""
        result = await check_cuda()
        assert result.component == "cuda"
        assert isinstance(result.status, HealthStatus)


class TestSerialisation:
    async def test_report_is_json_safe(self) -> None:
        import json

        reg = HealthRegistry()
        reg.register("a", _healthy)
        reg.register("b", _degraded)
        payload = (await reg.check_all()).to_dict()

        json.dumps(payload)  # must not raise
        assert payload["status"] in {"healthy", "degraded", "unhealthy"}
        assert len(payload["components"]) == 2

    def test_detail_is_omitted_when_absent(self) -> None:
        assert "detail" not in ComponentHealth("x", HealthStatus.HEALTHY).to_dict()

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (HealthStatus.HEALTHY, True),
            (HealthStatus.DEGRADED, True),
            (HealthStatus.UNHEALTHY, False),
        ],
    )
    def test_healthy_property(self, status: HealthStatus, expected: bool) -> None:
        assert ComponentHealth("x", status).healthy is expected
