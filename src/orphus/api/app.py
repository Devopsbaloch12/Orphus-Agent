"""FastAPI application factory and versioned session API."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# repo_root/frontend/out, the static export of the Next.js browser test
# client. Not present in dev/test checkouts that never ran `npm run build`.
_FRONTEND_EXPORT = Path(__file__).resolve().parents[3] / "frontend" / "out"

from orphus.audio.convert import encode_pcm
from orphus.audio.pipeline import InboundAudioPipeline
from orphus.config import Settings, load_settings
from orphus.conversation.errors import SessionLimitError, SessionNotFoundError
from orphus.conversation.manager import SessionManager
from orphus.domain.types import AudioEncoding
from orphus.middleware import ApiKeyAuthenticator, RateLimiter, RequestContextMiddleware
from orphus.observability.health import HealthRegistry, check_disk, check_memory
from orphus.observability.logging import get_logger
from orphus.observability.metrics import CONTENT_TYPE_LATEST, GpuMetricsCollector, get_metrics
from orphus.runtime import ModelRuntime
from orphus.streaming import VoicePipeline


logger = get_logger(__name__)


class CreateSessionRequest(BaseModel):
    voice: str | None = Field(default=None, max_length=64)
    language: str | None = Field(default=None, max_length=32)
    metadata: dict[str, str] = Field(default_factory=dict)


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_factory: Callable[[Any], VoicePipeline] | None = None,
) -> FastAPI:
    config = settings or load_settings()
    runtime = ModelRuntime(config) if pipeline_factory is None else None
    manager = SessionManager(
        max_concurrent=config.session.max_concurrent,
        idle_timeout_s=config.session.idle_timeout_s,
        max_duration_s=config.session.max_duration_s,
        history_max_turns=config.session.history_max_turns,
        history_max_chars=config.session.history_max_chars,
        system_prompt=config.llm.system_prompt,
    )
    health = HealthRegistry()
    health.register("disk", check_disk)
    health.register("memory", check_memory)
    auth = ApiKeyAuthenticator(
        [secret.get_secret_value() for secret in config.security.api_keys]
    )
    dependencies: list[Any] = [Depends(auth)]
    if config.security.rate_limit_enabled:
        dependencies.append(
            Depends(
                RateLimiter(
                    max_requests=config.security.rate_limit_requests,
                    window_s=config.security.rate_limit_window_s,
                )
            )
        )

    async def _reaper() -> None:
        """Expire idle and over-long sessions.

        SessionManager.reap() is the only thing that frees a seat when a caller
        vanishes without a DELETE -- a dropped carrier leg, a crashed client, a
        socket that just goes away. Nothing drove it, so sessions accumulated
        for the life of the process: after max_concurrent calls every new
        session got 503 forever, with the fix a restart away. Sweep on a
        fraction of the idle timeout so a freed seat is reusable promptly.
        """
        interval = max(5.0, min(config.session.idle_timeout_s / 4, 60.0))
        while True:
            await asyncio.sleep(interval)
            try:
                reaped = await manager.reap()
                if reaped:
                    logger.info("session.reaped", extra={"count": reaped})
            except asyncio.CancelledError:
                raise
            except Exception:  # a sweep failure must not kill the sweeper
                logger.exception("session.reap_failed")

    async def _gpu_poller(collector: GpuMetricsCollector) -> None:
        """Sample NVML on a fixed interval so GPU gauges are never stale-zero.

        Without this loop the Prometheus gauges exist but nothing ever calls
        ``poll()`` -- ``/metrics`` reports GPU utilisation/VRAM/temperature as
        permanently absent regardless of load.
        """
        interval = config.monitoring.gpu_poll_interval_s
        while True:
            await asyncio.sleep(interval)
            try:
                collector.poll()
            except asyncio.CancelledError:
                raise
            except Exception:  # a poll failure must not kill the poller
                logger.exception("gpu.metrics.poll_failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal pipeline_factory
        if runtime is not None and config.environment == "production":
            await runtime.load()
            pipeline_factory = runtime.pipeline
        reaper = asyncio.create_task(_reaper())
        gpu_collector: GpuMetricsCollector | None = None
        gpu_task: asyncio.Task[None] | None = None
        if config.monitoring.gpu_metrics_enabled:
            gpu_collector = GpuMetricsCollector(get_metrics()).start()
            if gpu_collector.available:
                gpu_task = asyncio.create_task(_gpu_poller(gpu_collector))
        try:
            yield
        finally:
            reaper.cancel()
            tasks = [reaper]
            if gpu_task is not None:
                gpu_task.cancel()
                tasks.append(gpu_task)
            await asyncio.gather(*tasks, return_exceptions=True)
            if gpu_collector is not None:
                gpu_collector.shutdown()
            await manager.aclose()
            if runtime is not None:
                await runtime.aclose()

    app = FastAPI(title="Orphus Voice AI", version="0.1.0", lifespan=lifespan)
    app.state.settings = config
    app.state.session_manager = manager
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", tags=["operations"])
    async def health_check(response: Response) -> dict[str, Any]:
        report = await health.check_all()
        if not report.healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return report.to_dict()

    @app.get(config.monitoring.metrics_path, tags=["operations"])
    async def metrics() -> Response:
        return Response(get_metrics().render(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/sessions", status_code=201, dependencies=dependencies)
    async def create_session(body: CreateSessionRequest) -> dict[str, Any]:
        metadata = dict(body.metadata)
        if body.voice:
            metadata["voice"] = body.voice
        if body.language:
            metadata["language"] = body.language
        try:
            session = await manager.create(metadata=metadata)
        except SessionLimitError as exc:
            raise HTTPException(
                status_code=503, detail=str(exc), headers={"Retry-After": "1"}
            ) from exc
        return {"session_id": session.session_id, "state": session.state.value}

    @app.get("/v1/sessions/{session_id}", dependencies=dependencies)
    async def session_status(session_id: str) -> dict[str, Any]:
        try:
            session = await manager.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "age_s": session.age_s,
            "idle_s": session.idle_s,
            "turn_count": session.turn_count,
        }

    @app.delete("/v1/sessions/{session_id}", status_code=204, dependencies=dependencies)
    async def close_session(session_id: str) -> Response:
        try:
            await manager.close(session_id, reason="api_request")
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.websocket("/v1/ws/{session_id}")
    async def voice_socket(websocket: WebSocket, session_id: str) -> None:
        configured_keys = [key.get_secret_value() for key in config.security.api_keys]
        # Browsers cannot set custom headers on a WebSocket handshake, so the
        # browser test client authenticates via query param instead. Server
        # clients (Asterisk/ViciDial, the Python reference client) keep using
        # the header.
        supplied = websocket.headers.get("x-api-key") or websocket.query_params.get(
            "api_key"
        )
        if configured_keys and not (
            supplied and any(hmac.compare_digest(supplied, key) for key in configured_keys)
        ):
            await websocket.close(code=4401, reason="invalid API key")
            return
        try:
            session = await manager.get(session_id)
        except SessionNotFoundError:
            await websocket.close(code=4404, reason="session not found")
            return
        if pipeline_factory is None:
            await websocket.close(code=1013, reason="voice models are not available")
            return
        pipeline = pipeline_factory(session)
        inbound = InboundAudioPipeline(source_rate=16_000)
        await websocket.accept()

        async def send_audio() -> None:
            chunks = frames = 0
            try:
                async for chunk in pipeline.output():
                    payload = encode_pcm(chunk.samples, AudioEncoding.PCM_S16LE)
                    await websocket.send_bytes(payload)
                    chunks += 1
                    frames += len(payload)
            except Exception:
                logger.exception(
                    f"voice_socket.send_audio_failed session={session_id} "
                    f"chunks_sent={chunks} bytes_sent={frames}"
                )
                raise
            finally:
                logger.info(
                    f"voice_socket.send_audio_done session={session_id} "
                    f"chunks_sent={chunks} bytes_sent={frames}"
                )

        sender = asyncio.create_task(send_audio())
        frames_received = 0
        try:
            while True:
                payload = await websocket.receive_bytes()
                frames_received += 1
                if len(payload) > config.security.max_audio_upload_bytes:
                    logger.warning(
                        f"voice_socket.frame_too_large session={session_id} "
                        f"bytes={len(payload)} limit={config.security.max_audio_upload_bytes} "
                        f"frames_received={frames_received}"
                    )
                    await websocket.close(code=1009, reason="audio frame too large")
                    return
                for frame in inbound.push_bytes(payload):
                    await pipeline.push_audio(frame)
        except WebSocketDisconnect as exc:
            logger.info(
                f"voice_socket.client_disconnected session={session_id} "
                f"code={exc.code} reason={exc.reason!r} frames_received={frames_received}"
            )
        except Exception:
            # Anything else here (a VAD/ASR error, a session-state error, a
            # bug) previously propagated uncaught: FastAPI closes the socket
            # for us, but silently -- the call just "stops" with nothing in
            # the logs to say why. Log it explicitly before it closes.
            logger.exception(
                f"voice_socket.receive_loop_failed session={session_id} "
                f"frames_received={frames_received}"
            )
            raise
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await pipeline.aclose()
            # Release the concurrency slot immediately, not just on an
            # explicit DELETE the client may never get to send (crash, tab
            # close, network drop). Left to the idle reaper, a lingering
            # "zombie" session -- pipeline resources already freed, slot
            # still held -- silently erodes capacity for up to
            # session.idle_timeout_s and surfaces as unrelated 503s
            # elsewhere, not as an error here.
            try:
                await manager.close(session_id, reason="websocket_disconnect")
            except SessionNotFoundError:
                pass  # already closed by a concurrent explicit DELETE

    # Registered last: a catch-all, so it never shadows the API routes above.
    if _FRONTEND_EXPORT.is_dir():
        app.mount(
            "/", StaticFiles(directory=_FRONTEND_EXPORT, html=True), name="frontend"
        )
    else:
        logger.warning("frontend.export_missing", extra={"path": str(_FRONTEND_EXPORT)})

    return app
