"""FastAPI application factory and versioned session API."""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
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
from pydantic import BaseModel, Field

from orphus.audio.convert import encode_pcm
from orphus.audio.pipeline import InboundAudioPipeline
from orphus.config import Settings, load_settings
from orphus.conversation.errors import SessionLimitError, SessionNotFoundError
from orphus.conversation.manager import SessionManager
from orphus.domain.types import AudioEncoding
from orphus.middleware import ApiKeyAuthenticator, RateLimiter, RequestContextMiddleware
from orphus.observability.health import HealthRegistry, check_disk, check_memory
from orphus.observability.logging import get_logger
from orphus.observability.metrics import CONTENT_TYPE_LATEST, get_metrics
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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal pipeline_factory
        if runtime is not None and config.environment == "production":
            await runtime.load()
            pipeline_factory = runtime.pipeline
        reaper = asyncio.create_task(_reaper())
        try:
            yield
        finally:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
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
        supplied = websocket.headers.get("x-api-key")
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
            async for chunk in pipeline.output():
                await websocket.send_bytes(encode_pcm(chunk.samples, AudioEncoding.PCM_S16LE))

        sender = asyncio.create_task(send_audio())
        try:
            while True:
                payload = await websocket.receive_bytes()
                if len(payload) > config.security.max_audio_upload_bytes:
                    await websocket.close(code=1009, reason="audio frame too large")
                    return
                for frame in inbound.push_bytes(payload):
                    await pipeline.push_audio(frame)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await pipeline.aclose()

    return app
