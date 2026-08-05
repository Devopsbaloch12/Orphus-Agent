"""Streaming Grok adapter for the OpenAI-compatible xAI API."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence

import httpx

from orphus.config.settings import LLMSettings
from orphus.domain.types import LLMUsage, Message, SessionId, TextDelta
from orphus.observability.health import ComponentHealth, HealthStatus

__all__ = ["GrokProvider"]


class GrokProvider:
    """One pooled, retrying client with a small failure circuit breaker."""

    def __init__(self, settings: LLMSettings, *, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            timeout=httpx.Timeout(settings.timeout_s, connect=settings.connect_timeout_s),
        )
        self._owns_client = client is None
        self._failures = 0
        self._opened_at: float | None = None
        self._usage: dict[SessionId, LLMUsage] = {}

    @property
    def name(self) -> str:
        return "grok"

    @property
    def model(self) -> str:
        return self._settings.model

    def _ensure_available(self) -> None:
        if self._opened_at is None:
            return
        if time.monotonic() - self._opened_at >= self._settings.circuit_breaker_reset_s:
            self._opened_at = None
            self._failures = 0
            return
        raise RuntimeError("Grok circuit breaker is open")

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._settings.circuit_breaker_threshold:
            self._opened_at = time.monotonic()

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        session_id: SessionId,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[TextDelta]:
        return self._stream_chat(messages, session_id, max_tokens, temperature)

    async def _stream_chat(
        self,
        messages: Sequence[Message],
        session_id: SessionId,
        max_tokens: int | None,
        temperature: float | None,
    ) -> AsyncIterator[TextDelta]:
        self._ensure_available()
        payload = {
            "model": self.model,
            "messages": [message.to_wire() for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens or self._settings.max_tokens,
            "temperature": (
                self._settings.temperature if temperature is None else temperature
            ),
        }
        for attempt in range(self._settings.max_retries + 1):
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            self._failures = 0
                            yield TextDelta("", is_final=True)
                            return
                        event = json.loads(data)
                        usage = event.get("usage")
                        if usage:
                            self._usage[session_id] = LLMUsage(
                                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                                completion_tokens=int(usage.get("completion_tokens", 0)),
                            )
                        choices = event.get("choices") or []
                        text = choices[0].get("delta", {}).get("content") if choices else None
                        if text:
                            yield TextDelta(str(text))
                return
            except (httpx.HTTPError, json.JSONDecodeError):
                self._record_failure()
                if attempt >= self._settings.max_retries:
                    raise
                await asyncio.sleep(self._settings.retry_backoff_s * (2**attempt))

    async def last_usage(self, session_id: SessionId) -> LLMUsage | None:
        return self._usage.get(session_id)

    async def health(self) -> ComponentHealth:
        try:
            self._ensure_available()
            response = await self._client.get("/models")
            response.raise_for_status()
        except Exception as exc:
            return ComponentHealth("grok", HealthStatus.UNHEALTHY, str(exc))
        return ComponentHealth("grok", HealthStatus.HEALTHY)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
