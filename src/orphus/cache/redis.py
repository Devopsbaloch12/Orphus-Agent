"""Namespaced async Redis access for session hot state."""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from orphus.config.settings import RedisSettings


class RedisCache:
    def __init__(self, client: Redis, prefix: str) -> None:
        self._client = client
        self._prefix = prefix

    @classmethod
    def connect(cls, settings: RedisSettings) -> RedisCache:
        client = Redis.from_url(
            settings.url,
            max_connections=settings.max_connections,
            socket_timeout=settings.socket_timeout_s,
            decode_responses=True,
        )
        return cls(client, settings.key_prefix)

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get_json(self, key: str) -> dict[str, Any] | None:
        value = await self._client.get(self._key(key))
        return None if value is None else dict(json.loads(value))

    async def set_json(self, key: str, value: dict[str, Any], *, ttl_s: int) -> None:
        await self._client.set(self._key(key), json.dumps(value), ex=ttl_s)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def health(self) -> None:
        if not await self._client.ping():
            raise RuntimeError("Redis ping failed")

    async def aclose(self) -> None:
        await self._client.aclose()

