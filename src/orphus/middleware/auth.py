"""API key authentication hook.

Deliberately minimal: a single shared-secret scheme via ``Authorization:
Bearer <key>`` or ``X-API-Key``, checked against ``security.api_keys``. This is
a *hook*, not a full auth system -- the spec asks for "authentication hooks",
and a voice platform's real identity boundary is usually upstream (a gateway,
an SSO proxy). Building more here would be scope creep and a maintenance
burden nobody asked for.

When ``security.api_keys`` is empty the hook is a deliberate no-op: local
development and CI must not require a key, but :class:`~orphus.config.Settings`
already refuses to validate a ``production`` environment without at least one
key configured, so this can never silently disable auth in prod.

Key comparison uses ``hmac.compare_digest`` -- a naive ``==`` leaks key length
and prefix through response-time differences.
"""

from __future__ import annotations

import hmac
from collections.abc import Sequence

from fastapi import HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param

from orphus.observability.logging import get_logger

__all__ = ["ApiKeyAuthenticator"]

logger = get_logger(__name__)

_API_KEY_HEADER = "x-api-key"


class ApiKeyAuthenticator:
    """Validates inbound requests against a configured set of API keys."""

    def __init__(self, api_keys: Sequence[str]) -> None:
        """Initialise the authenticator.

        Args:
            api_keys: Accepted keys in plaintext. Callers should pass
                ``[k.get_secret_value() for k in settings.security.api_keys]``
                so the ``SecretStr`` wrapper is unwrapped in exactly one place.
        """
        self._keys = tuple(api_keys)

    @property
    def enabled(self) -> bool:
        """Whether any key is configured. When ``False``, every request passes."""
        return bool(self._keys)

    def _matches_any(self, candidate: str) -> bool:
        # compare_digest over every key, not short-circuited on the first
        # mismatch, so response time does not reveal which key almost matched.
        return any(hmac.compare_digest(candidate, key) for key in self._keys)

    def _extract_key(self, request: Request) -> str | None:
        header = request.headers.get(_API_KEY_HEADER)
        if header:
            return header

        authorization = request.headers.get("authorization")
        if authorization:
            scheme, credential = get_authorization_scheme_param(authorization)
            if scheme.lower() == "bearer" and credential:
                return credential

        return None

    async def __call__(self, request: Request) -> None:
        """FastAPI dependency: raises 401 when the request is unauthenticated.

        Registered as a router-level dependency rather than global middleware so
        that health and metrics endpoints -- which orchestrators and Prometheus
        scrape without credentials -- can opt out individually.
        """
        if not self.enabled:
            return

        key = self._extract_key(request)
        if key is None or not self._matches_any(key):
            logger.warning(
                "auth.rejected",
                extra={"path": request.url.path, "has_key": key is not None},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
