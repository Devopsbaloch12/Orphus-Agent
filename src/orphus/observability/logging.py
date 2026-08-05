"""Structured JSON logging with per-service files, rotation, and redaction.

Design notes:

* **Context propagation via ``contextvars``.** In an async pipeline a single
  request touches VAD, ASR, LLM, and TTS across many tasks. Threading a
  ``session_id`` through every call signature would be invasive and easy to
  forget, so it is bound once per session and every subsequent log record picks
  it up automatically -- including records emitted by third-party libraries.
* **Redaction at the formatter, not the call site.** Relying on developers to
  remember not to log a secret fails eventually. Filtering in the one place all
  records pass through means a leak requires defeating the formatter, not just
  forgetting a rule.
* **stdlib ``logging`` rather than a logging framework.** The requirements here
  are narrow (JSON, rotation, per-service files, context binding) and the stdlib
  covers all of them. Third-party GPU libraries already log through stdlib, so
  their output lands in our JSON stream for free.

Usage::

    from orphus.observability.logging import get_logger, bind_session

    logger = get_logger(__name__)
    with bind_session(session_id):
        logger.info("asr.partial", extra={"chars": len(text)})
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import logging.handlers
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Final

__all__ = [
    "JsonFormatter",
    "bind_session",
    "bind_turn",
    "configure_logging",
    "current_session_id",
    "get_logger",
]

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

_session_id_var: ContextVar[str | None] = ContextVar("orphus_session_id", default=None)
_turn_id_var: ContextVar[str | None] = ContextVar("orphus_turn_id", default=None)

_DEFAULT_REDACT_KEYS: Final[frozenset[str]] = frozenset(
    {"api_key", "authorization", "grok_api_key", "password", "token", "secret"}
)

_REDACTED: Final[str] = "***redacted***"

# Attributes the stdlib puts on every LogRecord. Anything outside this set was
# supplied by the caller via `extra=` and belongs in the JSON payload.
_STANDARD_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "stacklevel", "thread", "threadName",
        "taskName",
    }
)


@contextlib.contextmanager
def bind_session(session_id: str) -> Iterator[None]:
    """Attach ``session_id`` to every log record emitted inside the block."""
    token = _session_id_var.set(session_id)
    try:
        yield
    finally:
        _session_id_var.reset(token)


@contextlib.contextmanager
def bind_turn(turn_id: str) -> Iterator[None]:
    """Attach ``turn_id`` to every log record emitted inside the block."""
    token = _turn_id_var.set(turn_id)
    try:
        yield
    finally:
        _turn_id_var.reset(token)


def current_session_id() -> str | None:
    """Return the session bound to the current context, if any."""
    return _session_id_var.get()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON with secrets stripped."""

    def __init__(self, *, redact_keys: Sequence[str] | None = None, service: str | None = None):
        """Initialise the formatter.

        Args:
            redact_keys: Field names whose values are replaced with a redaction
                marker. Matching is case-insensitive and substring-based, so
                ``api_key`` also covers ``grok_api_key`` and ``x_api_key``.
            service: Logical service name stamped onto every record.
        """
        super().__init__()
        keys = _DEFAULT_REDACT_KEYS if redact_keys is None else frozenset(redact_keys)
        self._redact_keys = frozenset(key.lower() for key in keys)
        self._service = service

    def _should_redact(self, key: str) -> bool:
        lowered = key.lower()
        return any(secret in lowered for secret in self._redact_keys)

    def _sanitize(self, value: Any, depth: int = 0) -> Any:
        """Recursively redact secrets and coerce values to JSON-safe types.

        Depth is capped because log payloads occasionally carry deeply nested or
        self-referential objects, and a logging call must never be the thing
        that blows the stack.
        """
        if depth > 6:
            return "<max-depth>"
        if isinstance(value, Mapping):
            return {
                key: (
                    _REDACTED
                    if self._should_redact(str(key))
                    else self._sanitize(val, depth + 1)
                )
                for key, val in value.items()
            }
        if isinstance(value, list | tuple | set):
            return [self._sanitize(item, depth + 1) for item in value]
        if isinstance(value, str | int | float | bool | type(None)):
            return value
        return repr(value)

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as a JSON object."""
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self._service:
            payload["service"] = self._service

        session_id = _session_id_var.get()
        if session_id:
            payload["session_id"] = session_id
        turn_id = _turn_id_var.get()
        if turn_id:
            payload["turn_id"] = turn_id

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = (
                _REDACTED if self._should_redact(key) else self._sanitize(value)
            )

        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            }

        payload["source"] = f"{record.module}:{record.lineno}"

        # default=str keeps a stray non-serialisable object from raising inside
        # the logging call and swallowing the very error being reported.
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    _COLORS: Final[dict[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    _RESET: Final[str] = "\033[0m"

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Render one record as an aligned, optionally coloured line."""
        stamp = dt.datetime.fromtimestamp(record.created, tz=dt.UTC).strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname
        if self._use_color:
            level = f"{self._COLORS.get(level, '')}{level:<8}{self._RESET}"
        else:
            level = f"{level:<8}"

        session_id = _session_id_var.get()
        prefix = f"[{session_id[-8:]}] " if session_id else ""

        line = f"{stamp} {level} {record.name:<34} {prefix}{record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_configured: bool = False


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "json",
    directory: Path | str | None = None,
    per_service_files: bool = True,
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 10,
    console: bool = True,
    redact_keys: Sequence[str] | None = None,
    services: Sequence[str] = ("api", "asr", "tts", "vad", "scheduler", "worker"),
) -> None:
    """Install handlers on the root logger. Safe to call more than once.

    Args:
        level: Root log level name.
        fmt: ``"json"`` or ``"console"``.
        directory: Log directory. ``None`` disables file logging entirely,
            which is what you want when running under a supervisor that
            captures stdout.
        per_service_files: Give each service its own rotating file so a chatty
            ASR worker cannot bury an API error.
        max_bytes: Rotation threshold per file.
        backup_count: Rotated files retained per service.
        console: Also emit to stderr.
        redact_keys: Field names to redact; ``None`` uses the built-in set.
        services: Logical services that get their own file. A service's records
            are matched by the ``orphus.<service>`` logger prefix.
    """
    global _configured

    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    def _build_formatter(service: str | None) -> logging.Formatter:
        if fmt == "console":
            return ConsoleFormatter()
        return JsonFormatter(redact_keys=redact_keys, service=service)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_build_formatter(None))
        root.addHandler(stream)

    if directory is not None:
        log_dir = Path(directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        combined = logging.handlers.RotatingFileHandler(
            log_dir / "orphus.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        combined.setFormatter(_build_formatter(None))
        root.addHandler(combined)

        errors = logging.handlers.RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        errors.setLevel(logging.ERROR)
        errors.setFormatter(_build_formatter(None))
        root.addHandler(errors)

        if per_service_files:
            for service in services:
                handler = logging.handlers.RotatingFileHandler(
                    log_dir / f"{service}.log",
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                handler.setFormatter(_build_formatter(service))
                handler.addFilter(_ServiceFilter(service))
                root.addHandler(handler)

    # These libraries are extremely chatty at INFO and drown real signal.
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "vllm", "nemo_logger"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


class _ServiceFilter(logging.Filter):
    """Admit only records from ``orphus.<service>`` and its children."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._prefix = f"orphus.{service}"

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether the record belongs to this service."""
        return record.name == self._prefix or record.name.startswith(f"{self._prefix}.")


def get_logger(name: str) -> logging.Logger:
    """Return a logger, installing a sane default configuration on first use.

    The lazy default matters: modules are imported (and may log) before the
    application has read its configuration, and during tests
    :func:`configure_logging` is often never called at all. Without this, those
    records would hit the stdlib's "no handlers" fallback and vanish.

    Args:
        name: Usually ``__name__``.

    Returns:
        A standard library logger. Pass structured fields via ``extra=``.
    """
    if not _configured:
        configure_logging(level="INFO", fmt="console", directory=None, console=True)
    return logging.getLogger(name)
