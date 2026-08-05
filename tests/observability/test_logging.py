"""Tests for structured logging, context binding, and secret redaction.

The redaction tests carry the most weight here. Redaction is the last line of
defence between an API key and a log aggregator, and a regression in it is both
silent and expensive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from orphus.observability.logging import (
    JsonFormatter,
    bind_session,
    bind_turn,
    configure_logging,
    current_session_id,
    get_logger,
)

SECRET = "sk-live-DO-NOT-LEAK-12345"


def _render(
    formatter: JsonFormatter,
    message: str = "event",
    level: int = logging.INFO,
    **extra: Any,
) -> dict[str, Any]:
    """Format one record through the real formatter and parse the result."""
    record = logging.LogRecord(
        name="orphus.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    parsed: dict[str, Any] = json.loads(formatter.format(record))
    return parsed


class TestJsonShape:
    def test_emits_required_fields(self) -> None:
        out = _render(JsonFormatter())
        assert out["level"] == "INFO"
        assert out["logger"] == "orphus.test"
        assert out["message"] == "event"
        assert out["timestamp"].endswith("+00:00")  # UTC, unambiguous across regions
        assert "source" in out

    def test_extra_fields_are_promoted_to_top_level(self) -> None:
        out = _render(JsonFormatter(), latency_ms=42.5, session_count=3)
        assert out["latency_ms"] == 42.5
        assert out["session_count"] == 3

    def test_output_is_a_single_line(self) -> None:
        record = logging.LogRecord(
            "orphus.test", logging.INFO, __file__, 1, "a\nb\nc", (), None
        )
        assert "\n" not in JsonFormatter().format(record)

    def test_service_is_stamped_when_configured(self) -> None:
        assert _render(JsonFormatter(service="asr"))["service"] == "asr"

    def test_exception_is_captured_with_traceback(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "orphus.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
            )
        out = json.loads(JsonFormatter().format(record))
        assert out["exception"]["type"] == "ValueError"
        assert out["exception"]["message"] == "boom"
        assert "ValueError: boom" in out["exception"]["traceback"]

    def test_unserialisable_value_does_not_raise(self) -> None:
        class Opaque:
            def __repr__(self) -> str:
                return "<opaque>"

        # A logging call must never be the thing that raises, especially when it
        # is reporting the error that produced the odd object in the first place.
        assert _render(JsonFormatter(), thing=Opaque())["thing"] == "<opaque>"

    def test_self_referential_payload_does_not_recurse_forever(self) -> None:
        payload: dict[str, Any] = {"name": "loop"}
        payload["self"] = payload
        assert _render(JsonFormatter(), data=payload)["data"]["name"] == "loop"


class TestRedaction:
    @pytest.mark.parametrize(
        "field",
        ["api_key", "grok_api_key", "authorization", "password", "token", "secret"],
    )
    def test_known_secret_fields_are_redacted(self, field: str) -> None:
        out = _render(JsonFormatter(), **{field: SECRET})
        assert out[field] == "***redacted***"
        assert SECRET not in json.dumps(out)

    def test_matching_is_case_insensitive(self) -> None:
        assert _render(JsonFormatter(), API_KEY=SECRET)["API_KEY"] == "***redacted***"

    def test_matching_is_substring_based(self) -> None:
        # Covers x_api_key, grok_api_key, refresh_token, and future variants
        # without needing an exhaustive list.
        out = _render(JsonFormatter(), x_api_key=SECRET, refresh_token=SECRET)
        assert SECRET not in json.dumps(out)

    def test_nested_secrets_are_redacted(self) -> None:
        out = _render(
            JsonFormatter(),
            request={"headers": {"authorization": f"Bearer {SECRET}"}, "path": "/v1/chat"},
        )
        assert out["request"]["headers"]["authorization"] == "***redacted***"
        assert out["request"]["path"] == "/v1/chat"  # non-secrets survive
        assert SECRET not in json.dumps(out)

    def test_secrets_inside_lists_are_redacted(self) -> None:
        out = _render(JsonFormatter(), items=[{"token": SECRET}, {"ok": "visible"}])
        assert SECRET not in json.dumps(out)
        assert out["items"][1]["ok"] == "visible"

    def test_custom_redact_keys_replace_defaults(self) -> None:
        out = _render(JsonFormatter(redact_keys=["custom_field"]), custom_field=SECRET)
        assert out["custom_field"] == "***redacted***"

    def test_non_secret_fields_are_untouched(self) -> None:
        assert _render(JsonFormatter(), user_id="u-123")["user_id"] == "u-123"


class TestContextBinding:
    def test_session_id_is_attached_inside_the_block(self) -> None:
        with bind_session("sess_abc"):
            assert _render(JsonFormatter())["session_id"] == "sess_abc"

    def test_session_id_is_absent_outside_the_block(self) -> None:
        assert "session_id" not in _render(JsonFormatter())

    def test_binding_is_restored_after_nesting(self) -> None:
        with bind_session("outer"):
            with bind_session("inner"):
                assert current_session_id() == "inner"
            assert current_session_id() == "outer"
        assert current_session_id() is None

    def test_turn_id_binds_independently(self) -> None:
        with bind_session("sess_abc"), bind_turn("turn_xyz"):
            out = _render(JsonFormatter())
            assert out["session_id"] == "sess_abc"
            assert out["turn_id"] == "turn_xyz"

    def test_binding_is_restored_when_the_block_raises(self) -> None:
        with pytest.raises(RuntimeError), bind_session("sess_abc"):
            raise RuntimeError("boom")
        assert current_session_id() is None

    async def test_concurrent_tasks_do_not_share_bindings(self) -> None:
        """The whole point of contextvars: 20 sessions must not cross-contaminate."""
        observed: dict[str, str | None] = {}

        async def worker(session_id: str) -> None:
            with bind_session(session_id):
                await asyncio.sleep(0)  # force interleaving
                observed[session_id] = current_session_id()

        await asyncio.gather(*(worker(f"sess_{i}") for i in range(20)))
        assert observed == {f"sess_{i}": f"sess_{i}" for i in range(20)}


class TestConfiguration:
    def test_writes_rotating_files_per_service(self, tmp_path: Path) -> None:
        configure_logging(directory=tmp_path, per_service_files=True, console=False)
        try:
            get_logger("orphus.asr.adapter").info("asr event")
            get_logger("orphus.api.routes").info("api event")
            logging.shutdown()

            asr_log = (tmp_path / "asr.log").read_text(encoding="utf-8")
            api_log = (tmp_path / "api.log").read_text(encoding="utf-8")
            assert "asr event" in asr_log
            # A chatty ASR worker must not bury API records in the API log.
            assert "api event" not in asr_log
            assert "api event" in api_log
            assert "asr event" in (tmp_path / "orphus.log").read_text(encoding="utf-8")
        finally:
            configure_logging(directory=None, console=False)

    def test_error_log_only_receives_errors(self, tmp_path: Path) -> None:
        configure_logging(directory=tmp_path, console=False)
        try:
            log = get_logger("orphus.api.routes")
            log.info("routine")
            log.error("broken")
            logging.shutdown()

            errors = (tmp_path / "error.log").read_text(encoding="utf-8")
            assert "broken" in errors
            assert "routine" not in errors
        finally:
            configure_logging(directory=None, console=False)

    def test_configure_is_idempotent(self, tmp_path: Path) -> None:
        # Called twice, handlers must not double up and duplicate every line.
        configure_logging(directory=tmp_path, console=False)
        configure_logging(directory=tmp_path, console=False)
        try:
            get_logger("orphus.api.x").info("once")
            logging.shutdown()
            body = (tmp_path / "orphus.log").read_text(encoding="utf-8")
            assert body.count('"message": "once"') == 1
        finally:
            configure_logging(directory=None, console=False)

    def test_get_logger_works_before_configuration(self) -> None:
        # Modules log during import, long before the app reads its config.
        logger = get_logger("orphus.test.early")
        logger.info("no explosion")
        assert logging.getLogger().handlers
