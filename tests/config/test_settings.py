"""Tests for the layered configuration system."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from orphus.config import settings as settings_module
from orphus.config.settings import Settings, load_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every variable the settings loader reads.

    Without this the developer's own .env or shell leaks into assertions and the
    suite passes or fails depending on whose machine it runs on.
    """
    for name in settings_module._FLAT_ENV_ALIASES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ORPHUS_ENV", raising=False)
    # model_config is a TypedDict at runtime, so this is setitem, not setattr.
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def _load(**overrides: Any) -> Settings:
    return load_settings(overrides or None, use_cache=False)


class TestDefaults:
    def test_yaml_defaults_are_loaded(self) -> None:
        s = _load()
        assert s.asr.model_id == "nvidia/nemotron-3.5-asr-streaming-0.6b"
        assert s.tts.model_id == "canopylabs/orpheus-tts-0.1-finetune-prod"
        assert s.session.max_concurrent == 20

    def test_lookahead_derives_from_att_context(self) -> None:
        assert _load().asr.lookahead_ms == 320

    @pytest.mark.parametrize(
        ("right_context", "expected_ms"),
        [(0, 80), (1, 160), (3, 320), (6, 560), (13, 1120)],
    )
    def test_full_latency_ladder(self, right_context: int, expected_ms: int) -> None:
        s = _load(asr={"att_context_size": (56, right_context)})
        assert s.asr.lookahead_ms == expected_ms


class TestPrecedence:
    def test_flat_env_var_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "9999")
        assert _load().server.port == 9999

    def test_nested_env_var_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORPHUS_SERVER__PORT", "7777")
        assert _load().server.port == 7777

    def test_flat_alias_beats_nested_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The operator sets the flat name on the RunPod instance, so it must win.
        monkeypatch.setenv("PORT", "9999")
        monkeypatch.setenv("ORPHUS_SERVER__PORT", "7777")
        assert _load().server.port == 9999

    def test_runtime_override_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "9999")
        assert _load(server={"port": 8123}).server.port == 8123

    def test_blank_env_var_does_not_clobber_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An exported-but-empty var is the normal state of an unfilled .env.
        monkeypatch.setenv("GROK_MODEL", "   ")
        assert _load().llm.model == "grok-4-fast"

    def test_environment_yaml_layer_is_applied(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "default.yaml").write_text(
            textwrap.dedent("""
                server:
                  port: 1000
                session:
                  max_concurrent: 4
                asr:
                  max_batch_size: 4
            """),
            encoding="utf-8",
        )
        (tmp_path / "staging.yaml").write_text("server:\n  port: 2000\n", encoding="utf-8")
        monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path)
        monkeypatch.setenv("ORPHUS_ENV", "staging")

        s = _load()
        assert s.server.port == 2000  # staging layer wins
        assert s.session.max_concurrent == 4  # default layer still applies


class TestValidation:
    def test_untrained_att_context_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="right attention context"):
            _load(asr={"att_context_size": (56, 7)})

    def test_wrong_left_context_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="left attention context"):
            _load(asr={"att_context_size": (32, 3)})

    def test_repetition_penalty_floor_is_enforced(self) -> None:
        # Below 1.1 Orpheus degenerates into loops. This is a real failure mode,
        # not a style preference, so it belongs in validation.
        with pytest.raises(ValidationError):
            _load(tts={"repetition_penalty": 1.0})

    def test_default_voice_must_exist(self) -> None:
        with pytest.raises(ValidationError, match="default_voice"):
            _load(tts={"default_voice": "nonexistent"})

    def test_asr_batch_must_cover_concurrent_sessions(self) -> None:
        with pytest.raises(ValidationError, match="max_batch_size"):
            _load(asr={"max_batch_size": 4}, session={"max_concurrent": 20})

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _load(definitely_not_a_setting=1)


class TestProductionGuards:
    def test_production_requires_llm_api_key(self) -> None:
        with pytest.raises(ValidationError, match="GROK_API_KEY"):
            _load(environment="production", security={"api_keys": ["k"]})

    def test_production_requires_api_auth(self) -> None:
        with pytest.raises(ValidationError, match="api_keys"):
            _load(environment="production", llm={"api_key": "sk-test"})

    def test_production_passes_when_fully_configured(self) -> None:
        s = _load(
            environment="production",
            llm={"api_key": "sk-test"},
            security={"api_keys": ["client-key"]},
        )
        assert s.security.auth_required
        assert s.llm.is_configured

    def test_development_tolerates_missing_secrets(self) -> None:
        s = _load()
        assert not s.llm.is_configured
        assert not s.security.auth_required


class TestSecrets:
    def test_api_key_is_not_in_repr(self) -> None:
        s = _load(llm={"api_key": "sk-super-secret-value"})
        assert "sk-super-secret-value" not in repr(s)
        assert "sk-super-secret-value" not in str(s.llm.api_key)
        assert s.llm.api_key.get_secret_value() == "sk-super-secret-value"

    def test_describe_omits_secrets_entirely(self) -> None:
        s = _load(llm={"api_key": "sk-super-secret-value"})
        rendered = repr(s.describe())
        assert "sk-super-secret-value" not in rendered
        assert s.describe()["llm"]["configured"] is True

    def test_api_keys_accept_csv_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ORPHUS_API_KEYS", "alpha, beta ,gamma")
        s = _load()
        assert [k.get_secret_value() for k in s.security.api_keys] == ["alpha", "beta", "gamma"]


class TestModelPaths:
    def test_convention_used_when_path_unset(self) -> None:
        m = _load().models
        assert m.resolve("asr", m.asr_path) == Path("./models/asr")

    def test_explicit_path_wins(self) -> None:
        s = _load(models={"asr_path": "/mnt/weights/nemotron"})
        assert s.models.resolve("asr", s.models.asr_path) == Path("/mnt/weights/nemotron")
