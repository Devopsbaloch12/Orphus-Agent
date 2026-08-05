"""Typed, layered configuration.

Precedence, lowest to highest:

1. ``config/default.yaml``
2. ``config/<environment>.yaml`` -- selected by ``ORPHUS_ENV``
3. ``config/local.yaml`` -- gitignored developer overrides
4. ``.env``
5. process environment
6. runtime overrides passed to :func:`load_settings`

Two environment-variable spellings are supported. Nested keys use the prefixed
double-underscore form (``ORPHUS_ASR__ATT_CONTEXT_SIZE``), and the flat names
required by the deployment spec (``HOST``, ``PORT``, ``GROK_API_KEY``,
``REDIS_URL``, ...) are mapped onto their nested paths by
:class:`FlatEnvAliasSource`. The flat form wins, because that is what the
operator sets on the RunPod instance.

Secrets are ``SecretStr`` so that an accidental ``repr()`` of the settings
object -- in a log line, a traceback, or a health payload -- cannot leak them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

__all__ = [
    "ASRSettings",
    "DatabaseSettings",
    "LLMSettings",
    "LoggingSettings",
    "ModelSettings",
    "MonitoringSettings",
    "RedisSettings",
    "SchedulerSettings",
    "SecuritySettings",
    "ServerSettings",
    "SessionSettings",
    "Settings",
    "TTSSettings",
    "VADSettings",
    "load_settings",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG_DIR: Path = PROJECT_ROOT / "config"

# Flat environment variable -> dotted path in the settings tree. These are the
# names the deployment spec and deploy.sh use.
_FLAT_ENV_ALIASES: dict[str, str] = {
    "HOST": "server.host",
    "PORT": "server.port",
    "REQUEST_TIMEOUT": "server.request_timeout_s",
    "WORKER_COUNT": "scheduler.worker_count",
    "MAX_CONCURRENT_SESSIONS": "session.max_concurrent",
    "GROK_API_KEY": "llm.api_key",
    "GROK_BASE_URL": "llm.base_url",
    "GROK_MODEL": "llm.model",
    "REDIS_URL": "redis.url",
    "DATABASE_URL": "database.url",
    "LOG_LEVEL": "logging.level",
    "LOG_DIR": "logging.directory",
    "LOG_FORMAT": "logging.format",
    "MODEL_ROOT": "models.root",
    "ASR_MODEL_PATH": "models.asr_path",
    "TTS_MODEL_PATH": "models.tts_path",
    "VAD_MODEL_PATH": "models.vad_path",
    "ORPHUS_API_KEYS": "security.api_keys",
}

# Left attention context is fixed by the checkpoint; only these right-context
# values have trained cache configurations, and anything else silently degrades
# transcription quality rather than failing loudly. Module-level rather than a
# class attribute because a leading underscore inside a Pydantic model is
# captured as a ModelPrivateAttr and is not readable from a validator.
_VALID_RIGHT_CONTEXT: frozenset[int] = frozenset({0, 1, 3, 6, 13})


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class ServerSettings(BaseModel):
    """HTTP/WebSocket server configuration."""

    host: str = "0.0.0.0"  # noqa: S104 -- binding all interfaces is intended in a container
    port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1)
    request_timeout_s: float = Field(default=30.0, gt=0)
    shutdown_grace_s: float = Field(default=20.0, gt=0)
    cors_origins: list[str] = Field(default_factory=list)


class VADSettings(BaseModel):
    """Silero VAD tuning."""

    enabled: bool = True
    device: str = "cpu"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(default=250, ge=0)
    min_silence_ms: int = Field(default=700, ge=0)
    speech_pad_ms: int = Field(default=120, ge=0)


class ASRSettings(BaseModel):
    """Nemotron 3.5 cache-aware streaming ASR tuning."""

    model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    device: str = "cuda:0"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    att_context_size: tuple[int, int] = (56, 3)
    language: str | None = None
    max_batch_size: int = Field(default=20, ge=1)
    batch_timeout_ms: int = Field(default=12, ge=0)
    strip_lang_tags: bool = True

    @field_validator("att_context_size")
    @classmethod
    def _validate_context(cls, value: tuple[int, int]) -> tuple[int, int]:
        left, right = value
        if left != 56:
            raise ValueError(f"left attention context must be 56, got {left}")
        if right not in _VALID_RIGHT_CONTEXT:
            valid = sorted(_VALID_RIGHT_CONTEXT)
            raise ValueError(
                f"right attention context must be one of {valid}, got {right}. "
                "These map to 80/160/320/560/1120 ms of lookahead."
            )
        return value

    @property
    def lookahead_ms(self) -> int:
        """Decoder lookahead implied by ``att_context_size``, in milliseconds."""
        return (self.att_context_size[1] + 1) * 80


class TTSSettings(BaseModel):
    """Orpheus TTS tuning."""

    model_id: str = "canopylabs/orpheus-tts-0.1-finetune-prod"
    device: str = "cuda:0"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    default_voice: str = "tara"
    voices: list[str] = Field(
        default_factory=lambda: ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    )
    max_model_len: int = Field(default=2048, ge=256)
    gpu_memory_utilization: float = Field(default=0.45, gt=0.0, le=1.0)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.1)
    min_chunk_chars: int = Field(default=40, ge=1)
    sentence_terminators: list[str] = Field(
        default_factory=lambda: [".", "!", "?", ";", ":", "\n"]
    )

    @model_validator(mode="after")
    def _default_voice_is_available(self) -> Self:
        if self.default_voice not in self.voices:
            raise ValueError(
                f"default_voice {self.default_voice!r} is not in voices {self.voices}"
            )
        return self


class LLMSettings(BaseModel):
    """LLM provider configuration. Provider-agnostic by design."""

    provider: str = "grok"
    model: str = "grok-4-fast"
    base_url: str = "https://api.x.ai/v1"
    api_key: SecretStr = SecretStr("")
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_s: float = Field(default=30.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_s: float = Field(default=0.5, gt=0)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_reset_s: float = Field(default=30.0, gt=0)
    system_prompt: str = (
        "You are a helpful voice assistant. Your replies are spoken aloud, so "
        "keep them concise and conversational."
    )

    @property
    def is_configured(self) -> bool:
        """Whether an API key is present."""
        return bool(self.api_key.get_secret_value().strip())


class SessionSettings(BaseModel):
    """Conversation lifecycle limits."""

    max_concurrent: int = Field(default=20, ge=1)
    idle_timeout_s: float = Field(default=300.0, gt=0)
    max_duration_s: float = Field(default=3600.0, gt=0)
    history_max_turns: int = Field(default=20, ge=1)
    history_max_chars: int = Field(default=8000, ge=256)
    persist_transcripts: bool = True


class SchedulerSettings(BaseModel):
    """Worker pool and queue bounds."""

    audio_queue_size: int = Field(default=256, ge=1)
    asr_queue_size: int = Field(default=128, ge=1)
    llm_queue_size: int = Field(default=64, ge=1)
    tts_queue_size: int = Field(default=128, ge=1)
    overflow_policy: Literal["drop_oldest", "drop_newest", "reject"] = "drop_oldest"
    worker_count: int = Field(default=4, ge=1)
    admission_high_watermark: float = Field(default=0.85, gt=0.0, le=1.0)
    drain_timeout_s: float = Field(default=15.0, gt=0)


class RedisSettings(BaseModel):
    """Redis connection pool."""

    url: str = "redis://localhost:6379/0"
    max_connections: int = Field(default=50, ge=1)
    socket_timeout_s: float = Field(default=5.0, gt=0)
    key_prefix: str = "orphus:"


class DatabaseSettings(BaseModel):
    """PostgreSQL connection pool."""

    url: str = "postgresql+asyncpg://orphus:orphus@localhost:5432/orphus"
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout_s: float = Field(default=30.0, gt=0)
    echo: bool = False


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"
    directory: Path = Path("./logs")
    per_service_files: bool = True
    max_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)
    backup_count: int = Field(default=10, ge=0)
    console: bool = True
    redact_keys: list[str] = Field(
        default_factory=lambda: [
            "api_key",
            "authorization",
            "grok_api_key",
            "password",
            "token",
            "secret",
        ]
    )

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class MonitoringSettings(BaseModel):
    """Prometheus and tracing configuration."""

    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    gpu_metrics_enabled: bool = True
    gpu_poll_interval_s: float = Field(default=5.0, gt=0)
    tracing_enabled: bool = False
    otlp_endpoint: str | None = None


class SecuritySettings(BaseModel):
    """Authentication, rate limiting, and input bounds."""

    api_keys: list[SecretStr] = Field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=120, ge=1)
    rate_limit_window_s: int = Field(default=60, ge=1)
    max_audio_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_text_length: int = Field(default=4000, ge=1)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept ``ORPHUS_API_KEYS=key1,key2`` as well as a YAML list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def auth_required(self) -> bool:
        """Whether the auth hook should enforce a key."""
        return bool(self.api_keys)


class ModelSettings(BaseModel):
    """Where model weights live on disk."""

    root: Path = Path("./models")
    asr_path: Path | None = None
    tts_path: Path | None = None
    vad_path: Path | None = None
    download_timeout_s: float = Field(default=1800.0, gt=0)

    def resolve(self, component: str, explicit: Path | None) -> Path:
        """Resolve a component's weight directory.

        Args:
            component: One of ``asr``, ``tts``, ``vad``.
            explicit: Configured override, or ``None`` to use the convention.

        Returns:
            ``explicit`` when set, otherwise ``<root>/<component>``.
        """
        return explicit if explicit is not None else self.root / component


# ---------------------------------------------------------------------------
# Settings sources
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Layer the YAML files in ``config/`` beneath every other source."""

    def __init__(self, settings_cls: type[BaseSettings], config_dir: Path) -> None:
        super().__init__(settings_cls)
        self._config_dir = config_dir

    def _load(self) -> dict[str, Any]:
        environment = os.environ.get("ORPHUS_ENV", "development")
        candidates = [
            self._config_dir / "default.yaml",
            self._config_dir / f"{environment}.yaml",
            self._config_dir / "local.yaml",
        ]
        data: dict[str, Any] = {}
        for path in candidates:
            if not path.is_file():
                continue
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if loaded is None:
                continue
            if not isinstance(loaded, dict):
                raise TypeError(f"{path} must contain a YAML mapping at the top level")
            data = _deep_merge(data, loaded)
        return data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Unused: this source supplies the whole tree via ``__call__``."""
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        return self._load()


class FlatEnvAliasSource(PydanticBaseSettingsSource):
    """Map the spec's flat env var names onto their nested settings paths.

    ``GROK_API_KEY`` becomes ``{"llm": {"api_key": ...}}``. Only variables that
    are actually set and non-empty are emitted, so an exported-but-blank value
    never clobbers a real YAML default.
    """

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Unused: this source supplies the whole tree via ``__call__``."""
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for env_name, dotted_path in _FLAT_ENV_ALIASES.items():
            raw = os.environ.get(env_name)
            if raw is None or raw.strip() == "":
                continue
            cursor = data
            *parents, leaf = dotted_path.split(".")
            for part in parents:
                cursor = cursor.setdefault(part, {})
            cursor[leaf] = raw
        return data


class RuntimeOverrideSource(PydanticBaseSettingsSource):
    """Highest-precedence source, fed by :func:`load_settings`."""

    def __init__(self, settings_cls: type[BaseSettings], overrides: Mapping[str, Any]) -> None:
        super().__init__(settings_cls)
        self._overrides = dict(overrides)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Unused: this source supplies the whole tree via ``__call__``."""
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        return self._overrides


# Module-level channel for the runtime overrides, read by
# ``settings_customise_sources``. Pydantic constructs sources as a classmethod,
# so there is no instance to thread the overrides through.
_runtime_overrides: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Fully resolved platform configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ORPHUS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_default=True,
    )

    environment: Literal["development", "staging", "production"] = "development"

    server: ServerSettings = Field(default_factory=ServerSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    models: ModelSettings = Field(default_factory=ModelSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the configuration layers. Earlier sources win."""
        return (
            RuntimeOverrideSource(settings_cls, _runtime_overrides),
            init_settings,
            FlatEnvAliasSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlSettingsSource(settings_cls, CONFIG_DIR),
        )

    @model_validator(mode="after")
    def _check_cross_section_invariants(self) -> Self:
        """Reject configurations that would fail confusingly at runtime."""
        if self.asr.max_batch_size < self.session.max_concurrent:
            raise ValueError(
                f"asr.max_batch_size ({self.asr.max_batch_size}) is below "
                f"session.max_concurrent ({self.session.max_concurrent}); a full "
                "house of sessions would not fit in one ASR batch and latency "
                "would spike under load"
            )
        if self.environment == "production":
            if not self.llm.is_configured:
                raise ValueError("GROK_API_KEY must be set when environment=production")
            if not self.security.api_keys:
                raise ValueError(
                    "security.api_keys (ORPHUS_API_KEYS) must be set when "
                    "environment=production; the API would otherwise be unauthenticated"
                )
        return self

    def describe(self) -> dict[str, Any]:
        """Render a log-safe summary. Secrets are excluded, never redacted-in-place.

        Returns:
            A mapping suitable for emitting at startup.
        """
        return {
            "environment": self.environment,
            "server": {"host": self.server.host, "port": self.server.port},
            "asr": {
                "model_id": self.asr.model_id,
                "device": self.asr.device,
                "lookahead_ms": self.asr.lookahead_ms,
                "max_batch_size": self.asr.max_batch_size,
            },
            "tts": {
                "model_id": self.tts.model_id,
                "device": self.tts.device,
                "gpu_memory_utilization": self.tts.gpu_memory_utilization,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "configured": self.llm.is_configured,
            },
            "session": {"max_concurrent": self.session.max_concurrent},
            "auth_required": self.security.auth_required,
        }


def load_settings(
    overrides: Mapping[str, Any] | None = None,
    *,
    use_cache: bool = True,
) -> Settings:
    """Build the settings object.

    Args:
        overrides: Highest-precedence values, keyed by section. Intended for
            tests and for the CLI's ``--set`` flag, not for production config.
        use_cache: Reuse the process-wide cached instance. Pass ``False`` in
            tests that mutate the environment.

    Returns:
        A validated :class:`Settings`.

    Raises:
        pydantic.ValidationError: If any layer produces an invalid value.
    """
    global _runtime_overrides

    if overrides:
        _runtime_overrides = dict(overrides)
        use_cache = False
    else:
        _runtime_overrides = {}

    if use_cache:
        return _cached_settings()
    return Settings()


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Drop the cached settings. For tests that change the environment."""
    _cached_settings.cache_clear()
