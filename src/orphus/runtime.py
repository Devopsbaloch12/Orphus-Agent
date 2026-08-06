"""Production model composition and lifetime management."""

from __future__ import annotations

import asyncio
from pathlib import Path

from orphus.asr import NemotronASR
from orphus.cache import RedisCache
from orphus.config import Settings
from orphus.conversation.session import Session
from orphus.database import TurnRepository
from orphus.providers import GrokProvider
from orphus.observability.logging import get_logger
from orphus.streaming import VoicePipeline
from orphus.tts import OrpheusTTS

logger = get_logger(__name__)
from orphus.vad import SileroVAD


def _existing(path: Path) -> str | None:
    if path.is_file():
        return str(path)
    if path.is_dir() and any(item.is_file() for item in path.rglob("*")):
        return str(path)
    return None


def _vad_source(path: Path) -> str | None:
    if path.is_file():
        return str(path)
    if path.is_dir():
        for pattern in ("*.jit", "*.onnx"):
            candidate = next(path.glob(pattern), None)
            if candidate is not None:
                return str(candidate)
    return None


class ModelRuntime:
    """Load shared weights once and mint isolated per-session pipelines."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vad: SileroVAD | None = None
        self.asr: NemotronASR | None = None
        self.tts: OrpheusTTS | None = None
        self.llm: GrokProvider | None = None
        self.cache: RedisCache | None = None
        self.repository: TurnRepository | None = None

    async def load(self) -> None:
        models = self.settings.models
        vad_path = _vad_source(models.resolve("vad", models.vad_path))
        asr_path = _existing(models.resolve("asr", models.asr_path))
        tts_path = _existing(models.resolve("tts", models.tts_path))
        # Initialise GPU allocators sequentially. Concurrent CUDA/vLLM startup
        # races memory reservation and produces misleading intermittent OOMs.
        self.tts = await asyncio.to_thread(
            OrpheusTTS.load, self.settings.tts, model_path=tts_path
        )
        self.asr = await asyncio.to_thread(
            NemotronASR.load, self.settings.asr, model_path=asr_path
        )
        self.vad = await asyncio.to_thread(
            SileroVAD.load, self.settings.vad, model_path=vad_path
        )
        self.llm = GrokProvider(self.settings.llm)
        self.cache = RedisCache.connect(self.settings.redis)
        self.repository = TurnRepository.connect(self.settings.database)

    @property
    def ready(self) -> bool:
        return all((self.vad, self.asr, self.tts, self.llm))

    def pipeline(self, session: Session) -> VoicePipeline:
        if not self.ready:
            raise RuntimeError("model runtime is not loaded")
        assert self.vad is not None
        assert self.asr is not None
        assert self.tts is not None
        assert self.llm is not None
        assert self.repository is not None
        language = session.metadata.get("language")
        voice = session.metadata.get("voice")
        return VoicePipeline(
            session=session,
            vad=self.vad.open_session(session.session_id),
            asr=self.asr.open_session(session.session_id, language=language),
            llm=self.llm,
            tts=self.tts,
            voice=voice,
            output_queue_size=self.settings.scheduler.tts_queue_size,
            turn_sink=self.repository.save if self.settings.session.persist_transcripts else None,
        )

    async def aclose(self) -> None:
        components = [
            item
            for item in (
                self.llm,
                self.repository,
                self.cache,
                self.asr,
                self.tts,
                self.vad,
            )
            if item
        ]
        results = await asyncio.gather(
            *(item.aclose() for item in components), return_exceptions=True
        )
        for component, result in zip(components, results, strict=True):
            if isinstance(result, Exception):
                logger.exception(
                    f"runtime.component_close_failed component={type(component).__name__}",
                    exc_info=result,
                )
