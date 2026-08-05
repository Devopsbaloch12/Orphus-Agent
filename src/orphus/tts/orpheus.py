"""Resident Orpheus TTS adapter around the official streaming package."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from orphus.audio.convert import pcm16_to_float32
from orphus.config.settings import TTSSettings
from orphus.domain.types import TTS_SAMPLE_RATE, AudioChunk, SessionId, TextDelta


def _next_chunk(iterator: Any) -> tuple[bool, bytes]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, b""


class _OrpheusSession:
    def __init__(
        self,
        owner: OrpheusTTS,
        text_stream: AsyncIterator[TextDelta],
        voice: str,
        session_id: SessionId,
    ) -> None:
        self._owner = owner
        self._text_stream = text_stream
        self._voice = voice
        self._session_id = session_id
        self._cancelled = asyncio.Event()

    async def audio(self) -> AsyncIterator[AudioChunk]:
        buffer = ""
        async for delta in self._text_stream:
            if self._cancelled.is_set():
                return
            buffer += delta.text
            if delta.is_final or self._owner.ready(buffer):
                text, buffer = buffer.strip(), ""
                if text:
                    async for chunk in self._owner.synthesize_text(
                        text, self._voice, self._session_id, self._cancelled
                    ):
                        yield chunk
        if buffer.strip() and not self._cancelled.is_set():
            async for chunk in self._owner.synthesize_text(
                buffer.strip(), self._voice, self._session_id, self._cancelled
            ):
                yield chunk

    async def cancel(self) -> None:
        self._cancelled.set()


class OrpheusTTS:
    def __init__(self, model: Any, settings: TTSSettings) -> None:
        self._model = model
        self._settings = settings

    @classmethod
    def load(cls, settings: TTSSettings, *, model_path: str | None = None) -> OrpheusTTS:
        from pathlib import Path

        from orpheus_tts import OrpheusModel

        tokenizer: str | None = None
        if model_path:
            candidate = Path(model_path).parent / "tts-tokenizer"
            if candidate.is_dir():
                tokenizer = str(candidate)
        model = OrpheusModel(
            model_name=model_path or settings.model_id,
            tokenizer=tokenizer or settings.model_id,
            max_model_len=settings.max_model_len,
            gpu_memory_utilization=settings.gpu_memory_utilization,
        )
        return cls(model, settings)

    @property
    def available_voices(self) -> tuple[str, ...]:
        return tuple(self._settings.voices)

    def ready(self, text: str) -> bool:
        return len(text) >= self._settings.min_chunk_chars and any(
            marker in text for marker in self._settings.sentence_terminators
        )

    def synthesize(
        self,
        text_stream: AsyncIterator[TextDelta],
        *,
        session_id: SessionId,
        voice: str | None = None,
    ) -> _OrpheusSession:
        selected = voice or self._settings.default_voice
        if selected not in self.available_voices:
            raise ValueError(f"unsupported voice: {selected}")
        return _OrpheusSession(self, text_stream, selected, session_id)

    async def synthesize_text(
        self,
        text: str,
        voice: str,
        session_id: SessionId,
        cancelled: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        iterator = self._model.generate_speech(
            prompt=text,
            voice=voice,
            request_id=str(session_id),
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            repetition_penalty=self._settings.repetition_penalty,
            stop_token_ids=[128258],
            max_tokens=self._settings.max_model_len,
        )
        sequence = 0
        while not cancelled.is_set():
            found, raw = await asyncio.to_thread(_next_chunk, iterator)
            if not found:
                return
            yield AudioChunk(pcm16_to_float32(raw), TTS_SAMPLE_RATE, sequence=sequence)
            sequence += 1

    async def aclose(self) -> None:
        shutdown = getattr(self._model, "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if asyncio.iscoroutine(result):
                await result
        self._model = None
