"""Resident Orpheus TTS adapter around the official streaming package."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from orphus.audio.convert import pcm16_to_float32
from orphus.config.settings import TTSSettings
from orphus.domain.types import TTS_SAMPLE_RATE, AudioChunk, SessionId, TextDelta


async def _decode_tokens(token_gen: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """SNAC-decode a token stream into PCM, off the event loop.

    Mirrors ``orpheus_tts.decoder.tokens_decoder`` -- same 7-token cadence and
    28-token sliding window -- except that ``convert_to_audio`` is a synchronous
    CUDA decode, and upstream awaits it inline. On one call that is merely a
    stall; with twenty concurrent sessions it blocks the loop that every other
    session's audio, sockets and VAD depend on, so the decode is handed to a
    worker thread instead.
    """
    from orpheus_tts.decoder import convert_to_audio, turn_token_into_id

    buffer: list[int] = []
    count = 0
    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is None or token <= 0:
            continue
        buffer.append(token)
        count += 1
        if count % 7 == 0 and count > 27:
            audio = await asyncio.to_thread(convert_to_audio, buffer[-28:], count)
            if audio is not None:
                yield audio


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
        """Stream synthesis for one text chunk, driven on the server's loop.

        Deliberately *not* ``OrpheusModel.generate_speech``. That helper runs
        each synthesis in its own thread under ``asyncio.run()``, and vLLM's
        ``AsyncLLM`` binds its single output-handler task to whichever loop
        first issues a request. The first synthesis therefore adopted a
        throwaway loop, and when ``asyncio.run()`` tore that loop down the
        handler was cancelled -- after which *every* later request failed with
        ``EngineDeadError`` even though the engine process was healthy. One
        call worked, the rest heard silence.

        Driving ``engine.generate`` directly keeps the handler on uvicorn's
        long-lived loop and lets vLLM batch all concurrent sessions natively,
        which is what it exists to do.
        """
        from vllm import SamplingParams

        model = self._model
        sampling_params = SamplingParams(
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            max_tokens=self._settings.max_model_len,
            stop_token_ids=[128258],
            repetition_penalty=self._settings.repetition_penalty,
        )
        # Unique per synthesis: a session emits one of these per sentence chunk,
        # and vLLM requires request ids to be globally unique.
        request_id = f"{session_id}-{uuid.uuid4().hex}"

        async def token_gen() -> AsyncIterator[str]:
            async for result in model.engine.generate(
                prompt=model._format_prompt(text, voice),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                if cancelled.is_set():
                    await model.engine.abort(request_id)
                    return
                yield result.outputs[0].text

        sequence = 0
        async for raw in _decode_tokens(token_gen()):
            if cancelled.is_set():
                await model.engine.abort(request_id)
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
