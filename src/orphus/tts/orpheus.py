"""Resident Orpheus TTS adapter around the official streaming package."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from orphus.audio.convert import pcm16_to_float32
from orphus.config.settings import TTSSettings
from orphus.domain.types import TTS_SAMPLE_RATE, AudioChunk, SessionId, TextDelta
from orphus.observability.logging import get_logger

logger = get_logger(__name__)


# First id past the base Llama-3 vocabulary in this checkpoint's tokenizer:
# raw_token_id - CUSTOM_TOKEN_BASE == the N in "<custom_token_N>". Verified
# against this checkpoint (models/tts-tokenizer): 128258 -> <custom_token_2>,
# 130317 -> <custom_token_2061>, 135637 -> <custom_token_7381>.
_CUSTOM_TOKEN_BASE = 128256


async def _decode_tokens(token_id_gen: AsyncIterator[list[int]]) -> AsyncIterator[bytes]:
    """SNAC-decode a token stream into PCM, off the event loop.

    Same 7-token cadence and 28-token sliding window as
    ``orpheus_tts.decoder.tokens_decoder``, but consumes vLLM's raw,
    cumulative ``token_ids`` directly instead of re-parsing the cumulative
    *text* on every callback. ``orpheus_tts.decoder.turn_token_into_id``
    extracts only the single last ``<custom_token_N>`` substring via
    ``str.rfind`` -- correct for exactly one new token per callback, but
    vLLM's async engine does not guarantee that granularity, and any
    callback that advances by more than one token silently drops the
    skipped ones. That starves the ``count > 27`` gate below almost
    permanently: a full multi-sentence reply was observed producing 100+
    real tokens across 100+ callbacks yet decoding to a single ~85ms audio
    chunk. Indexing into ``token_ids`` (monotonically growing, one entry per
    real token, independent of callback granularity) has no such gap.

    ``convert_to_audio`` is a synchronous CUDA decode; upstream awaits it
    inline. On one call that is merely a stall; with twenty concurrent
    sessions it blocks the loop that every other session's audio, sockets
    and VAD depend on, so the decode is handed to a worker thread instead.
    """
    from orpheus_tts.decoder import convert_to_audio

    buffer: list[int] = []
    count = 0
    seen = 0
    async for token_ids in token_id_gen:
        for raw_id in token_ids[seen:]:
            seen += 1
            token = (raw_id - _CUSTOM_TOKEN_BASE) - 10 - ((count % 7) * 4096)
            if token <= 0:
                continue
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                decode_started = time.monotonic()
                audio = await asyncio.to_thread(convert_to_audio, buffer[-28:], count)
                logger.info(
                    f"tts.snac_decode_latency count={count} "
                    f"latency_s={time.monotonic() - decode_started:.4f}"
                )
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

    async def _pump(
        self, gen: AsyncIterator[AudioChunk], queue: asyncio.Queue[AudioChunk | None]
    ) -> None:
        # The sentinel must reach the consumer even when `gen` raises, or
        # `audio()`'s drain loop blocks on `queue.get()` forever -- a mid-
        # synthesis failure (GPU hiccup, vLLM error) would hang the call
        # instead of surfacing as an error. `finally` still lets the
        # original exception propagate afterward so `await task` on the
        # consumer side sees it.
        try:
            async for chunk in gen:
                await queue.put(chunk)
        finally:
            await queue.put(None)

    def _dispatch(
        self, text: str, pending: list[tuple[asyncio.Task[None], asyncio.Queue[AudioChunk | None]]]
    ) -> None:
        logger.info(f"tts.dispatch_sentence session={self._session_id} text={text!r}")
        queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        gen = self._owner.synthesize_text(
            text, self._voice, self._session_id, self._cancelled
        )
        task = asyncio.create_task(self._pump(gen, queue))
        pending.append((task, queue))

    async def audio(self) -> AsyncIterator[AudioChunk]:
        # Sentences are dispatched to vLLM as soon as each is detected in the
        # text stream, all running concurrently -- not one full sentence's
        # audio at a time. A sequential await-then-yield here means sentence
        # N+1's multi-second synthesis startup only ever begins after
        # sentence N's audio has *finished being consumed by the caller*,
        # producing an audible dead-air gap with no data to buffer around it.
        # Draining each sentence's queue in order still yields audio in the
        # correct sequence; only the GPU-side generation runs ahead.
        buffer = ""
        pending: list[tuple[asyncio.Task[None], asyncio.Queue[AudioChunk | None]]] = []
        async for delta in self._text_stream:
            if self._cancelled.is_set():
                logger.info(f"tts.session_cancelled session={self._session_id}")
                for task, _ in pending:
                    task.cancel()
                return
            buffer += delta.text
            if delta.is_final or self._owner.ready(buffer):
                text, buffer = buffer.strip(), ""
                if text:
                    self._dispatch(text, pending)
        logger.info(
            f"tts.text_stream_exhausted session={self._session_id} "
            f"remaining_buffer={buffer!r}"
        )
        if buffer.strip() and not self._cancelled.is_set():
            self._dispatch(buffer.strip(), pending)

        for task, queue in pending:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if self._cancelled.is_set():
                    task.cancel()
                    for remaining_task, _ in pending:
                        remaining_task.cancel()
                    return
                yield chunk
            await task

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

        issued = time.monotonic()

        async def token_gen() -> AsyncIterator[list[int]]:
            first = True
            async for result in model.engine.generate(
                prompt=model._format_prompt(text, voice),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                if first:
                    logger.info(
                        f"tts.first_token_latency request={request_id} "
                        f"latency_s={time.monotonic() - issued:.3f}"
                    )
                    first = False
                if cancelled.is_set():
                    await model.engine.abort(request_id)
                    return
                yield result.outputs[0].token_ids

        started = time.monotonic()
        sequence = 0
        async for raw in _decode_tokens(token_gen()):
            if cancelled.is_set():
                logger.info(
                    f"tts.synthesize_text_cancelled request={request_id} "
                    f"chunks={sequence} elapsed_s={time.monotonic() - started:.2f}"
                )
                await model.engine.abort(request_id)
                return
            yield AudioChunk(pcm16_to_float32(raw), TTS_SAMPLE_RATE, sequence=sequence)
            sequence += 1
        logger.info(
            f"tts.synthesize_text_done request={request_id} text_len={len(text)} "
            f"chunks={sequence} elapsed_s={time.monotonic() - started:.2f}"
        )

    async def aclose(self) -> None:
        shutdown = getattr(self._model, "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if asyncio.iscoroutine(result):
                await result
        self._model = None
