"""GPU-resident Nemotron 3.5 ASR adapter."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from orphus.config.settings import ASRSettings
from orphus.domain.types import AudioChunk, SessionId, TranscriptEvent


class _NemotronSession:
    def __init__(self, owner: NemotronASR, language: str | None) -> None:
        self._owner = owner
        self._language = language or "auto"
        self._chunks: list[np.ndarray] = []
        self._closed = False

    async def push(self, chunk: AudioChunk) -> None:
        if self._closed:
            raise RuntimeError("ASR session is closed")
        if chunk.sample_rate != 16_000:
            raise ValueError("Nemotron ASR requires 16kHz audio")
        self._chunks.append(chunk.samples.copy())

    async def _empty_events(self) -> AsyncIterator[TranscriptEvent]:
        if False:
            yield TranscriptEvent("", False, 0.0)

    def events(self) -> AsyncIterator[TranscriptEvent]:
        return self._empty_events()

    async def flush(self) -> TranscriptEvent | None:
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks).astype(np.float32, copy=False)
        self._chunks.clear()
        text = await self._owner.transcribe(audio, self._language)
        return TranscriptEvent(text=text, is_final=True, timestamp_s=time.monotonic())

    async def aclose(self) -> None:
        self._closed = True
        self._chunks.clear()


class NemotronASR:
    """One shared model with bounded concurrent GPU generation."""

    def __init__(self, model: Any, processor: Any, settings: ASRSettings) -> None:
        self._model = model
        self._processor = processor
        self._settings = settings
        self._gate = asyncio.Semaphore(settings.max_batch_size)

    @classmethod
    def load(cls, settings: ASRSettings, *, model_path: str | None = None) -> NemotronASR:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        source = model_path or settings.model_id
        processor = AutoProcessor.from_pretrained(source)
        dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
        model = AutoModelForRNNT.from_pretrained(source, torch_dtype=dtype).to(settings.device)
        model.eval()
        return cls(model, processor, settings)

    def open_session(
        self, session_id: SessionId, *, language: str | None = None
    ) -> _NemotronSession:
        del session_id
        return _NemotronSession(self, language or self._settings.language)

    def _view(self) -> Any:
        """A per-transcription view of the shared model.

        ``generate()`` on the Nemotron RNNT stack keeps its working state on the
        module instance (``get_audio_features``, ``_step_durations``,
        ``_symbols_at_frame``, ...) and deletes it when the call unwinds. That is
        fine for one caller and fatal for a voice platform: several sessions
        transcribe at once, one call's cleanup pulls the attributes out from
        under another, and the loser dies with ``AttributeError`` -- the caller
        simply never hears a reply.

        A shallow copy gives each transcription its own ``__dict__`` while the
        parameter, buffer and submodule dicts stay shared by reference, so this
        costs no VRAM and loads no second copy of the weights.
        """
        return copy.copy(self._model)

    async def transcribe(self, audio: np.ndarray, language: str) -> str:
        async with self._gate:
            return await asyncio.to_thread(self._transcribe_sync, audio, language)

    def _transcribe_sync(self, audio: np.ndarray, language: str) -> str:
        import torch

        model = self._view()
        inputs = self._processor(audio, sampling_rate=16_000, language=language)
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
        # decode() returns a list of strings, one per batch row. str() on that
        # yields "['hello there']" -- the brackets and quotes were reaching the
        # LLM as if the caller had spoken them.
        decoded = self._processor.decode(output.sequences, skip_special_tokens=True)
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0] if decoded else ""
        return str(decoded).strip()

    async def aclose(self) -> None:
        self._model = None
        self._processor = None
