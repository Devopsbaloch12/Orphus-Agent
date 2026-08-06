"""Silero VAD adapter with strictly isolated recurrent state per session."""

from __future__ import annotations

import copy
from typing import Any

from orphus.config.settings import VADSettings
from orphus.domain.types import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    AudioChunk,
    SessionId,
    VadEvent,
    VadEventKind,
)

__all__ = ["SileroVAD"]


class _SileroSession:
    def __init__(self, model: Any, settings: VADSettings) -> None:
        self._model = model
        self._settings = settings
        self._speech_frames = 0
        self._silence_frames = 0
        self._frame_index = 0
        self._speaking = False

    def process(self, chunk: AudioChunk) -> VadEvent | None:
        if chunk.sample_rate != VAD_SAMPLE_RATE or chunk.num_samples != VAD_FRAME_SAMPLES:
            raise ValueError("Silero requires exactly 512 mono samples at 16kHz")
        import torch

        probability = float(self._model(torch.from_numpy(chunk.samples), VAD_SAMPLE_RATE).item())
        self._frame_index += 1
        timestamp = self._frame_index * VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE
        if probability >= self._settings.threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            required = max(1, round(self._settings.min_speech_ms / 32))
            if not self._speaking and self._speech_frames >= required:
                self._speaking = True
                return VadEvent(VadEventKind.SPEECH_STARTED, timestamp, probability)
        else:
            self._silence_frames += 1
            self._speech_frames = 0
            required = max(1, round(self._settings.min_silence_ms / 32))
            if self._speaking and self._silence_frames >= required:
                self._speaking = False
                return VadEvent(VadEventKind.SPEECH_ENDED, timestamp, probability)
        return None

    def reset(self) -> None:
        reset = getattr(self._model, "reset_states", None)
        if reset is not None:
            reset()
        self._speech_frames = 0
        self._silence_frames = 0
        self._frame_index = 0
        self._speaking = False


class SileroVAD:
    """Shared model template that creates an independent recurrent model per call."""

    def __init__(self, model: Any, settings: VADSettings) -> None:
        self._model = model
        self._settings = settings

    @classmethod
    def load(cls, settings: VADSettings, *, model_path: str | None = None) -> SileroVAD:
        if model_path and model_path.endswith(".jit"):
            import torch

            model = torch.jit.load(model_path, map_location="cpu")
        else:
            from silero_vad import load_silero_vad

            model = load_silero_vad(onnx=bool(model_path and model_path.endswith(".onnx")))
        return cls(model, settings)

    def open_session(self, session_id: SessionId) -> _SileroSession:
        del session_id
        return _SileroSession(self._clone_model(), self._settings)

    def _clone_model(self) -> Any:
        """Per-call model with independent recurrent state.

        The ONNX build wraps an ``onnxruntime.InferenceSession``, which cannot
        be pickled and therefore cannot be deep-copied -- and does not need to
        be: ORT sessions are safe for concurrent ``run()`` calls. Only the
        recurrent state (``_state``/``_context``) may not be shared between
        calls, so shallow-copy the wrapper and reset the clone's state, which
        rebinds those tensors on the clone alone.
        """
        try:
            return copy.deepcopy(self._model)
        except TypeError:
            clone = copy.copy(self._model)
            reset = getattr(clone, "reset_states", None)
            if reset is not None:
                reset()
            return clone

    async def aclose(self) -> None:
        self._model = None
