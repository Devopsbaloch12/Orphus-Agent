"""Audio plumbing shared by the VAD, ASR, and TTS adapters.

Nothing in this package imports torch, ONNX Runtime, or vLLM. It is pure numpy,
so the API layer can normalise audio on a machine with no CUDA device, and the
model adapters can be unit-tested without weights.
"""

from __future__ import annotations

from orphus.audio.buffer import AudioRingBuffer
from orphus.audio.convert import (
    decode_pcm,
    downmix_to_mono,
    encode_pcm,
    float32_to_pcm16,
    float32_to_pcm16_bytes,
    pcm16_to_float32,
    rms_dbfs,
)
from orphus.audio.framing import FrameSlicer, concat_chunks
from orphus.audio.pipeline import InboundAudioPipeline
from orphus.audio.resample import StreamingResampler, resample, resample_chunk

__all__ = [
    "AudioRingBuffer",
    "FrameSlicer",
    "InboundAudioPipeline",
    "StreamingResampler",
    "concat_chunks",
    "decode_pcm",
    "downmix_to_mono",
    "encode_pcm",
    "float32_to_pcm16",
    "float32_to_pcm16_bytes",
    "pcm16_to_float32",
    "resample",
    "resample_chunk",
    "rms_dbfs",
]
