"""The inbound audio path, assembled.

Composes decode -> downmix -> resample -> frame into the single object the
session handler drives. Keeping the composition here rather than in the
WebSocket handler means the ordering constraint that actually matters is stated
once: **downmix before resampling, and resample before framing**. Resampling
interleaved stereo filters across channels and produces garbage; framing before
resampling produces frames of the wrong length.
"""

from __future__ import annotations

import numpy as np

from orphus.audio.convert import decode_pcm, downmix_to_mono
from orphus.audio.framing import FrameSlicer
from orphus.audio.resample import StreamingResampler
from orphus.domain.types import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    AudioChunk,
    AudioEncoding,
    PcmArray,
)

__all__ = ["InboundAudioPipeline"]


class InboundAudioPipeline:
    """Normalise one session's inbound audio into model-ready frames.

    One instance per session: it owns the resampler's filter history and the
    slicer's partial frame, both of which are per-stream state.

    Args:
        source_rate: Sample rate the client is sending.
        target_rate: Rate the models require.
        frame_samples: Samples per emitted frame.
        channels: Interleaved channel count in the source.
        encoding: Wire encoding used by :meth:`push_bytes`.

    Raises:
        ValueError: If ``channels`` is below 1.
    """

    __slots__ = ("_channels", "_encoding", "_resampler", "_slicer", "_source_rate")

    def __init__(
        self,
        *,
        source_rate: int,
        target_rate: int = VAD_SAMPLE_RATE,
        frame_samples: int = VAD_FRAME_SAMPLES,
        channels: int = 1,
        encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
    ) -> None:
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self._source_rate = source_rate
        self._channels = channels
        self._encoding = encoding
        self._resampler = StreamingResampler(source_rate, target_rate)
        self._slicer = FrameSlicer(frame_samples, sample_rate=target_rate)

    @property
    def source_rate(self) -> int:
        """Sample rate expected from the client."""
        return self._source_rate

    @property
    def target_rate(self) -> int:
        """Sample rate of the emitted frames."""
        return self._slicer.sample_rate

    @property
    def frame_samples(self) -> int:
        """Samples in each emitted frame."""
        return self._slicer.frame_samples

    @property
    def latency_ms(self) -> float:
        """Algorithmic delay from resampling plus a full frame, in milliseconds.

        This is the floor on how quickly a sample that arrives from the network
        can reach the VAD. It is reported so the latency budget can be audited
        rather than assumed.
        """
        resample_ms = 1000.0 * self._resampler.latency_samples / self._source_rate
        return resample_ms + 1000.0 * self._slicer.frame_duration_s

    def reset(self) -> None:
        """Clear resampler history and any partial frame."""
        self._resampler.reset()
        self._slicer.reset()

    def push_bytes(
        self, data: bytes | bytearray | memoryview, *, captured_at: float | None = None
    ) -> list[AudioChunk]:
        """Feed raw wire bytes and collect any complete frames.

        Args:
            data: PCM in this pipeline's ``encoding``, interleaved across
                ``channels``.
            captured_at: Monotonic capture time of the first sample.

        Returns:
            Zero or more frames of exactly ``frame_samples`` samples.
        """
        return self.push_samples(decode_pcm(data, self._encoding), captured_at=captured_at)

    def push(self, chunk: AudioChunk) -> list[AudioChunk]:
        """Feed an already-decoded chunk and collect any complete frames.

        Args:
            chunk: Float32 audio at ``source_rate``, interleaved across
                ``channels``.

        Returns:
            Zero or more frames of exactly ``frame_samples`` samples.

        Raises:
            ValueError: If the chunk's rate does not match ``source_rate``.
        """
        if chunk.sample_rate != self._source_rate:
            raise ValueError(
                f"pipeline configured for {self._source_rate} Hz, got {chunk.sample_rate} Hz"
            )
        return self.push_samples(chunk.samples, captured_at=chunk.captured_at)

    def push_samples(
        self, samples: PcmArray, *, captured_at: float | None = None
    ) -> list[AudioChunk]:
        """Feed decoded samples and collect any complete frames.

        Args:
            samples: Float32 PCM at ``source_rate``, interleaved across
                ``channels``.
            captured_at: Monotonic capture time of the first sample. The
                resampler's sub-millisecond group delay is not subtracted; it is
                two orders of magnitude below the ASR lookahead.

        Returns:
            Zero or more frames of exactly ``frame_samples`` samples.
        """
        mono = downmix_to_mono(np.asarray(samples, dtype=np.float32), self._channels)
        converted = self._resampler.process(mono)
        return self._frame(converted, captured_at)

    def flush(self, *, pad: bool = True) -> list[AudioChunk]:
        """Drain the resampler tail and the partial frame at end of stream.

        Args:
            pad: Zero-pad the final partial frame to full length.

        Returns:
            Any remaining frames, in order.
        """
        frames = self._frame(self._resampler.flush(), None)
        trailing = self._slicer.flush(pad=pad)
        if trailing is not None:
            frames.append(trailing)
        return frames

    def _frame(self, samples: PcmArray, captured_at: float | None) -> list[AudioChunk]:
        """Slice resampled audio into frames, preserving capture timing."""
        if samples.size == 0:
            return []
        if captured_at is None:
            block = AudioChunk(samples=samples, sample_rate=self.target_rate)
        else:
            block = AudioChunk(
                samples=samples, sample_rate=self.target_rate, captured_at=captured_at
            )
        return self._slicer.push(block)
