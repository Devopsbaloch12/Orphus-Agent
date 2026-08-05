"""Re-framing arbitrary chunk sizes into exactly-sized model frames.

Silero VAD v5 accepts *exactly* 512 samples at 16 kHz and raises otherwise, and
the cache-aware ASR encoder expects a fixed window per streaming step. Inbound
audio, meanwhile, arrives in whatever size the client's encoder happens to emit
(20 ms Opus frames, 1024-sample Web Audio blocks, ragged WebSocket messages).
The slicer absorbs that mismatch and carries the remainder across calls.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orphus.domain.types import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    AudioChunk,
    PcmArray,
)

__all__ = ["FrameSlicer", "concat_chunks"]


def concat_chunks(chunks: Sequence[AudioChunk]) -> AudioChunk:
    """Join consecutive chunks of one stream into a single chunk.

    Used to materialise a :class:`~orphus.domain.types.SpeechSegment`'s audio
    once the VAD closes an utterance.

    Args:
        chunks: Non-empty sequence in stream order, all at the same rate.

    Returns:
        One chunk carrying the concatenated samples; the first chunk's
        ``sequence`` and ``captured_at`` are preserved so latency accounting
        still refers to when the audio was captured, not when it was joined.

    Raises:
        ValueError: If ``chunks`` is empty or mixes sample rates.
    """
    if not chunks:
        raise ValueError("cannot concatenate an empty sequence of chunks")
    rate = chunks[0].sample_rate
    for chunk in chunks:
        if chunk.sample_rate != rate:
            raise ValueError(
                f"cannot concatenate chunks at different rates: {rate} and {chunk.sample_rate}"
            )
    if len(chunks) == 1:
        return chunks[0]
    return AudioChunk(
        samples=np.concatenate([chunk.samples for chunk in chunks]),
        sample_rate=rate,
        sequence=chunks[0].sequence,
        captured_at=chunks[0].captured_at,
    )


class FrameSlicer:
    """Buffer a stream and emit fixed-size frames.

    Args:
        frame_samples: Samples per emitted frame.
        sample_rate: Rate the incoming audio is expected to be at; pushes at a
            different rate are rejected rather than silently mis-timed.

    Raises:
        ValueError: If ``frame_samples`` is not positive.
    """

    __slots__ = ("_frame_samples", "_pending", "_pending_captured_at", "_sample_rate", "_sequence")

    def __init__(
        self,
        frame_samples: int = VAD_FRAME_SAMPLES,
        *,
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> None:
        if frame_samples <= 0:
            raise ValueError(f"frame_samples must be positive, got {frame_samples}")
        self._frame_samples = frame_samples
        self._sample_rate = sample_rate
        self._pending: PcmArray = np.zeros(0, dtype=np.float32)
        self._pending_captured_at: float | None = None
        self._sequence = 0

    @property
    def frame_samples(self) -> int:
        """Samples in each emitted frame."""
        return self._frame_samples

    @property
    def sample_rate(self) -> int:
        """Sample rate of the frames this slicer emits."""
        return self._sample_rate

    @property
    def pending(self) -> int:
        """Samples buffered but not yet forming a complete frame."""
        return int(self._pending.size)

    @property
    def frame_duration_s(self) -> float:
        """Wall duration of one frame in seconds."""
        return self._frame_samples / self._sample_rate

    def reset(self) -> None:
        """Discard the partial frame and restart frame numbering."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_captured_at = None
        self._sequence = 0

    def push_samples(
        self, samples: PcmArray, *, captured_at: float | None = None
    ) -> list[PcmArray]:
        """Append raw samples and return every complete frame they produced.

        Args:
            samples: Mono float32 PCM at ``sample_rate``.
            captured_at: Monotonic capture time of ``samples[0]``, used to date
                the emitted frames.

        Returns:
            Zero or more arrays of exactly ``frame_samples`` samples.
        """
        block = np.asarray(samples, dtype=np.float32)
        if block.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {block.shape}")
        if self._pending_captured_at is None and captured_at is not None:
            self._pending_captured_at = captured_at

        if self._pending.size:
            block = np.concatenate((self._pending, block))

        total = block.size
        complete = total // self._frame_samples
        if complete == 0:
            self._pending = block
            return []

        cut = complete * self._frame_samples
        frames = [
            block[index * self._frame_samples : (index + 1) * self._frame_samples]
            for index in range(complete)
        ]
        self._pending = block[cut:].copy()
        if self._pending.size == 0:
            self._pending_captured_at = None
        elif self._pending_captured_at is not None:
            self._pending_captured_at += cut / self._sample_rate
        return frames

    def push(self, chunk: AudioChunk) -> list[AudioChunk]:
        """Append a chunk and return every complete frame it produced.

        Args:
            chunk: Mono float32 audio at this slicer's ``sample_rate``.

        Returns:
            Zero or more chunks of exactly ``frame_samples`` samples, numbered
            consecutively and dated from the capture time of their first sample.

        Raises:
            ValueError: If the chunk's sample rate does not match. Resampling is
                the caller's job; doing it implicitly here would hide a
                misconfigured pipeline until it showed up as a WER regression.
        """
        if chunk.sample_rate != self._sample_rate:
            raise ValueError(
                f"slicer expects {self._sample_rate} Hz, got {chunk.sample_rate} Hz; "
                "resample before framing"
            )
        origin = self._pending_captured_at
        frames = self.push_samples(chunk.samples, captured_at=chunk.captured_at)
        if origin is None:
            origin = chunk.captured_at
        return [self._wrap(samples, origin, index) for index, samples in enumerate(frames)]

    def flush(self, *, pad: bool = True) -> AudioChunk | None:
        """Emit whatever remains, optionally zero-padded to a full frame.

        Args:
            pad: Zero-pad the remainder up to ``frame_samples``. Required before
                handing a final partial window to a model with a fixed input
                size; set ``False`` to recover the raw tail instead.

        Returns:
            The trailing frame, or ``None`` when nothing is buffered.
        """
        if self._pending.size == 0:
            return None
        samples = self._pending
        origin = self._pending_captured_at
        if pad:
            samples = np.concatenate(
                (samples, np.zeros(self._frame_samples - samples.size, dtype=np.float32))
            )
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_captured_at = None
        return self._wrap(samples, origin, 0)

    def _wrap(self, samples: PcmArray, origin: float | None, index: int) -> AudioChunk:
        """Package one frame with a sequence number and a capture timestamp."""
        sequence = self._sequence
        self._sequence += 1
        if origin is None:
            return AudioChunk(samples=samples, sample_rate=self._sample_rate, sequence=sequence)
        return AudioChunk(
            samples=samples,
            sample_rate=self._sample_rate,
            sequence=sequence,
            captured_at=origin + index * self.frame_duration_s,
        )
