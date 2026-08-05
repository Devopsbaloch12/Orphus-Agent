"""Fixed-capacity PCM ring buffer.

Every queue on a realtime audio path must be bounded. An unbounded buffer in
front of a model that has fallen behind does not prevent the failure, it just
converts a dropped frame into unbounded latency growth followed by an OOM. This
buffer therefore drops the *oldest* audio on overflow and counts the loss, which
matches ``scheduler.overflow_policy: drop_oldest`` in the default config: for
live conversation, stale audio is worth less than fresh audio.
"""

from __future__ import annotations

import numpy as np

from orphus.audio._logging import get_logger
from orphus.domain.types import PcmArray

__all__ = ["AudioRingBuffer"]

logger = get_logger(__name__)


class AudioRingBuffer:
    """A mono float32 ring buffer with drop-oldest overflow.

    Args:
        capacity: Maximum samples retained.

    Raises:
        ValueError: If ``capacity`` is not positive.
    """

    __slots__ = ("_capacity", "_data", "_dropped", "_head", "_size")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        # Preallocated once. Growth on the audio path would mean a malloc every
        # 20 ms per session.
        self._data: PcmArray = np.zeros(capacity, dtype=np.float32)
        self._head = 0
        self._size = 0
        self._dropped = 0

    def __len__(self) -> int:
        """Number of samples currently buffered."""
        return self._size

    @property
    def capacity(self) -> int:
        """Maximum samples the buffer can hold."""
        return self._capacity

    @property
    def free(self) -> int:
        """Samples that can be written before the oldest audio is evicted."""
        return self._capacity - self._size

    @property
    def is_empty(self) -> bool:
        """Whether the buffer holds no samples."""
        return self._size == 0

    @property
    def is_full(self) -> bool:
        """Whether the next write will evict older audio."""
        return self._size == self._capacity

    @property
    def dropped_samples(self) -> int:
        """Total samples evicted by overflow over this buffer's lifetime."""
        return self._dropped

    def clear(self) -> None:
        """Discard all buffered audio. The drop counter is preserved."""
        self._head = 0
        self._size = 0

    def write(self, samples: PcmArray) -> int:
        """Append samples, evicting the oldest audio if necessary.

        Args:
            samples: Mono float32 PCM.

        Returns:
            Number of previously-buffered samples evicted to make room.
        """
        block = np.asarray(samples, dtype=np.float32)
        if block.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {block.shape}")
        count = block.size
        if count == 0:
            return 0

        if count >= self._capacity:
            # The write alone overruns the buffer: keep only its newest tail.
            evicted = self._size + (count - self._capacity)
            self._data[:] = block[-self._capacity :]
            self._head = 0
            self._size = self._capacity
            self._dropped += evicted
            logger.warning(
                "ring buffer overrun",
                extra={"capacity": self._capacity, "written": count, "dropped": evicted},
            )
            return evicted

        evicted = max(0, count - self.free)
        if evicted:
            self._head = (self._head + evicted) % self._capacity
            self._size -= evicted
            self._dropped += evicted

        tail = (self._head + self._size) % self._capacity
        first = min(count, self._capacity - tail)
        self._data[tail : tail + first] = block[:first]
        if first < count:
            self._data[: count - first] = block[first:]
        self._size += count
        return evicted

    def peek(self, count: int) -> PcmArray:
        """Copy the oldest ``count`` samples without consuming them.

        Args:
            count: Samples requested; clamped to what is buffered.

        Returns:
            A newly allocated contiguous float32 array.
        """
        take = min(max(count, 0), self._size)
        if take == 0:
            return np.zeros(0, dtype=np.float32)
        first = min(take, self._capacity - self._head)
        if first == take:
            return self._data[self._head : self._head + take].copy()
        return np.concatenate((self._data[self._head :], self._data[: take - first]))

    def read(self, count: int) -> PcmArray:
        """Consume and return the oldest ``count`` samples.

        Args:
            count: Samples requested; clamped to what is buffered.

        Returns:
            A newly allocated contiguous float32 array.
        """
        out = self.peek(count)
        self.drop(out.size)
        return out

    def read_all(self) -> PcmArray:
        """Consume and return everything currently buffered."""
        return self.read(self._size)

    def drop(self, count: int) -> int:
        """Discard the oldest ``count`` samples without copying them.

        Args:
            count: Samples to discard; clamped to what is buffered.

        Returns:
            Number of samples actually discarded.
        """
        take = min(max(count, 0), self._size)
        self._head = (self._head + take) % self._capacity
        self._size -= take
        return take

    def tail(self, count: int) -> PcmArray:
        """Copy the *newest* ``count`` samples without consuming them.

        This is the pre-roll read: a buffer sized to ``vad.speech_pad_ms`` keeps
        exactly the audio that precedes a ``SPEECH_STARTED`` event, so the ASR
        still sees the first phoneme the VAD needed in order to fire.

        Args:
            count: Samples requested; clamped to what is buffered.

        Returns:
            A newly allocated contiguous float32 array.
        """
        take = min(max(count, 0), self._size)
        if take == 0:
            return np.zeros(0, dtype=np.float32)
        start = (self._head + self._size - take) % self._capacity
        first = min(take, self._capacity - start)
        if first == take:
            return self._data[start : start + take].copy()
        return np.concatenate((self._data[start:], self._data[: take - first]))
