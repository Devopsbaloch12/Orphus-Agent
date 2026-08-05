"""Streaming sample-rate conversion.

Clients arrive at 8k (telephony), 16k (native), 44.1k, or 48k (browser
``getUserMedia``); the ASR and the VAD demand exactly 16k and Orpheus emits 24k.
Conversion therefore happens on every session, on the hot path.

Design notes:

* **Windowed-sinc with a per-phase kernel table**, not ``upfirdn`` zero-stuffing.
  For 44.1k -> 16k the rational factors are 160/441; a polyphase implementation
  that zero-stuffs by 160 does 160x the necessary multiplies. Evaluating the
  kernel at the output sample's fractional position instead keeps the cost at
  ``K`` taps per output sample regardless of how ugly the ratio is.
* **Exact rational phase accumulation** (integer numerator over ``up``), so a
  one-hour session accumulates zero timing drift.
* **State is carried across chunks.** Resampling each 20 ms chunk independently
  produces a discontinuity at every boundary — an audible tick at 50 Hz, and a
  measurable WER regression. The filter history is retained instead.
* ``src_rate == dst_rate`` is a true passthrough. The common case must not pay
  for a filter it does not need.

There is no third-party dependency here on purpose: ``scipy`` is not in the
runtime dependency set, and pulling it in for one function would add ~90 MB to
an image that already carries CUDA.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from orphus.domain.types import AudioChunk, PcmArray

__all__ = ["StreamingResampler", "resample", "resample_chunk"]

# Half the number of sinc zero-crossings retained on each side at the *narrower*
# of the two rates. 16 gives roughly -80 dB stopband with a Kaiser window, which
# is far below anything the ASR front-end can distinguish, at ~100 taps for the
# worst realistic ratio (48k -> 16k).
_DEFAULT_HALF_LEN: Final[int] = 16

# Kaiser beta for ~-90 dB sidelobes.
_KAISER_BETA: Final[float] = 8.6

# Cap on the phase table when ``up`` is pathological (e.g. a client reporting
# 44_101 Hz). Quantising the fractional position to 1/1024 of an input sample is
# ~20 ns of jitter at 48 kHz -- inaudible, and it bounds the table at a few
# hundred kilobytes instead of megabytes.
_MAX_PHASES: Final[int] = 1024


def _kernel_table(phases: int, half_taps: int, cutoff: float) -> np.ndarray:
    """Precompute one windowed-sinc FIR per fractional output phase.

    Args:
        phases: Number of distinct fractional positions represented.
        half_taps: Taps retained on each side of the kernel centre.
        cutoff: Passband edge in cycles per *input* sample.

    Returns:
        A ``(phases, 2 * half_taps + 1)`` float32 array; row ``q`` filters an
        output sample whose position falls ``q / phases`` of an input sample past
        an input sample boundary.
    """
    taps = 2 * half_taps + 1
    offsets = np.arange(taps, dtype=np.float64)
    fractions = np.arange(phases, dtype=np.float64) / float(phases)
    # Distance, in input samples, from each tap to the output sample position.
    distance = fractions[:, None] + float(half_taps) - offsets[None, :]

    support = float(half_taps) + 1.0
    normalised = np.clip(1.0 - (distance / support) ** 2, 0.0, None)
    window = np.i0(_KAISER_BETA * np.sqrt(normalised)) / np.i0(_KAISER_BETA)

    kernel = np.sinc(2.0 * cutoff * distance) * window
    # Row-normalise to unity DC gain. This also absorbs the 2*cutoff amplitude
    # factor, so no separate gain compensation is needed.
    kernel /= kernel.sum(axis=1, keepdims=True)
    return kernel.astype(np.float32)


class StreamingResampler:
    """Stateful rational resampler for one audio stream.

    One instance belongs to exactly one stream in one direction: the filter
    history it carries is what removes the boundary discontinuity between
    consecutive chunks, so sharing an instance across sessions corrupts both.

    Args:
        src_rate: Input sample rate in Hz.
        dst_rate: Output sample rate in Hz.
        half_len: Sinc zero-crossings retained per side; higher is sharper and
            slower.

    Raises:
        ValueError: If either rate is not positive.
    """

    __slots__ = (
        "_buffer",
        "_down",
        "_dst_rate",
        "_frac",
        "_half_taps",
        "_origin",
        "_passthrough",
        "_phases",
        "_src_rate",
        "_table",
        "_taps",
        "_up",
    )

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        half_len: int = _DEFAULT_HALF_LEN,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"sample rates must be positive, got {src_rate} -> {dst_rate}")
        self._src_rate = src_rate
        self._dst_rate = dst_rate
        self._passthrough = src_rate == dst_rate

        if self._passthrough:
            self._up = 1
            self._down = 1
            self._phases = 1
            self._half_taps = 0
            self._taps = 1
            self._table = np.ones((1, 1), dtype=np.float32)
        else:
            common = math.gcd(src_rate, dst_rate)
            self._up = dst_rate // common
            self._down = src_rate // common
            ratio = dst_rate / src_rate
            # Anti-alias at the lower of the two Nyquist limits, expressed in
            # cycles per input sample.
            cutoff = 0.5 * min(1.0, ratio)
            self._half_taps = math.ceil(half_len / (2.0 * cutoff))
            self._taps = 2 * self._half_taps + 1
            self._phases = self._up if self._up <= _MAX_PHASES else _MAX_PHASES
            self._table = _kernel_table(self._phases, self._half_taps, cutoff)

        self._buffer: PcmArray = np.zeros(0, dtype=np.float32)
        self._origin = 0
        self._frac = 0
        self.reset()

    @property
    def src_rate(self) -> int:
        """Input sample rate in Hz."""
        return self._src_rate

    @property
    def dst_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._dst_rate

    @property
    def is_passthrough(self) -> bool:
        """Whether input and output rates match and no filtering is applied."""
        return self._passthrough

    @property
    def latency_samples(self) -> int:
        """Input samples of algorithmic delay held back by the filter.

        Half the kernel width. At 48 kHz with the default settings this is ~1 ms,
        which is an order of magnitude below the 320 ms ASR lookahead and so is
        not worth trading quality for.
        """
        return self._half_taps

    def reset(self) -> None:
        """Discard filter history. Call between unrelated streams, not chunks."""
        # Prime with a half-kernel of zeros so the very first output sample is
        # centred on input sample 0 rather than lagging by half the kernel.
        self._buffer = np.zeros(self._half_taps, dtype=np.float32)
        self._origin = self._half_taps
        self._frac = 0

    def process(self, samples: PcmArray) -> PcmArray:
        """Resample the next block of a continuous stream.

        Args:
            samples: Mono float32 PCM at ``src_rate``.

        Returns:
            Mono float32 PCM at ``dst_rate``. May be empty when the block is
            shorter than the decimation step; the remainder is retained.
        """
        block = np.asarray(samples, dtype=np.float32)
        if self._passthrough:
            return block
        if block.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {block.shape}")

        buffer = np.concatenate((self._buffer, block)) if block.size else self._buffer

        # Highest output index whose kernel support is fully inside ``buffer``.
        headroom = buffer.size - 1 - self._half_taps - self._origin
        if headroom < 0:
            count = 0
        else:
            # origin + (frac + m*down) // up <= origin + headroom
            #   <=> frac + m*down <= up*(headroom + 1) - 1
            last = (self._up * (headroom + 1) - 1 - self._frac) // self._down
            count = last + 1 if last >= 0 else 0

        if count == 0:
            self._buffer = buffer
            return np.zeros(0, dtype=np.float32)

        steps = np.arange(count, dtype=np.int64) * self._down + self._frac
        centres = self._origin + steps // self._up
        fractions = steps % self._up
        if self._phases == self._up:
            phase_index = fractions
        else:
            phase_index = np.minimum(
                (fractions * self._phases) // self._up, self._phases - 1
            )

        taps = np.arange(self._taps, dtype=np.int64)
        gather = centres[:, None] - self._half_taps + taps[None, :]
        out = np.einsum("ij,ij->i", buffer[gather], self._table[phase_index], dtype=np.float32)

        advanced = self._frac + count * self._down
        self._origin += advanced // self._up
        self._frac = advanced % self._up

        keep_from = self._origin - self._half_taps
        if keep_from > 0:
            # Copy rather than slice so the (potentially large) concatenated
            # block is not kept alive by a view for the life of the session.
            self._buffer = buffer[keep_from:].copy()
            self._origin -= keep_from
        else:
            self._buffer = buffer

        return out.astype(np.float32, copy=False)

    def flush(self) -> PcmArray:
        """Drain the filter tail at end of stream.

        Returns:
            The final output samples, produced by feeding the kernel's
            right-hand support with silence. Empty in passthrough mode.
        """
        if self._passthrough:
            return np.zeros(0, dtype=np.float32)
        return self.process(np.zeros(self._half_taps, dtype=np.float32))


def resample(
    samples: PcmArray,
    src_rate: int,
    dst_rate: int,
    *,
    half_len: int = _DEFAULT_HALF_LEN,
) -> PcmArray:
    """One-shot resample of a complete, self-contained buffer.

    Prefer :class:`StreamingResampler` for live audio: this helper resets the
    filter on every call, which is correct only when the buffer really is the
    whole signal.

    Args:
        samples: Mono float32 PCM at ``src_rate``.
        src_rate: Input sample rate in Hz.
        dst_rate: Output sample rate in Hz.
        half_len: Sinc zero-crossings retained per side.

    Returns:
        Mono float32 PCM at ``dst_rate``, approximately
        ``len(samples) * dst_rate / src_rate`` samples long.
    """
    if src_rate == dst_rate:
        return np.asarray(samples, dtype=np.float32)
    converter = StreamingResampler(src_rate, dst_rate, half_len=half_len)
    head = converter.process(samples)
    tail = converter.flush()
    return np.concatenate((head, tail)) if tail.size else head


def resample_chunk(chunk: AudioChunk, dst_rate: int) -> AudioChunk:
    """Resample a whole :class:`AudioChunk`, preserving its metadata.

    Args:
        chunk: Source audio.
        dst_rate: Target sample rate in Hz.

    Returns:
        A new chunk at ``dst_rate``; the original when the rates already match.
    """
    if chunk.sample_rate == dst_rate:
        return chunk
    return AudioChunk(
        samples=resample(chunk.samples, chunk.sample_rate, dst_rate),
        sample_rate=dst_rate,
        sequence=chunk.sequence,
        captured_at=chunk.captured_at,
    )
