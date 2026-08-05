"""PCM representation conversions for the wire edge.

Everything inside the pipeline is mono ``float32`` in ``[-1.0, 1.0]``; browsers,
telephony gateways, and the WebSocket API all speak little-endian ``s16le``.
This module is the only place that boundary is crossed, so the endianness and
the scale factors are asserted in exactly one spot.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from orphus.domain.types import AudioEncoding, PcmArray

__all__ = [
    "decode_pcm",
    "downmix_to_mono",
    "encode_pcm",
    "float32_to_pcm16",
    "float32_to_pcm16_bytes",
    "pcm16_to_float32",
    "rms_dbfs",
]

# PCM16 has one more negative value than positive value. Decode against 32768;
# encoding uses the full negative range for in-range samples and the safe
# positive range otherwise. Inputs outside [-1, 1] are clipped to +/-32767 so
# malformed audio can never wrap around at the wire boundary.
_DECODE_SCALE: Final[float] = 1.0 / 32768.0
_ENCODE_SCALE: Final[float] = 32767.0

# Explicit byte order. ``np.int16`` would follow host endianness, which is a
# silent data-corruption bug on any big-endian deployment target.
_S16LE: Final[np.dtype[np.int16]] = np.dtype("<i2")
_F32LE: Final[np.dtype[np.float32]] = np.dtype("<f4")

_SILENCE_DBFS: Final[float] = float("-inf")


def pcm16_to_float32(data: bytes | bytearray | memoryview | np.ndarray) -> PcmArray:
    """Decode little-endian signed 16-bit PCM to normalised float32.

    Args:
        data: Raw ``s16le`` bytes, or an already-parsed int16 array.

    Returns:
        A fresh, writable float32 array in ``[-1.0, 1.0)``.

    Raises:
        ValueError: If a byte buffer's length is not a multiple of 2.
    """
    if isinstance(data, np.ndarray):
        samples = data.astype(np.int16, copy=False)
    else:
        raw = bytes(data)
        if len(raw) % _S16LE.itemsize != 0:
            raise ValueError(
                f"s16le buffer length {len(raw)} is not a multiple of {_S16LE.itemsize}; "
                "a frame was truncated in transit"
            )
        samples = np.frombuffer(raw, dtype=_S16LE)
    return np.multiply(samples, _DECODE_SCALE, dtype=np.float32)


def float32_to_pcm16(samples: PcmArray) -> np.ndarray:
    """Quantise normalised float32 to little-endian int16.

    Out-of-range samples are clipped rather than wrapped: a wrapped sample is an
    audible full-scale click, a clipped one is not.

    Args:
        samples: Mono float32 PCM.

    Returns:
        An int16 array with explicit little-endian byte order.
    """
    scaled = np.clip(samples, -1.0, 1.0) * 32768.0
    # Saturate both malformed out-of-range input and the unrepresentable
    # positive endpoint. Keep an exact in-range -1.0 as -32768.
    scaled = np.where(samples < -1.0, -_ENCODE_SCALE, scaled)
    scaled = np.minimum(scaled, _ENCODE_SCALE)
    return np.rint(scaled).astype(_S16LE)


def float32_to_pcm16_bytes(samples: PcmArray) -> bytes:
    """Quantise normalised float32 and serialise to ``s16le`` bytes.

    Args:
        samples: Mono float32 PCM.

    Returns:
        Little-endian 16-bit PCM bytes.
    """
    return float32_to_pcm16(samples).tobytes()


def decode_pcm(data: bytes | bytearray | memoryview, encoding: AudioEncoding) -> PcmArray:
    """Decode a wire buffer in the given encoding to normalised float32.

    Args:
        data: Raw PCM bytes as received from the client.
        encoding: Declared wire encoding.

    Returns:
        A fresh, writable mono float32 array.

    Raises:
        ValueError: If the buffer length does not divide evenly by the sample
            width for ``encoding``.
    """
    if encoding is AudioEncoding.PCM_S16LE:
        return pcm16_to_float32(data)
    raw = bytes(data)
    if len(raw) % _F32LE.itemsize != 0:
        raise ValueError(
            f"f32le buffer length {len(raw)} is not a multiple of {_F32LE.itemsize}; "
            "a frame was truncated in transit"
        )
    # ``astype`` (not ``view``) because the result must be host-order, writable,
    # and independent of the caller's buffer.
    return np.frombuffer(raw, dtype=_F32LE).astype(np.float32)


def encode_pcm(samples: PcmArray, encoding: AudioEncoding) -> bytes:
    """Serialise normalised float32 PCM to the given wire encoding.

    Args:
        samples: Mono float32 PCM.
        encoding: Target wire encoding.

    Returns:
        Bytes ready to put on the wire.
    """
    if encoding is AudioEncoding.PCM_S16LE:
        return float32_to_pcm16_bytes(samples)
    return samples.astype(_F32LE, copy=False).tobytes()


def downmix_to_mono(samples: PcmArray, channels: int) -> PcmArray:
    """Average interleaved multi-channel PCM down to mono.

    Every model in the pipeline is mono; downmixing at the edge is cheaper than
    carrying a channel dimension through the ring buffers and the VAD framer.

    Args:
        samples: Interleaved float32 PCM, ``channels`` samples per frame.
        channels: Number of interleaved channels. ``1`` returns the input.

    Returns:
        Mono float32 PCM.

    Raises:
        ValueError: If ``channels`` is below 1 or does not divide the length.
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    if channels == 1:
        return samples
    if samples.size % channels != 0:
        raise ValueError(
            f"interleaved buffer of {samples.size} samples does not divide into {channels} channels"
        )
    return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)


def rms_dbfs(samples: PcmArray) -> float:
    """Root-mean-square level of a buffer in dBFS.

    Used by the energy-threshold test double and by diagnostics that need a
    cheap "is anything actually arriving" signal without running the VAD.

    Args:
        samples: Mono float32 PCM.

    Returns:
        Level in dBFS, or ``-inf`` for digital silence and empty buffers.
    """
    if samples.size == 0:
        return _SILENCE_DBFS
    mean_square = float(np.mean(np.square(samples, dtype=np.float64)))
    if mean_square <= 0.0:
        return _SILENCE_DBFS
    return 10.0 * float(np.log10(mean_square))
