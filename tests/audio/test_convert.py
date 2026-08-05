"""PCM conversion correctness, including the byte-order contract."""

from __future__ import annotations

import numpy as np
import pytest

from orphus.audio.convert import (
    decode_pcm,
    downmix_to_mono,
    encode_pcm,
    float32_to_pcm16,
    float32_to_pcm16_bytes,
    pcm16_to_float32,
    rms_dbfs,
)
from orphus.domain.types import AudioEncoding


def test_decode_is_little_endian() -> None:
    # 0x8000 little-endian is the most negative int16 -> exactly -1.0.
    assert pcm16_to_float32(b"\x00\x80")[0] == pytest.approx(-1.0)
    # 0xFF7F little-endian is the most positive int16 -> just under +1.0.
    assert pcm16_to_float32(b"\xff\x7f")[0] == pytest.approx(32767 / 32768)


def test_decode_returns_writable_float32() -> None:
    decoded = pcm16_to_float32(b"\x00\x00\x01\x00")
    assert decoded.dtype == np.float32
    decoded[0] = 0.5  # must not raise: np.frombuffer views are read-only


def test_roundtrip_within_one_lsb() -> None:
    original = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    restored = pcm16_to_float32(float32_to_pcm16_bytes(original))
    assert np.max(np.abs(restored - original)) < 1.0 / 32767


def test_encode_clips_rather_than_wraps() -> None:
    encoded = float32_to_pcm16(np.array([2.0, -2.0], dtype=np.float32))
    assert encoded.tolist() == [32767, -32767]


def test_odd_length_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        pcm16_to_float32(b"\x00\x80\x00")


def test_f32le_roundtrip_via_encoding_enum() -> None:
    original = np.array([0.0, 0.25, -0.75], dtype=np.float32)
    raw = encode_pcm(original, AudioEncoding.PCM_F32LE)
    assert np.array_equal(decode_pcm(raw, AudioEncoding.PCM_F32LE), original)


def test_f32le_truncated_buffer_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a multiple"):
        decode_pcm(b"\x00\x00\x00", AudioEncoding.PCM_F32LE)


def test_downmix_averages_channels() -> None:
    interleaved = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)
    assert downmix_to_mono(interleaved, 2).tolist() == [0.5, 0.0]


def test_downmix_mono_is_identity() -> None:
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert downmix_to_mono(mono, 1) is mono


def test_downmix_rejects_ragged_buffer() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        downmix_to_mono(np.zeros(5, dtype=np.float32), 2)


def test_rms_dbfs_levels() -> None:
    assert rms_dbfs(np.zeros(512, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.zeros(0, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.ones(512, dtype=np.float32)) == pytest.approx(0.0)
    assert rms_dbfs(np.full(512, 0.1, dtype=np.float32)) == pytest.approx(-20.0, abs=0.01)
