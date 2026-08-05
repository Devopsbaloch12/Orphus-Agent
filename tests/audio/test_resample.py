"""Resampler correctness: rate, amplitude, anti-aliasing, and stream continuity."""

from __future__ import annotations

import numpy as np
import pytest

from orphus.audio.resample import StreamingResampler, resample, resample_chunk
from orphus.domain.types import AudioChunk


def _tone(freq_hz: float, rate: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _dominant_freq(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(np.argmax(spectrum))])


def _peak(samples: np.ndarray, skip: int = 512) -> float:
    """Peak amplitude ignoring the filter's transient edges."""
    return float(np.max(np.abs(samples[skip:-skip])))


@pytest.mark.parametrize(("src", "dst"), [(48_000, 16_000), (8_000, 16_000), (24_000, 16_000)])
def test_output_length_matches_ratio(src: int, dst: int) -> None:
    out = resample(_tone(440.0, src, 1.0), src, dst)
    assert abs(out.size - dst) <= 1


def test_equal_rates_are_a_passthrough() -> None:
    samples = _tone(440.0, 16_000, 0.1)
    assert resample(samples, 16_000, 16_000) is samples
    assert StreamingResampler(16_000, 16_000).is_passthrough


@pytest.mark.parametrize(("src", "dst"), [(48_000, 16_000), (8_000, 16_000), (16_000, 24_000)])
def test_tone_survives_conversion(src: int, dst: int) -> None:
    out = resample(_tone(1_000.0, src, 0.5), src, dst)
    assert _dominant_freq(out, dst) == pytest.approx(1_000.0, abs=15.0)
    assert _peak(out) == pytest.approx(0.5, abs=0.02)


def test_downsampling_rejects_content_above_the_new_nyquist() -> None:
    # 6 kHz cannot exist at 8 kHz output (4 kHz Nyquist). Without an anti-alias
    # filter it would fold down to 2 kHz at nearly full amplitude.
    aliased = resample(_tone(6_000.0, 16_000, 0.5), 16_000, 8_000)
    assert _peak(aliased) < 0.01


def test_streaming_in_chunks_matches_one_shot_exactly() -> None:
    src, dst = 48_000, 16_000
    signal = _tone(700.0, src, 0.5)
    reference = resample(signal, src, dst)

    converter = StreamingResampler(src, dst)
    pieces = [
        converter.process(signal[start : start + 960])
        for start in range(0, signal.size, 960)
    ]
    pieces.append(converter.flush())
    streamed = np.concatenate(pieces)

    assert streamed.size == reference.size
    np.testing.assert_array_equal(streamed, reference)


def test_no_discontinuity_at_chunk_boundaries() -> None:
    """A stateless per-chunk resampler ticks audibly here; a stateful one does not."""
    src, dst = 48_000, 16_000
    signal = _tone(300.0, src, 0.25)
    converter = StreamingResampler(src, dst)
    out = np.concatenate(
        [converter.process(signal[start : start + 480]) for start in range(0, signal.size, 480)]
    )
    # A 300 Hz sine at 16 kHz steps by at most 2*pi*300/16000 * 0.5 ~= 0.06 per sample.
    assert float(np.max(np.abs(np.diff(out[256:-256])))) < 0.08


def test_partial_block_is_retained_not_dropped() -> None:
    converter = StreamingResampler(48_000, 16_000)
    assert converter.process(np.zeros(1, dtype=np.float32)).size == 0
    total = converter.process(_tone(440.0, 48_000, 0.1)).size + converter.flush().size
    assert abs(total - 1600) <= 2


def test_reset_clears_history() -> None:
    converter = StreamingResampler(48_000, 16_000)
    converter.process(_tone(440.0, 48_000, 0.1))
    converter.reset()
    fresh = StreamingResampler(48_000, 16_000)
    signal = _tone(440.0, 48_000, 0.1)
    np.testing.assert_array_equal(converter.process(signal), fresh.process(signal))


def test_awkward_ratio_is_handled() -> None:
    out = resample(_tone(1_000.0, 44_100, 0.5), 44_100, 16_000)
    assert abs(out.size - 8_000) <= 2
    assert _dominant_freq(out, 16_000) == pytest.approx(1_000.0, abs=20.0)


def test_resample_chunk_preserves_metadata() -> None:
    source = AudioChunk(
        samples=_tone(440.0, 48_000, 0.1), sample_rate=48_000, sequence=7, captured_at=123.5
    )
    out = resample_chunk(source, 16_000)
    assert out.sample_rate == 16_000
    assert out.sequence == 7
    assert out.captured_at == 123.5
    assert resample_chunk(source, 48_000) is source


def test_invalid_rates_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        StreamingResampler(0, 16_000)
