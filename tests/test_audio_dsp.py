"""Tests for channel downmix and streaming resampling."""
import numpy as np
import pytest

from audio_dsp import StreamResampler, to_mono


def tone(freq: float, rate: int, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spectrum)])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_to_mono_averages_stereo_frames():
    interleaved = np.array([1.0, 0.0, 0.5, 0.5, -1.0, 1.0], dtype=np.float32)
    assert np.allclose(to_mono(interleaved, 2), [0.5, 0.5, 0.0])


def test_to_mono_passes_through_single_channel():
    x = np.array([0.1, 0.2], dtype=np.float32)
    assert np.array_equal(to_mono(x, 1), x)


def test_to_mono_ignores_partial_trailing_frame():
    x = np.array([1.0, 1.0, 0.5], dtype=np.float32)
    assert np.allclose(to_mono(x, 2), [1.0])


@pytest.mark.parametrize("src_rate", [48000, 44100, 16000, 8000])
def test_resampler_output_length_matches_ratio(src_rate):
    resampler = StreamResampler(src_rate, 16000)
    out = resampler.process(tone(440, src_rate, 2.0))
    assert out.dtype == np.float32
    assert abs(len(out) - 32000) <= 2


@pytest.mark.parametrize("src_rate", [48000, 44100])
def test_resampler_preserves_speech_band_tone(src_rate):
    out = StreamResampler(src_rate, 16000).process(tone(1000, src_rate, 1.0))
    assert abs(dominant_freq(out[200:], 16000) - 1000) < 5
    assert rms(out[200:]) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)


def test_resampler_suppresses_content_above_target_nyquist():
    # 12 kHz cannot be represented at 16 kHz; without filtering it aliases to 4 kHz
    out = StreamResampler(48000, 16000).process(tone(12000, 48000, 1.0))
    assert rms(out[200:]) < 0.01


@pytest.mark.parametrize("src_rate,chunk", [(48000, 4800), (44100, 4410), (48000, 997)])
def test_streaming_matches_single_pass(src_rate, chunk):
    signal = tone(523, src_rate, 1.5)
    whole = StreamResampler(src_rate, 16000).process(signal)
    streamer = StreamResampler(src_rate, 16000)
    pieces = [streamer.process(signal[i:i + chunk]) for i in range(0, len(signal), chunk)]
    streamed = np.concatenate(pieces)
    n = min(len(whole), len(streamed))
    assert abs(len(whole) - len(streamed)) <= 1
    assert np.allclose(whole[:n], streamed[:n], atol=1e-5)


def test_resampler_handles_empty_chunk():
    assert StreamResampler(48000, 16000).process(np.zeros(0, dtype=np.float32)).size == 0


def test_resampler_rejects_invalid_rates():
    with pytest.raises(ValueError):
        StreamResampler(0, 16000)
