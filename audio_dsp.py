"""Small numpy-only audio helpers: downmix, streaming resampling, WAV I/O."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

PCM16_SCALE = 32767

FIR_TAPS = 63            # low-pass filter length (odd, symmetric)
CUTOFF_MARGIN = 0.9      # keep the cutoff slightly below the target Nyquist


def to_mono(interleaved: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved multi-channel samples into one float32 channel."""
    if channels <= 1:
        return interleaved.astype(np.float32, copy=False)
    usable = len(interleaved) - len(interleaved) % channels
    frames = interleaved[:usable].reshape(-1, channels)
    return frames.mean(axis=1, dtype=np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save mono float32 audio (-1..1) as a 16-bit PCM WAV file."""
    pcm = (np.clip(audio, -1.0, 1.0) * PCM16_SCALE).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a 16-bit PCM WAV as mono float32; returns (audio, sample_rate)."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        channels = wav.getnchannels()
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return to_mono(pcm.astype(np.float32) / PCM16_SCALE, channels), rate


def design_lowpass(src_rate: int, dst_rate: int, taps: int = FIR_TAPS) -> np.ndarray:
    """Windowed-sinc anti-aliasing filter for downsampling src_rate -> dst_rate."""
    cutoff = CUTOFF_MARGIN * (dst_rate / 2) / src_rate  # cycles per input sample
    n = np.arange(taps) - (taps - 1) / 2
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
    return (kernel / kernel.sum()).astype(np.float32)


class StreamResampler:
    """Resample a continuous stream chunk by chunk without boundary clicks.

    Applies a low-pass FIR (when downsampling) and then linear interpolation,
    carrying filter and phase state between calls.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("Sample rates must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._step = src_rate / dst_rate
        self._fir = design_lowpass(src_rate, dst_rate) if src_rate > dst_rate else None
        self._fir_state = np.zeros(FIR_TAPS - 1, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)
        self._pos = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return resampled float32 samples for this chunk."""
        samples = samples.astype(np.float32, copy=False)
        if self.src_rate == self.dst_rate or samples.size == 0:
            return samples

        if self._fir is not None:
            buf = np.concatenate([self._fir_state, samples])
            filtered = np.convolve(buf, self._fir, mode="valid").astype(np.float32)
            self._fir_state = buf[-(FIR_TAPS - 1):]
        else:
            filtered = samples

        data = np.concatenate([self._carry, filtered])
        last = len(data) - 1
        if last < self._pos:
            self._carry = data
            return np.zeros(0, dtype=np.float32)

        n_out = int((last - self._pos) // self._step) + 1
        positions = self._pos + self._step * np.arange(n_out)
        out = np.interp(positions, np.arange(len(data)), data).astype(np.float32)

        next_pos = self._pos + self._step * n_out
        drop = min(int(next_pos), len(data))
        self._carry = data[drop:]
        self._pos = next_pos - drop
        return out
