"""Streaming wrapper around the Silero VAD model bundled with faster-whisper."""
from __future__ import annotations

import numpy as np

FRAME_SAMPLES = 512        # 32 ms at 16 kHz, the window Silero expects
CONTEXT_SAMPLES = 64       # tail of the previous frame fed with each new frame
STATE_SHAPE = (1, 1, 128)  # LSTM hidden/cell state shape


class SileroVad:
    """Frame-by-frame speech probability with LSTM state carried between calls."""

    def __init__(self) -> None:
        try:
            from faster_whisper.vad import get_vad_model
        except ImportError as exc:
            raise RuntimeError("faster-whisper (with onnxruntime) is required for VAD") from exc
        self._session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        """Clear the model state, e.g. after a long gap in the audio."""
        self._h = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Return the speech probability (0..1) for one 512-sample 16 kHz frame."""
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"VAD frame must have {FRAME_SAMPLES} samples, got {frame.shape}")
        model_input = np.concatenate([self._context, frame]).astype(np.float32)[None, :]
        probs, self._h, self._c = self._session.run(
            None, {"input": model_input, "h": self._h, "c": self._c}
        )
        self._context = frame[-CONTEXT_SAMPLES:].astype(np.float32)
        return float(np.asarray(probs).reshape(-1)[0])
