"""faster-whisper backend for offline subtitles: model download prompt, language detection,
and the standard transcribe() call (30 s window, VAD, word timestamps, beam search)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from config import TranscribeConfig
from models import approx_size_text, download_size_gb, is_downloaded
from offline_lang import Detect, Span
from offline_state import RawSegment

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SAMPLE_RATE = 16000
BEAM_SIZE = 5
# VAD settings for transcription. The shorter silence (default 2000 ms) stops two close
# utterances being merged into one chunk, where word alignment can put a first word inside
# the pause. Measured on real read speech (FLEURS, 4.3 min):
#  - small/medium with Whisper's default threshold and padding skipped whole utterances (the
#    default VAD clips soft word starts/ends: 7% of speech energy missed). A lower threshold
#    and wider padding keep the edges (<2% missed): word error rate 0.109 -> 0.075.
#  - large-v3 did not skip speech with the default threshold/padding, and the wider padding
#    made it slower (real-time factor 2.7 -> 4.2) and gave it more invented text (0.13 -> 0.24).
VAD_TUNED = dict(min_silence_duration_ms=500, threshold=0.35, speech_pad_ms=800)
VAD_DEFAULT = dict(min_silence_duration_ms=500)


def transcribe_vad(model_name: str) -> dict:
    return VAD_DEFAULT if model_name == "large-v3" else VAD_TUNED
VAD_MIN_SILENCE_MS = 500     # for planning only: pauses used to place language block edges
VAD_PAD_MS = 100


def model_dir() -> Path:
    """Where models are stored (D:\\LiveSub\\models for the standard install)."""
    return TranscribeConfig().model_path


def _confirmed(question: str) -> bool:
    """Ask y/N; no answer available (closed or non-interactive input) counts as no."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def ensure_model(name: str, assume_yes: bool) -> bool:
    """True if the model is on disk or the user agreed to download it."""
    folder = model_dir()
    if is_downloaded(name, folder):
        return True
    size = download_size_gb(name)
    size_text = f"{size:.2f} GB" if size else approx_size_text(name)
    print(f"Model '{name}' is not downloaded yet.\n"
          f"  Download size: {size_text}\n  Saved to:      {folder}")
    if not assume_yes and not _confirmed("Download it now? [y/N] "):
        print("Nothing was downloaded. Answer y at the prompt, or re-run with --yes.")
        return False
    from faster_whisper.utils import download_model

    folder.mkdir(parents=True, exist_ok=True)
    download_model(name, cache_dir=str(folder))
    return True


def _slice(audio: np.ndarray, spans: Sequence[Span]) -> np.ndarray:
    parts = [audio[s * SAMPLE_RATE // 1000: e * SAMPLE_RATE // 1000] for s, e in spans]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class WhisperBackend:
    """A loaded Whisper model (int8, CPU)."""

    def __init__(self, name: str, threads: int = 0) -> None:
        from faster_whisper import WhisperModel

        self.name = name
        self._model = WhisperModel(name, device="cpu", compute_type="int8",
                                   cpu_threads=threads, download_root=str(model_dir()),
                                   local_files_only=True)
        if not self._model.model.is_multilingual:
            raise ValueError(f"Model '{name}' is English-only; use a multilingual model")

    @staticmethod
    def speech_spans(audio: np.ndarray) -> list[Span]:
        """Where there is speech, in ms, found with the Silero VAD that ships with Whisper."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(min_silence_duration_ms=VAD_MIN_SILENCE_MS, speech_pad_ms=VAD_PAD_MS)
        return [(c["start"] * 1000 // SAMPLE_RATE, c["end"] * 1000 // SAMPLE_RATE)
                for c in get_speech_timestamps(audio, options)]

    def detector(self, audio: np.ndarray) -> Detect:
        """Language-probability function over speech spans of `audio`."""
        def detect(spans: Sequence[Span]) -> dict[str, float]:
            _, _, probs = self._model.detect_language(audio=_slice(audio, spans))
            return dict(probs)
        return detect

    def transcribe(self, audio: np.ndarray, language: str, task: str,
                   prompt: str | None, vad: bool = True) -> Iterator[RawSegment]:
        """`vad=False` is for re-checking a stretch already known to be speech."""
        segments, _info = self._model.transcribe(
            audio, language=language, task=task, beam_size=BEAM_SIZE, vad_filter=vad,
            word_timestamps=True, condition_on_previous_text=True, initial_prompt=prompt,
            vad_parameters=transcribe_vad(self.name) if vad else None)
        for seg in segments:
            words = tuple((w.word, w.start, w.end, w.probability) for w in (seg.words or ()))
            yield RawSegment(seg.start, seg.end, seg.text, words)
