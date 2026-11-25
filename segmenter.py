"""Split a live 16 kHz audio stream into short speech segments using VAD.

Silence is dropped here, so the transcriber only ever sees speech.
Segments end at natural pauses (2-4 s by default); continuous speech is cut
at the quietest point near the limit, with a short overlap if no real pause
was found so words at the boundary are not lost.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import VadConfig
from queues import put_drop_oldest
from vad import FRAME_SAMPLES, SileroVad

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE
GET_TIMEOUT_S = 0.1

SpeechProbFn = Callable[[np.ndarray], float]


@dataclass(frozen=True, eq=False)
class Segment:
    """A chunk of speech ready for transcription (times are stream seconds)."""

    audio: np.ndarray
    start_s: float
    end_s: float
    overlap_s: float = 0.0   # leading audio repeated from the previous segment
    reason: str = "pause"    # pause | split | cut | flush
    created_at: float = field(default_factory=time.monotonic)  # when the segment closed

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _frames(seconds: float) -> int:
    return max(0, round(seconds / FRAME_S))


class Segmenter:
    """Frame-level state machine turning audio + speech probabilities into Segments."""

    def __init__(self, cfg: VadConfig, speech_prob: SpeechProbFn) -> None:
        self._cfg = cfg
        self._prob = speech_prob
        self._min_speech = _frames(cfg.min_speech_s)
        self._min_seg = _frames(cfg.min_segment_s)
        self._max_seg = _frames(cfg.max_segment_s)
        self._split_sil = max(1, _frames(cfg.split_silence_s))
        self._end_sil = max(1, _frames(cfg.end_silence_s))
        self._pad = _frames(cfg.pad_s)
        self._overlap = _frames(cfg.overlap_s)
        self._search = max(1, _frames(cfg.cut_search_s))
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, _frames(cfg.pre_roll_s)))
        self._frame_index = 0
        self._reset_segment()

    @property
    def active(self) -> bool:
        """True while a segment is being collected."""
        return self._active

    def feed(self, chunk: np.ndarray) -> list[Segment]:
        """Consume audio of any length; return segments completed by it."""
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32, copy=False)])
        count = len(self._pending) // FRAME_SAMPLES
        done: list[Segment] = []
        for i in range(count):
            frame = self._pending[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            segment = self._step(frame, self._prob(frame))
            if segment is not None:
                done.append(segment)
        self._pending = self._pending[count * FRAME_SAMPLES:]
        return done

    def flush(self) -> Segment | None:
        """Close any open segment (e.g. playback paused) and clear buffers."""
        segment = self._finish("flush") if self._active else None
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        return segment

    # -- state machine -------------------------------------------------

    def _reset_segment(self) -> None:
        self._active = False
        self._seg_frames: list[np.ndarray] = []
        self._seg_probs: list[float] = []
        self._seg_start = 0
        self._seg_overlap = 0
        self._speech_count = 0
        self._silence_run = 0

    def _step(self, frame: np.ndarray, prob: float) -> Segment | None:
        index = self._frame_index
        self._frame_index += 1
        if not self._active:
            if prob < self._cfg.threshold:
                self._preroll.append(frame)
                return None
            self._begin(index, frame, prob)
            return None

        self._seg_frames.append(frame)
        self._seg_probs.append(prob)
        if prob >= self._cfg.neg_threshold:
            self._speech_count += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        length = len(self._seg_frames)
        if self._silence_run >= self._end_sil:
            return self._finish("pause")
        if length >= self._min_seg and self._silence_run >= self._split_sil:
            return self._finish("split")
        if length >= self._max_seg:
            return self._cut()
        return None

    def _begin(self, index: int, frame: np.ndarray, prob: float) -> None:
        pre = list(self._preroll)
        self._preroll.clear()
        self._active = True
        self._seg_frames = [*pre, frame]
        self._seg_probs = [0.0] * len(pre) + [prob]
        self._seg_start = index - len(pre)
        self._seg_overlap = 0
        self._speech_count = 1
        self._silence_run = 0

    def _make_segment(self, frames: list[np.ndarray], reason: str) -> Segment:
        start_s = self._seg_start * FRAME_S
        return Segment(
            audio=np.concatenate(frames),
            start_s=start_s,
            end_s=start_s + len(frames) * FRAME_S,
            overlap_s=self._seg_overlap * FRAME_S,
            reason=reason,
        )

    def _finish(self, reason: str) -> Segment | None:
        """End the segment, trimming trailing silence to the configured pad."""
        keep = len(self._seg_frames) - self._silence_run + min(self._silence_run, self._pad)
        kept, tail = self._seg_frames[:keep], self._seg_frames[keep:]
        enough_speech = self._speech_count >= self._min_speech
        segment = self._make_segment(kept, reason) if enough_speech else None
        self._reset_segment()
        self._preroll.extend(tail)
        return segment

    def _cut(self) -> Segment | None:
        """Hard cut at the quietest frame near the limit; keep the rest active."""
        length = len(self._seg_frames)
        lo = max(self._seg_overlap + 1, length - self._search)
        window = self._seg_probs[lo:length]
        cut = lo + int(np.argmin(window))           # last frame of the emitted segment
        real_pause = window[cut - lo] < self._cfg.neg_threshold
        overlap = 0 if real_pause else min(self._overlap, cut + 1)

        emitted = self._seg_frames[:cut + 1]
        segment = (self._make_segment(emitted, "cut")
                   if self._speech_count >= self._min_speech else None)

        rest_from = cut + 1 - overlap
        rest_frames = self._seg_frames[rest_from:]
        rest_probs = self._seg_probs[rest_from:]
        self._seg_start += rest_from
        self._seg_frames = rest_frames
        self._seg_probs = rest_probs
        self._seg_overlap = overlap
        new_probs = rest_probs[overlap:]
        self._speech_count = sum(p >= self._cfg.neg_threshold for p in new_probs)
        self._silence_run = 0
        for p in reversed(new_probs):
            if p >= self._cfg.neg_threshold:
                break
            self._silence_run += 1
        return segment


class SegmenterWorker:
    """Thread: audio chunk queue -> Segmenter -> segment queue."""

    def __init__(self, in_queue: queue.Queue[np.ndarray], out_queue: queue.Queue[Segment],
                 cfg: VadConfig, sample_rate: int,
                 speech_prob: SpeechProbFn | None = None) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"VAD requires {SAMPLE_RATE} Hz audio, got {sample_rate}")
        self._in = in_queue
        self._out = out_queue
        self._cfg = cfg
        self._vad = speech_prob if speech_prob is not None else SileroVad()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """True while the segmentation thread is alive."""
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        """Start the segmentation thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="Segmenter", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the thread; any open segment is flushed first."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def _emit(self, segment: Segment | None) -> None:
        if segment is None:
            return
        log.debug("segment %.2f-%.2fs (%s)", segment.start_s, segment.end_s, segment.reason)
        if put_drop_oldest(self._out, segment):
            log.warning("Segment queue full; dropped oldest segment (transcriber too slow)")

    def _run(self) -> None:
        segmenter = Segmenter(self._cfg, self._vad)
        last_audio = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._in.get(timeout=GET_TIMEOUT_S)
                except queue.Empty:
                    idle = time.monotonic() - last_audio
                    if segmenter.active and idle > self._cfg.gap_flush_s:
                        self._emit(segmenter.flush())
                        if isinstance(self._vad, SileroVad):
                            self._vad.reset()
                    continue
                last_audio = time.monotonic()
                for segment in segmenter.feed(chunk):
                    self._emit(segment)
            self._emit(segmenter.flush())
        except Exception:
            log.exception("Segmenter crashed")
        log.info("Segmenter stopped")


# -- self-test -----------------------------------------------------------

def _summarize(segments: list[Segment], total_s: float) -> None:
    if not segments:
        log.info("No speech segments found in %.1fs of audio", total_s)
        return
    durations = [s.duration_s for s in segments]
    speech = sum(durations) - sum(s.overlap_s for s in segments)
    log.info("%d segments | duration min %.2fs avg %.2fs max %.2fs | "
             "%.1fs of %.1fs audio sent to transcription (%.0f%% skipped)",
             len(segments), min(durations), sum(durations) / len(durations),
             max(durations), speech, total_s, 100 * (1 - speech / max(total_s, 1e-9)))


def _log_segment(n: int, s: Segment) -> None:
    log.info("#%03d  %7.2f-%7.2fs  %.2fs  overlap %.2fs  end=%s",
             n, s.start_s, s.end_s, s.duration_s, s.overlap_s, s.reason)


def _run_file(path: str, cfg: VadConfig, out_dir: str | None) -> list[Segment]:
    from audio_dsp import StreamResampler, read_wav, write_wav

    audio, rate = read_wav(path)
    audio = StreamResampler(rate, SAMPLE_RATE).process(audio)
    segmenter = Segmenter(cfg, SileroVad())
    step = SAMPLE_RATE // 10
    started = time.perf_counter()
    segments: list[Segment] = []
    for i in range(0, len(audio), step):
        segments.extend(segmenter.feed(audio[i:i + step]))
    tail = segmenter.flush()
    segments.extend([tail] if tail else [])
    elapsed = time.perf_counter() - started
    for n, s in enumerate(segments, 1):
        _log_segment(n, s)
        if out_dir:
            write_wav(f"{out_dir}/seg_{n:03d}.wav", s.audio, SAMPLE_RATE)
    _summarize(segments, len(audio) / SAMPLE_RATE)
    log.info("VAD processing took %.2fs for %.1fs of audio",
             elapsed, len(audio) / SAMPLE_RATE)
    return segments


def _run_live(seconds: float, out_dir: str | None) -> None:
    from audio_dsp import write_wav
    from capture import AudioCapture, make_audio_queue
    from config import load_config

    cfg = load_config()
    audio_q = make_audio_queue(cfg.audio)
    seg_q: queue.Queue[Segment] = queue.Queue(maxsize=50)
    capture = AudioCapture(audio_q, cfg.audio)
    worker = SegmenterWorker(audio_q, seg_q, cfg.vad, cfg.audio.sample_rate)
    capture.start()
    worker.start()
    segments: list[Segment] = []
    end = time.monotonic() + seconds
    log.info("Listening for %.0fs - play something with speech", seconds)
    try:
        while time.monotonic() < end or not seg_q.empty():
            if time.monotonic() >= end and capture.running:
                capture.stop()
                worker.stop()
            try:
                segment = seg_q.get(timeout=0.2)
            except queue.Empty:
                continue
            segments.append(segment)
            _log_segment(len(segments), segment)
            if out_dir:
                write_wav(f"{out_dir}/seg_{len(segments):03d}.wav", segment.audio, SAMPLE_RATE)
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        capture.stop()
        worker.stop()
    speech = sum(s.duration_s for s in segments)
    log.info("%d segments, %.1fs of speech in %.0fs", len(segments), speech, seconds)


def _prepare_out_dir(out_dir: str) -> None:
    from pathlib import Path

    folder = Path(out_dir)
    folder.mkdir(exist_ok=True)
    for old in folder.glob("seg_*.wav"):
        old.unlink()


if __name__ == "__main__":
    import argparse

    from config import load_config

    parser = argparse.ArgumentParser(description="Test speech segmentation")
    parser.add_argument("--file", help="16-bit WAV to segment instead of live audio")
    parser.add_argument("--seconds", type=float, default=30.0, help="live capture length")
    parser.add_argument("--out", default="segments_test", help="folder for segment WAVs")
    parser.add_argument("--no-save", action="store_true", help="do not write segment WAVs")
    args = parser.parse_args()
    from logsetup import setup_logging

    setup_logging()
    out_dir = None if args.no_save else args.out
    if out_dir:
        _prepare_out_dir(out_dir)
    if args.file:
        _run_file(args.file, load_config().vad, out_dir)
    else:
        _run_live(args.seconds, out_dir)
