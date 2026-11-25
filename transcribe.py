"""Speech-to-text worker built on faster-whisper (CPU, int8).

Segments are decoded with a short encoder window (6 s instead of Whisper's
fixed 30 s), which is about 4x faster on CPU for 1-4 s clips. Language is
detected from the same encoder pass and smoothed across segments.
"""
from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import numpy as np

from config import MODES, TranscribeConfig
from queues import put_drop_oldest
from segmenter import SAMPLE_RATE, Segment
from textproc import (MAX_COMPRESSION_RATIO, collapse_repeats, compression_ratio,
                      echoes_context, is_hallucination, remove_overlap, strip_broken)

log = logging.getLogger(__name__)

# Windows without Developer Mode cannot symlink; the cache still works, so hide the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FRAMES_PER_SECOND = 100       # Whisper mel frames per second
MAX_WINDOW_S = 30.0
PROMPT_MAX_TOKENS = 64        # previous-text context passed to the decoder
MAX_TOKENS_BASE = 12          # output length cap: base + per-second allowance
MAX_TOKENS_PER_S = 12
WHISPER_MAX_LENGTH = 448
GAP_BETWEEN_MERGED_S = 0.3    # silence inserted between merged, non-adjacent segments
SWITCH_AFTER_DISAGREEMENTS = 2   # long segments in a row that reject the current language
CURRENT_LANGUAGE_GONE = 0.1      # current language's probability below this = rejected
GET_TIMEOUT_S = 0.2


@dataclass(frozen=True)
class Recognition:
    """Raw engine output for one audio clip."""

    text: str
    language: str
    language_prob: float
    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True)
class Subtitle:
    """A finished line of text for display."""

    text: str
    language: str
    mode: str
    start_s: float
    end_s: float
    latency_s: float   # from the end of speech to text ready
    compute_s: float


class Engine(Protocol):
    def recognize(self, audio: np.ndarray, task: str, prompt: str | None) -> Recognition: ...


def whisper_languages() -> tuple[str, ...]:
    """Language codes Whisper understands (e.g. 'en', 'bn', 'zh')."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    return tuple(_LANGUAGE_CODES)


def parse_language(value: str) -> str:
    """Validate a --language value; returns '' for auto-detect."""
    code = value.strip().lower()
    if code in ("", "auto"):
        return ""
    if code not in whisper_languages():
        raise ValueError(f"unknown language code '{value}' (examples: bn, hi, en, zh, ja, "
                         "or 'auto')")
    return code


def apply_overrides(cfg: TranscribeConfig, mode: str | None = None, model: str | None = None,
                    language: str | None = None) -> TranscribeConfig:
    """Return cfg with command-line overrides applied (None = keep the configured value)."""
    changes: dict[str, str] = {}
    if mode:
        changes["mode"] = mode
    if model:
        changes["model"] = model
    if language is not None:
        changes["language"] = language
    return replace(cfg, **changes)


class LanguageTracker:
    """Pick the language per segment without trusting short or unsure detections.

    Reliable detections (long enough, confident enough) are blended into a
    running score; other segments reuse the language established so far.
    """

    def __init__(self, forced: str, weight: float, min_duration_s: float = 0.0,
                 min_confidence: float = 0.0) -> None:
        self._forced = forced.strip().lower()
        self._weight = weight
        self._min_duration = min_duration_s
        self._min_confidence = min_confidence
        self._scores: dict[str, float] = {}
        self._disagreements = 0

    @property
    def current(self) -> str | None:
        """Language established so far (None until a reliable detection)."""
        if self._forced:
            return self._forced
        return max(self._scores, key=self._scores.__getitem__) if self._scores else None

    def set_forced(self, language: str) -> None:
        """Force a language ('' = auto-detect); detection history starts over."""
        self._forced = language.strip().lower()
        self._scores = {}
        self._disagreements = 0

    def update(self, probs: dict[str, float], duration_s: float) -> tuple[str, float]:
        """Fold in one segment's probabilities; return (language, probability)."""
        if self._forced:
            return self._forced, probs.get(self._forced, 0.0)
        top_lang, top_prob = max(probs.items(), key=lambda kv: kv[1], default=("en", 0.0))
        long_enough = duration_s >= self._min_duration
        current = self.current
        if (long_enough and current and top_lang != current
                and probs.get(current, 0.0) < CURRENT_LANGUAGE_GONE):
            self._disagreements += 1
        elif long_enough:
            self._disagreements = 0
        if self._disagreements >= SWITCH_AFTER_DISAGREEMENTS:
            # the audio clearly changed language (e.g. a new video): start over
            self._scores = {}
            self._disagreements = 0
            reliable = True
        else:
            reliable = long_enough and top_prob >= self._min_confidence
        if not reliable:
            if self._scores:
                best = self.current
                return best, self._scores[best]
            return top_lang, top_prob  # nothing established yet: best guess, not remembered
        # less certain segments move the estimate less
        weight = self._weight * top_prob if self._scores else 1.0
        langs = set(self._scores) | set(probs)
        scores = {lang: weight * probs.get(lang, 0.0) + (1 - weight) * self._scores.get(lang, 0.0)
                  for lang in langs}
        self._scores = {lang: p for lang, p in scores.items() if p > 1e-4}
        best = max(self._scores, key=self._scores.__getitem__)
        return best, self._scores[best]


class WhisperEngine:
    """faster-whisper model driven directly through CTranslate2 with a short window."""

    def __init__(self, cfg: TranscribeConfig) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        if cfg.language and cfg.language not in whisper_languages():
            raise ValueError(f"Unknown language code '{cfg.language}' in config")
        cfg.model_path.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        options = dict(device="cpu", compute_type=cfg.compute_type,
                       cpu_threads=cfg.cpu_threads, download_root=str(cfg.model_path))
        try:
            self._whisper = WhisperModel(cfg.model, local_files_only=True, **options)
        except Exception:
            log.info("Downloading Whisper model '%s' (first run only)...", cfg.model)
            self._whisper = WhisperModel(cfg.model, **options)
        if not self._whisper.model.is_multilingual:
            raise ValueError(f"Model '{cfg.model}' is English-only; use a multilingual model")
        log.info("Loaded Whisper '%s' (%s, %d threads) in %.1fs", cfg.model, cfg.compute_type,
                 cfg.cpu_threads, time.perf_counter() - started)
        self._cfg = cfg
        self._storage = ctranslate2.StorageView
        self._tokenizer_cls = Tokenizer
        base_tok = Tokenizer(self._whisper.hf_tokenizer, True, task="transcribe", language="en")
        self._suppress = get_suppressed_tokens(base_tok, [-1])
        self._languages = LanguageTracker(cfg.language, cfg.language_smoothing,
                                          cfg.language_min_duration_s,
                                          cfg.language_min_confidence)

    def _encode(self, audio: np.ndarray):
        duration = len(audio) / SAMPLE_RATE
        window = min(MAX_WINDOW_S, max(self._cfg.min_window_s,
                                       math.ceil(duration + self._cfg.window_margin_s)))
        samples = int(window * SAMPLE_RATE)
        padded = np.pad(audio[:samples], (0, max(0, samples - len(audio))))
        features = self._whisper.feature_extractor(padded)[:, : int(window * FRAMES_PER_SECOND)]
        batch = self._storage.from_array(np.ascontiguousarray(features[None], dtype=np.float32))
        return self._whisper.model.encode(batch, to_cpu=False)

    def set_language(self, language: str) -> None:
        """Force a language code, or '' for auto-detect (history is reset)."""
        if language and language not in whisper_languages():
            raise ValueError(f"Unknown language code '{language}'")
        self._languages.set_forced(language)

    def recognize(self, audio: np.ndarray, task: str, prompt: str | None) -> Recognition:
        """Transcribe or translate one clip (16 kHz mono float32)."""
        encoded = self._encode(audio)
        raw = self._whisper.model.detect_language(encoded)[0]
        language, lang_prob = self._languages.update({tok[2:-2]: p for tok, p in raw},
                                                     len(audio) / SAMPLE_RATE)
        if task == "translate" and language == "en":
            task = "transcribe"  # English needs no translation
        tok = self._tokenizer_cls(self._whisper.hf_tokenizer, True, task=task, language=language)

        new_tokens = int(MAX_TOKENS_BASE + MAX_TOKENS_PER_S * len(audio) / SAMPLE_RATE)
        prefix: list[int] = []
        if prompt:
            prefix = [tok.sot_prev] + tok.encode(" " + prompt.strip())[-PROMPT_MAX_TOKENS:]
        text, logprob, no_speech = self._generate(encoded, tok, prefix, new_tokens)
        if prompt and echoes_context(prompt, text):
            log.debug("Output repeated the context; decoding again without it")
            text, logprob, no_speech = self._generate(encoded, tok, [], new_tokens)
        return Recognition(text=text, language=language, language_prob=lang_prob,
                           avg_logprob=logprob, no_speech_prob=no_speech)

    def _generate(self, encoded, tok, prefix: list[int],
                  new_tokens: int) -> tuple[str, float, float]:
        """Greedy/beam decode; returns (text, avg log-prob, no-speech prob)."""
        start = prefix + list(tok.sot_sequence) + [tok.no_timestamps]
        result = self._whisper.model.generate(
            encoded, [start], beam_size=self._cfg.beam_size,
            max_length=min(WHISPER_MAX_LENGTH, len(start) + new_tokens),
            return_scores=True, return_no_speech_prob=True,
            suppress_blank=True, suppress_tokens=self._suppress,
        )[0]
        ids = [i for i in result.sequences_ids[0] if i < tok.eot]
        return tok.decode(ids).strip(), float(result.scores[0]), float(result.no_speech_prob)


def merge_segments(segments: list[Segment]) -> np.ndarray:
    """Join queued segments into one clip, removing overlaps between neighbours."""
    parts: list[np.ndarray] = []
    gap = np.zeros(int(GAP_BETWEEN_MERGED_S * SAMPLE_RATE), dtype=np.float32)
    for i, seg in enumerate(segments):
        audio = seg.audio
        if i > 0:
            prev = segments[i - 1]
            if seg.overlap_s > 0 and abs(prev.end_s - seg.start_s - seg.overlap_s) < 1e-3:
                audio = audio[int(round(seg.overlap_s * SAMPLE_RATE)):]
            elif seg.start_s > prev.end_s + 1e-3:
                parts.append(gap)
        parts.append(audio)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def batch_segments(segments: list[Segment], max_s: float) -> list[list[Segment]]:
    """Group consecutive segments into batches no longer than max_s."""
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    length = 0.0
    for seg in segments:
        if current and length + seg.duration_s > max_s:
            batches.append(current)
            current, length = [], 0.0
        current.append(seg)
        length += seg.duration_s
    if current:
        batches.append(current)
    return batches


class Transcriber:
    """Worker thread: Segment queue -> Whisper -> Subtitle queue."""

    def __init__(self, in_queue: queue.Queue[Segment], out_queue: queue.Queue[Subtitle],
                 cfg: TranscribeConfig,
                 engine_factory: Callable[[TranscribeConfig], Engine] = WhisperEngine,
                 text_filter: Callable[[str], str] | None = None) -> None:
        self._in = in_queue
        self._out = out_queue
        self._cfg = cfg
        self._factory = engine_factory
        self._mode = cfg.mode
        self._language = cfg.language        # '' = auto-detect
        self._language_dirty = False          # applied to the engine before the next segment
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: Subtitle | None = None
        self.last_output = 0.0               # monotonic time of the last subtitle
        self._last_raw = ""              # unmasked text, used as Whisper context
        self._last_spoken_end = 0.0      # monotonic time the last subtitled speech ended
        self._filter = text_filter or (lambda text: text)   # e.g. profanity masking
        self.ready = threading.Event()   # set once the model loaded (or failed)
        self.error: str | None = None
        self.processed = 0               # batches handled so far (for tests/diagnostics)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch between 'original' and 'english'; applies to the next segment."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            self._mode = mode
        log.info("Subtitle mode: %s", mode)

    def toggle_mode(self) -> str:
        """Flip the mode and return the new one."""
        new = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self.set_mode(new)
        return new

    @property
    def language(self) -> str:
        """Selected language code, '' for auto-detect."""
        with self._lock:
            return self._language

    def set_language(self, language: str) -> None:
        """Force a language code or '' / 'auto' to detect; applies to the next segment."""
        code = parse_language(language)
        with self._lock:
            self._language = code
            self._language_dirty = True
        log.info("Subtitle language: %s", code or "auto-detect")

    def _apply_language(self, engine: Engine) -> None:
        """Hand a changed language selection to the engine and drop stale context."""
        with self._lock:
            if not self._language_dirty:
                return
            self._language_dirty = False
            code = self._language
        setter = getattr(engine, "set_language", None)
        if setter:
            setter(code)
        self._last, self._last_raw = None, ""

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        """Start the worker; the model loads in the background."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="Transcriber", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop after the current segment finishes."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            engine = self._factory(self._cfg)
        except Exception as exc:
            self.error = f"Could not load Whisper model '{self._cfg.model}': {exc}"
            log.error("%s. On first run an internet connection is needed to download it.",
                      self.error)
            self.ready.set()
            return
        with self._lock:
            self._language_dirty = True   # a restarted engine must get the runtime language
        self.ready.set()
        while not self._stop.is_set():
            for batch in self._next_batches():
                if self._stop.is_set():
                    break
                try:
                    self._process(engine, batch)
                except Exception:
                    log.exception("Transcription failed for one segment; continuing")
                finally:
                    self.processed += 1
        log.info("Transcriber stopped")

    def _next_batches(self) -> list[list[Segment]]:
        """Wait for segments, drain the backlog, drop stale ones, and batch the rest."""
        try:
            pending = [self._in.get(timeout=GET_TIMEOUT_S)]
        except queue.Empty:
            return []
        while True:
            try:
                pending.append(self._in.get_nowait())
            except queue.Empty:
                break
        now = time.monotonic()
        fresh = [s for s in pending if now - s.created_at <= self._cfg.max_lag_s]
        if len(fresh) < len(pending):
            log.warning("Skipped %d segment(s) older than %.0fs (CPU too slow?)",
                        len(pending) - len(fresh), self._cfg.max_lag_s)
            self.processed += len(pending) - len(fresh)
        return batch_segments(fresh, self._cfg.merge_max_s)

    def _context_for(self, first: Segment, mode: str) -> str | None:
        last = self._last
        if not (self._cfg.use_context and last and last.mode == mode):
            return None
        # wall-clock gap: stream time does not advance while no audio arrives
        # (paused video), so it cannot tell a pause or a new video from continuous speech
        speech_started = first.created_at - first.duration_s
        gap_ok = speech_started - self._last_spoken_end <= self._cfg.context_max_gap_s
        return self._last_raw if gap_ok else None

    def _process(self, engine: Engine, batch: list[Segment]) -> None:
        self._apply_language(engine)
        mode = self.mode
        task = "translate" if mode == "english" else "transcribe"
        first, last = batch[0], batch[-1]
        started = time.perf_counter()
        rec = engine.recognize(merge_segments(batch), task, self._context_for(first, mode))
        compute = time.perf_counter() - started

        text = self._clean(rec)
        if text and first.overlap_s > 0 and self._last is not None:
            text = remove_overlap(self._last_raw, text)
        if not text:
            return
        subtitle = Subtitle(text=self._filter(text), language=rec.language, mode=mode,
                            start_s=first.start_s, end_s=last.end_s,
                            latency_s=time.monotonic() - last.created_at, compute_s=compute)
        self._last = subtitle
        self._last_raw = text
        self._last_spoken_end = last.created_at
        self.last_output = time.monotonic()
        put_drop_oldest(self._out, subtitle)

    def _clean(self, rec: Recognition) -> str:
        """Apply Whisper's silence rule, repetition and hallucination filters."""
        if (rec.no_speech_prob > self._cfg.no_speech_threshold
                and rec.avg_logprob < self._cfg.logprob_threshold):
            log.debug("Dropped as non-speech: %r", self._filter(rec.text))
            return ""
        text = strip_broken(rec.text)
        if compression_ratio(text) > MAX_COMPRESSION_RATIO:
            text = collapse_repeats(text)
            if compression_ratio(text) > MAX_COMPRESSION_RATIO:
                log.debug("Dropped repetitive output: %r", self._filter(rec.text))
                return ""
        if is_hallucination(text):
            log.debug("Dropped filler output: %r", self._filter(text))
            return ""
        return text


# -- self-test -----------------------------------------------------------

def _profanity_filter() -> Callable[[str], str]:
    """Masking filter from config (self-tests mask exactly like the app)."""
    from config import load_config
    from profanity import ProfanityFilter

    return ProfanityFilter.from_config(load_config().profanity)


def _print_subtitle(sub: Subtitle) -> None:
    log.info("[%6.1fs] %-8s %-2s | latency %.1fs (compute %.1fs) | %s",
             sub.start_s, sub.mode, sub.language, sub.latency_s, sub.compute_s, sub.text)


def _run_file(path: str, cfg: TranscribeConfig) -> None:
    """Segment a WAV and feed segments one by one, as they would arrive live."""
    from dataclasses import replace

    from audio_dsp import StreamResampler, read_wav
    from config import load_config
    from segmenter import Segmenter
    from vad import SileroVad

    audio, rate = read_wav(path)
    audio = StreamResampler(rate, SAMPLE_RATE).process(audio)
    segmenter = Segmenter(load_config().vad, SileroVad())
    segments = segmenter.feed(audio)
    tail = segmenter.flush()
    segments += [tail] if tail else []

    seg_q: queue.Queue[Segment] = queue.Queue()
    sub_q: queue.Queue[Subtitle] = queue.Queue()
    worker = Transcriber(seg_q, sub_q, cfg, text_filter=_profanity_filter())
    worker.start()
    worker.ready.wait()
    if worker.error:
        return
    started = time.perf_counter()
    for i, seg in enumerate(segments, 1):
        seg_q.put(replace(seg, created_at=time.monotonic()))
        while worker.processed < i and worker.running:
            time.sleep(0.02)
        while not sub_q.empty():
            _print_subtitle(sub_q.get_nowait())
    worker.stop()
    total = sum(s.duration_s for s in segments)
    elapsed = time.perf_counter() - started
    log.info("%d segments, %.1fs of speech processed in %.1fs (%.2fx real time)",
             len(segments), total, elapsed, elapsed / max(total, 1e-9))


def _run_live(seconds: float, cfg: TranscribeConfig) -> None:
    """Capture -> segment -> transcribe from system audio; Enter toggles the mode."""
    from capture import AudioCapture, make_audio_queue
    from config import load_config
    from segmenter import SegmenterWorker

    app = load_config()
    audio_q = make_audio_queue(app.audio)
    seg_q: queue.Queue[Segment] = queue.Queue(maxsize=20)
    sub_q: queue.Queue[Subtitle] = queue.Queue(maxsize=20)
    transcriber = Transcriber(seg_q, sub_q, cfg, text_filter=_profanity_filter())
    transcriber.start()
    log.info("Loading model...")
    transcriber.ready.wait()
    if transcriber.error:
        return
    capture = AudioCapture(audio_q, app.audio)
    segmenter = SegmenterWorker(audio_q, seg_q, app.vad, app.audio.sample_rate)
    capture.start()
    segmenter.start()

    def toggle_on_enter() -> None:
        while True:
            try:
                input()
            except EOFError:
                return
            transcriber.toggle_mode()

    threading.Thread(target=toggle_on_enter, daemon=True).start()
    log.info("Listening for %.0fs. Press Enter to switch original/english, Ctrl+C to quit.",
             seconds)
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            try:
                _print_subtitle(sub_q.get(timeout=0.2))
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        capture.stop()
        segmenter.stop()
        transcriber.stop()


def _language_arg(value: str) -> str:
    """argparse adapter for parse_language with a readable error."""
    import argparse

    try:
        return parse_language(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_arg_parser():
    """Command-line options for the transcription self-test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test transcription")
    parser.add_argument("--file", help="16-bit WAV to transcribe instead of live audio")
    parser.add_argument("--seconds", type=float, default=60.0, help="live test length")
    parser.add_argument("--mode", choices=MODES, type=str.lower,
                        help="start in 'original' or 'english' (translation) mode; "
                             "Enter still toggles while running")
    parser.add_argument("--model", help="override the configured model, e.g. base")
    parser.add_argument("--language", type=_language_arg,
                        help="force a language code such as bn, hi, ja, or 'auto' to detect")
    return parser


def config_from_args(args, base: TranscribeConfig) -> TranscribeConfig:
    """Apply parsed command-line options on top of the loaded config."""
    return apply_overrides(base, mode=args.mode, model=args.model, language=args.language)


if __name__ == "__main__":
    from config import app_dir, load_config
    from logsetup import setup_logging

    args = build_arg_parser().parse_args()
    setup_logging(log_file=app_dir() / "livesub.log")
    tcfg = config_from_args(args, load_config().transcribe)
    log.info("Model %s | mode %s | language %s", tcfg.model, tcfg.mode,
             tcfg.language or "auto-detect")
    if args.file:
        _run_file(args.file, tcfg)
    else:
        _run_live(args.seconds, tcfg)
