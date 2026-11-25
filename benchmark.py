"""Compare Whisper models on this CPU using saved speech segments.

Uses exactly the decoding path of transcribe.py (short encoder window, int8,
CPU) so the numbers match what the live app will do. Results are printed and
also written to benchmark_results.md (UTF-8, readable in any editor).
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio_dsp import read_wav
from config import TranscribeConfig, app_dir
from logsetup import setup_logging
from models import download_size_gb, is_downloaded, repo_for
from segmenter import SAMPLE_RATE

log = logging.getLogger("benchmark")

DEFAULT_MODELS = ("small", "medium", "large-v3-turbo", "large-v3")
BENGALI = (0x0980, 0x09FF)
DEVANAGARI = (0x0900, 0x097F)


@dataclass
class ModelResult:
    """Measurements for one model."""

    name: str
    load_s: float = 0.0
    times: list[float] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    error: str | None = None

    def rtf(self, audio_s: float) -> float:
        return sum(self.times) / audio_s if audio_s else 0.0


def script_share(texts: list[str], block: tuple[int, int]) -> float:
    """Percentage of letters that fall in a Unicode block (e.g. Bengali)."""
    letters = [ch for text in texts for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    inside = sum(block[0] <= ord(ch) <= block[1] for ch in letters)
    return 100.0 * inside / len(letters)


def confirm_downloads(models: list[str], model_dir: Path, assume_yes: bool) -> list[str]:
    """Show sizes of missing models and ask before downloading; return models to run."""
    missing = [m for m in models if not is_downloaded(m, model_dir)]
    for model in models:
        status = "downloaded" if model not in missing else "needs download"
        size = download_size_gb(model) if model in missing else None
        size_txt = f"{size:.2f} GB" if size is not None else "-"
        log.info("%-15s %-45s %-15s %s", model, repo_for(model), status, size_txt)
    if not missing or assume_yes:
        return models
    answer = input(f"Download {', '.join(missing)} into {model_dir}? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        return models
    log.info("Skipping models that are not downloaded")
    return [m for m in models if m not in missing]


def run_model(name: str, segments: list[tuple[str, np.ndarray]], args) -> ModelResult:
    """Load one model, warm it up, and time every segment."""
    from transcribe import WhisperEngine

    result = ModelResult(name)
    cfg = TranscribeConfig(model=name, language=args.language, cpu_threads=args.threads,
                           model_dir=str(args.model_dir), min_window_s=args.window)
    engine = None
    try:
        started = time.perf_counter()
        engine = WhisperEngine(cfg)
        result.load_s = time.perf_counter() - started
        engine.recognize(segments[0][1], args.task, None)  # warm-up, not timed
        for seg_name, audio in segments:
            started = time.perf_counter()
            text = engine.recognize(audio, args.task, None).text
            result.times.append(time.perf_counter() - started)
            result.texts.append(text)
            log.info("%-15s %s %.2fs | %s", name, seg_name, result.times[-1], text)
    except Exception as exc:
        result.error = str(exc)
        log.exception("Model '%s' failed", name)
    finally:
        del engine
        gc.collect()
    return result


def build_report(results: list[ModelResult], segments: list[tuple[str, np.ndarray]],
                 audio_s: float, args) -> str:
    """Markdown summary plus side-by-side text for the first segments."""
    lines = [f"# LiveSub model benchmark ({args.task}, language={args.language or 'auto'}, "
             f"{args.threads} threads, {args.window:g}s window)", "",
             f"{len(segments)} segments, {audio_s:.1f}s of audio", "",
             "| model | load s | avg s/segment | max s | RTF | Bengali script % | "
             "Devanagari % |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if r.error or not r.times:
            lines.append(f"| {r.name} | failed: {r.error} | | | | | |")
            continue
        lines.append(
            f"| {r.name} | {r.load_s:.1f} | {np.mean(r.times):.2f} | {max(r.times):.2f} | "
            f"{r.rtf(audio_s):.2f} | {script_share(r.texts, BENGALI):.0f} | "
            f"{script_share(r.texts, DEVANAGARI):.0f} |")
    lines += ["", "RTF < 1 means faster than real time. Live use needs RTF well below 1 "
              "(about 0.5 or less), because the CPU also plays the video.", ""]
    ok = [r for r in results if not r.error]
    for i, (seg_name, audio) in enumerate(segments[:args.show]):
        lines += [f"## {seg_name} ({len(audio) / SAMPLE_RATE:.2f}s)", "",
                  "| model | text |", "|---|---|"]
        lines += [f"| {r.name} | {r.texts[i] or '(empty)'} |" for r in ok if i < len(r.texts)]
        lines.append("")
    return "\n".join(lines)


def load_segments(folder: Path, limit: int | None) -> list[tuple[str, np.ndarray]]:
    """Read seg_*.wav files as 16 kHz mono float32."""
    paths = sorted(folder.glob("seg_*.wav"))[:limit]
    if not paths:
        raise SystemExit(f"No seg_*.wav files in {folder}. Run: python segmenter.py --seconds 60")
    segments = []
    for path in paths:
        audio, rate = read_wav(path)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{path.name}: expected {SAMPLE_RATE} Hz, got {rate}")
        segments.append((path.name, audio))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Whisper models on this CPU")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated list (default: %(default)s)")
    parser.add_argument("--segments-dir", type=Path, default=app_dir() / "segments_test")
    parser.add_argument("--model-dir", type=Path, default=app_dir() / "models")
    parser.add_argument("--language", default="bn", help="language code, or 'auto'")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe",
                        help="transcribe = original mode, translate = English mode")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window", type=float, default=6.0,
                        help="encoder window in seconds (transcribe.py uses 6; Whisper's "
                             "standard is 30)")
    parser.add_argument("--limit", type=int, help="only use the first N segments")
    parser.add_argument("--show", type=int, default=5, help="segments shown side by side")
    parser.add_argument("--yes", action="store_true", help="download without asking")
    parser.add_argument("--out", type=Path, default=app_dir() / "benchmark_results.md")
    args = parser.parse_args()
    setup_logging()
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    segments = load_segments(args.segments_dir, args.limit)
    audio_s = sum(len(a) for _, a in segments) / SAMPLE_RATE
    log.info("%d segments, %.1fs of audio from %s", len(segments), audio_s, args.segments_dir)
    models = confirm_downloads([m.strip() for m in args.models.split(",") if m.strip()],
                               args.model_dir, args.yes)
    results = [run_model(name, segments, args) for name in models]
    report = build_report(results, segments, audio_s, args)
    args.out.write_text(report, encoding="utf-8")
    print("\n" + report)
    log.info("Report saved to %s (open it in VS Code to see Bengali text)", args.out)


if __name__ == "__main__":
    main()
