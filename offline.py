"""Offline subtitle generator: a video or audio file in, an accurate .srt file out.

    python offline.py movie.mkv
    python offline.py movie.mkv --model medium --language hi --also-translate
    python offline.py --batch D:\\Movies --resume

Runs on the CPU, needs no internet after the one-time model download. The live subtitle
app (main.py) is not involved.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import ProfanityConfig, language_name
from offline_audio import (AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, MediaError, burn_subtitles,
                           extract_audio, ffmpeg_help, find_ffmpeg, has_video)
from offline_clean import cues_from_segments
from offline_cues import validate_cues
from offline_lang import Chunk, Plan, plan_language
from offline_progress import Progress, format_duration
from offline_srt import write_srt
from offline_state import StateDir, last_position_ms, plan_matches, run_pass

SAMPLE_RATE = 16000
MODELS = ("small", "medium", "large-v3")
PASS_LABEL = {"transcribe": "Transcribing", "translate": "Translating "}


class UsageError(Exception):
    """A problem the user can fix by changing the command (message is shown as is)."""


@dataclass
class Report:
    """What happened to one file, for the final summary."""

    name: str
    status: str = "done"
    notes: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="offline.py", description="Create an accurate, synced .srt from a video/audio file.")
    p.add_argument("input", nargs="?", type=Path, help="video or audio file")
    p.add_argument("--batch", type=Path, metavar="FOLDER",
                   help="process every video/audio file in this folder")
    p.add_argument("--model", choices=MODELS, default="large-v3")
    p.add_argument("--language", default="auto", help="auto (default), en, ko, hi, ml, ta, bn, ...")
    p.add_argument("--task", choices=("transcribe", "translate"), default="transcribe",
                   help="translate = English subtitles")
    p.add_argument("--output", type=Path, help="output .srt (single file only)")
    p.add_argument("--also-translate", action="store_true",
                   help="write the original-language .srt and an English .en.srt")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--restart", action="store_true",
                   help="discard an interrupted run and start over")
    p.add_argument("--mask", action="store_true", help="mask profanity (f**k), off by default")
    p.add_argument("--burn", action="store_true",
                   help="also write a copy of the video with subtitles burned in")
    p.add_argument("--overwrite", action="store_true",
                   help="batch: redo files that already have an .srt")
    p.add_argument("--audio-track", type=int, default=0, help="audio track number, 0 = first")
    p.add_argument("--lang-samples", type=int, default=6,
                   help="speech samples used to detect the language")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = automatic)")
    p.add_argument("--yes", action="store_true", help="download a missing model without asking")
    args = p.parse_args(argv)
    if bool(args.input) == bool(args.batch):
        p.error("give either an input file or --batch FOLDER")
    if args.batch and args.output:
        p.error("--output cannot be used with --batch")
    if args.resume and args.restart:
        p.error("--resume and --restart cannot be combined")
    args.language = args.language.strip().lower()
    return args


def check_language(code: str) -> None:
    if code == "auto":
        return
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    if code not in _LANGUAGE_CODES:
        raise UsageError(f"Unknown language code '{code}'. Examples: en, ko, hi, ml, ta, bn.")


def output_paths(source: Path, args: argparse.Namespace) -> dict[str, Path]:
    """{task: srt path}. Original next to the input, English as '<name>.en.srt'."""
    base = args.output if args.output else source.with_suffix(".srt")
    english = base.with_name(base.stem + ".en.srt")
    if args.task == "translate":
        return {"translate": args.output if args.output else english}
    paths = {"transcribe": base}
    if args.also_translate:
        paths["translate"] = english
    return paths


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        if not args.input.is_file():
            raise UsageError(f"Input file not found: {args.input}")
        return [args.input]
    if not args.batch.is_dir():
        raise UsageError(f"Not a folder: {args.batch}")
    media = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
    files = sorted(f for f in args.batch.iterdir() if f.is_file() and f.suffix.lower() in media)
    if not files:
        raise UsageError(f"No video or audio files found in {args.batch}")
    return files


def fingerprint(source: Path, args: argparse.Namespace) -> dict:
    stat = source.stat()
    return {"input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns, "model": args.model,
            "language": args.language, "audio_track": args.audio_track}


def prepare_plan(state: StateDir, meta: dict, audio, spans, backend, args
                 ) -> tuple[Plan | None, list[Chunk], float]:
    """(fresh plan or None when resuming, chunks, seconds spent detecting language)."""
    if state.exists() and not (args.resume or args.restart):
        raise UsageError(f"An interrupted run was found ({state.path.name}). Use --resume to "
                         "continue it or --restart to discard it.")
    if args.restart:
        state.remove()
    saved = state.read_plan() if args.resume and state.exists() else None
    if saved:
        changed = plan_matches(saved[0], meta)
        if changed:
            raise UsageError("The interrupted run was made with different settings or a changed "
                             f"input file ({', '.join(changed)}). Use --restart to start over.")
        print("Resuming: language plan loaded from the interrupted run.")
        return None, saved[1], 0.0
    began = time.monotonic()
    total_ms = len(audio) * 1000 // SAMPLE_RATE
    forced = "" if args.language == "auto" else args.language
    plan = plan_language(total_ms, spans, backend.detector(audio), forced=forced,
                         samples=args.lang_samples, on_note=lambda m: print("  " + m))
    state.write_plan(meta, plan.chunks)
    return plan, list(plan.chunks), time.monotonic() - began


def describe_plan(plan: Plan) -> str:
    if plan.forced:
        return f"Language: {language_name(plan.dominant)} (set by you)"
    if not plan.samples:
        return "Language: no speech found to detect from"
    if plan.mixed:
        parts = ", ".join(f"{language_name(c.language)} {format_duration(c.start_ms / 1000)}-"
                          f"{format_duration(c.end_ms / 1000)}" for c in plan.chunks)
        return f"Language: mixed file, split by section: {parts}"
    return (f"Language: {language_name(plan.dominant)} ({plan.agreement:.0%} of "
            f"{plan.samples} speech samples agree)")


def write_pass(out: Path, segments, mask) -> int:
    cues = cues_from_segments(segments, mask)
    problems = validate_cues(cues)
    if problems:
        raise RuntimeError("Internal error: invalid cue timing: " + "; ".join(problems[:3]))
    write_srt(out, cues)
    return len(cues)


def process_file(source: Path, args: argparse.Namespace, ctx: "Context") -> Report:
    report = Report(source.name)
    paths = output_paths(source, args)
    primary = paths.get("transcribe") or paths["translate"]
    state = StateDir(primary.with_name(primary.stem + ".partial"))
    if args.batch and primary.exists() and not args.overwrite and not state.exists():
        report.status = "skipped (subtitle exists)"
        return report
    print(f"\n== {source.name}")
    began = time.monotonic()
    audio = extract_audio(ctx.ffmpeg, source, args.audio_track)
    total_ms = len(audio) * 1000 // SAMPLE_RATE
    print(f"Audio: {format_duration(total_ms / 1000)} (read in {time.monotonic() - began:.0f} s)")
    spans = ctx.backend().speech_spans(audio)
    plan, chunks, detect_s = prepare_plan(state, fingerprint(source, args), audio, spans,
                                          ctx.backend(), args)
    if plan:
        print(describe_plan(plan))
    english_only = all(c.language == "en" for c in chunks)
    tasks = [t for t in paths if t != "translate" or args.task == "translate" or not english_only]
    if "translate" in paths and "translate" not in tasks:
        report.notes.append("audio is English: no separate English file needed")
        print("Audio is English: no separate English file is needed.")
    for task in tasks:
        report.notes.append(run_one_pass(task, paths[task], audio, chunks, spans, state, ctx,
                                         detect_s))
        detect_s = 0.0
    if args.burn:
        burn(source, paths[args.task], total_ms, ctx, report)
    state.remove()
    return report


def run_one_pass(task: str, out: Path, audio, chunks, spans, state: StateDir, ctx,
                 detect_s: float) -> str:
    store = state.store(task)
    done, _, _ = store.load()
    progress = Progress(PASS_LABEL[task], len(audio) * 1000 // SAMPLE_RATE, last_position_ms(done))
    segments = run_pass(ctx.backend(), audio, chunks, task, store, progress, spans,
                        note=lambda msg: print("\n" + msg))
    progress.finish()
    count = write_pass(out, segments, ctx.mask)
    rtf = progress.real_time_factor()
    speed = (f"real-time factor {rtf:.2f} ({progress.elapsed_s / 60:.1f} min for "
             f"{progress.processed_s / 60:.1f} min of audio)") if rtf else "nothing new to process"
    detect = f", language detection {detect_s:.0f} s" if detect_s else ""
    line = f"{out.name}: {count} subtitles, {speed}{detect}"
    print("Wrote " + line)
    return line


def burn(source: Path, srt: Path, total_ms: int, ctx, report: Report) -> None:
    if not has_video(ctx.ffmpeg, source):
        report.notes.append("--burn skipped: the file has no video")
        print("--burn skipped: the file has no video.")
        return
    target = srt.with_name(srt.stem + ".subbed.mp4")
    burn_subtitles(ctx.ffmpeg, source, srt, target, total_ms)
    print(f"Wrote {target.name}")


@dataclass
class Context:
    ffmpeg: str
    model: str
    threads: int
    mask: object
    _backend: object = None
    load_s: float = 0.0

    def backend(self):
        if self._backend is None:
            from offline_engine import WhisperBackend

            print(f"Loading model '{self.model}' (int8, CPU)...")
            began = time.monotonic()
            self._backend = WhisperBackend(self.model, self.threads)
            self.load_s = time.monotonic() - began
            print(f"Model loaded in {self.load_s:.0f} s")
        return self._backend


def make_mask(enabled: bool):
    if not enabled:
        return None
    from profanity import ProfanityFilter

    return ProfanityFilter.from_config(ProfanityConfig(enabled=True))


def run(args: argparse.Namespace) -> int:
    check_language(args.language)
    files = collect_inputs(args)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(ffmpeg_help(), file=sys.stderr)
        return 2
    from offline_engine import ensure_model

    if not ensure_model(args.model, args.yes):
        return 2
    ctx = Context(ffmpeg, args.model, args.threads, make_mask(args.mask))
    reports = []
    for source in files:
        try:
            reports.append(process_file(source, args, ctx))
        except (UsageError, MediaError) as exc:
            print(f"ERROR {source.name}: {exc}", file=sys.stderr)
            reports.append(Report(source.name, "failed", [str(exc)]))
            if not args.batch:
                return 1
    if args.batch:
        print("\nSummary:")
        for r in reports:
            print(f"  {r.status:<26} {r.name}")
    return 1 if any(r.status == "failed" for r in reports) else 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    try:
        return run(parse_args(argv))
    except UsageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; run the same command with --resume to continue.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
