"""Resumable transcription: partial results on disk, and the pass that fills them.

State lives in one folder per output file:
    plan.json        what is being done (input fingerprint, model, language plan)
    <task>.jsonl     one line per finished Whisper segment, flushed as it is produced
                     (a crash can only damage the last line, which is dropped on load)
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from offline_cues import Word
from offline_lang import Chunk, Span
from offline_progress import Progress
from offline_srt import seconds_to_ms

SAMPLE_RATE = 16000
PLAN_VERSION = 1
PROMPT_CHARS = 200          # tail of the previous text given to Whisper when a run resumes
MIN_REMAINING_MS = 500      # a chunk with less audio left than this counts as finished
GAP_JOIN_MS = 1000          # speech separated by less than this is one stretch
GAP_MIN_MS = 1500           # shorter stretches (a cough, a click) are not worth re-checking
GAP_COVERED = 0.5           # a stretch with less than this share of its time in segments is a gap
GAP_PAD_MS = 300            # audio added on both sides of a re-checked stretch


@dataclass(frozen=True)
class RawSegment:
    """One segment straight from the engine; times in seconds from the start of the audio
    handed to it. Each word is (text with leading space, start_s, end_s, probability)."""

    start_s: float
    end_s: float
    text: str
    words: tuple[tuple[str, float, float, float], ...]


@dataclass(frozen=True)
class Segment:
    chunk: int
    start_ms: int
    end_ms: int
    text: str
    prob: float                       # mean word probability
    words: tuple[Word, ...]


class Engine(Protocol):
    def transcribe(self, audio: np.ndarray, language: str, task: str,
                   prompt: str | None, vad: bool = True) -> Iterable[RawSegment]: ...


def words_from_text(text: str, start_ms: int, end_ms: int) -> tuple[Word, ...]:
    """Spread a segment's words evenly over its time (used when Whisper gave no word times)."""
    parts = text.split()
    if not parts:
        return ()
    step = (end_ms - start_ms) / len(parts)
    return tuple(Word(" " + p, round(start_ms + i * step), round(start_ms + (i + 1) * step))
                 for i, p in enumerate(parts))


def to_segment(raw: RawSegment, chunk: int, offset_ms: int) -> Segment:
    start, end = offset_ms + seconds_to_ms(raw.start_s), offset_ms + seconds_to_ms(raw.end_s)
    words = tuple(Word(t, offset_ms + seconds_to_ms(s), offset_ms + seconds_to_ms(e))
                  for t, s, e, _ in raw.words if t.strip())
    if not words:
        words = words_from_text(raw.text, start, end)
    probs = [p for _, _, _, p in raw.words]
    return Segment(chunk, start, end, raw.text.strip(), sum(probs) / len(probs) if probs else 1.0,
                   words)


def _encode(seg: Segment) -> dict:
    return {"k": "seg", "c": seg.chunk, "s": seg.start_ms, "e": seg.end_ms, "t": seg.text,
            "p": round(seg.prob, 3), "w": [[w.text, w.start_ms, w.end_ms] for w in seg.words]}


def _decode(row: dict) -> Segment:
    return Segment(row["c"], row["s"], row["e"], row["t"], row["p"],
                   tuple(Word(t, s, e) for t, s, e in row["w"]))


class PassStore:
    """Append-only record of one pass (transcribe or translate)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[list[Segment], set[int], bool]:
        """(segments, finished chunk numbers, pass finished). Drops a damaged tail line."""
        segments: list[Segment] = []
        chunks_done: set[int] = set()
        finished = False
        if not self.path.exists():
            return segments, chunks_done, finished
        good_bytes = 0
        with self.path.open("rb") as fh:
            for raw_line in fh:
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                    kind = row["k"]
                    if kind == "seg":
                        segments.append(_decode(row))
                    elif kind == "chunk_done":
                        chunks_done.add(row["c"])
                    elif kind == "done":
                        finished = True
                    else:
                        raise ValueError(kind)
                except (ValueError, KeyError, TypeError):
                    break
                good_bytes += len(raw_line)
        if good_bytes != self.path.stat().st_size:
            with self.path.open("r+b") as fh:
                fh.truncate(good_bytes)
        return segments, chunks_done, finished

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append_segment(self, seg: Segment) -> None:
        self._append(_encode(seg))

    def mark_chunk_done(self, chunk: int) -> None:
        self._append({"k": "chunk_done", "c": chunk})

    def mark_finished(self) -> None:
        self._append({"k": "done"})


class StateDir:
    """The partial-results folder of one output file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        return (self.path / "plan.json").exists()

    def store(self, task: str) -> PassStore:
        return PassStore(self.path / f"{task}.jsonl")

    def write_plan(self, meta: dict, chunks: Sequence[Chunk]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        doc = {"version": PLAN_VERSION, **meta,
               "chunks": [[c.start_ms, c.end_ms, c.language] for c in chunks]}
        tmp = self.path / "plan.json.tmp"
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path / "plan.json")

    def read_plan(self) -> tuple[dict, list[Chunk]] | None:
        """(meta, chunks), or None if the plan is missing or unreadable."""
        try:
            doc = json.loads((self.path / "plan.json").read_text(encoding="utf-8"))
            chunks = [Chunk(s, e, lang) for s, e, lang in doc.pop("chunks")]
            return doc, chunks
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def remove(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def plan_matches(saved: dict, expected: dict) -> list[str]:
    """Names of settings that differ between a saved plan and this run (empty = compatible)."""
    keys = ("version", "input_size", "input_mtime_ns", "model", "language", "audio_track")
    wanted = {"version": PLAN_VERSION, **expected}
    return [k for k in keys if saved.get(k) != wanted.get(k)]


def last_position_ms(segments: Sequence[Segment]) -> int:
    return max((s.end_ms for s in segments), default=0)


def find_skipped_speech(spans: Sequence[Span], segments: Sequence[Segment]) -> list[Span]:
    """Stretches where the VAD hears speech but the transcript has (almost) no text.

    Whisper sometimes skips a whole window without any warning. Speech spans closer than
    GAP_JOIN_MS are treated as one stretch; a stretch of at least GAP_MIN_MS whose time is
    less than GAP_COVERED covered by segments is returned, padded a little on both sides.
    """
    regions: list[list[int]] = []
    for start, end in spans:
        if regions and start - regions[-1][1] < GAP_JOIN_MS:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    found = []
    for start, end in regions:
        if end - start < GAP_MIN_MS:
            continue
        covered = sum(max(0, min(end, s.end_ms) - max(start, s.start_ms)) for s in segments)
        if covered < GAP_COVERED * (end - start):
            found.append((max(0, start - GAP_PAD_MS), end + GAP_PAD_MS))
    return found


def fill_skipped_speech(engine: Engine, audio: np.ndarray, chunks: Sequence[Chunk], task: str,
                        store: PassStore, segments: list[Segment], spans: Sequence[Span],
                        note: Callable[[str], None]) -> None:
    """Re-transcribe stretches Whisper skipped, one by one without the VAD, and store them."""
    total_ms = len(audio) * 1000 // SAMPLE_RATE
    gaps = find_skipped_speech(spans, segments)
    if gaps:
        note(f"Re-checking {len(gaps)} stretch(es) of speech the model skipped "
             f"({sum(e - s for s, e in gaps) / 1000:.0f} s)")
    for start, end in gaps:
        end = min(end, total_ms)
        middle = (start + end) // 2
        number = next((i for i, c in enumerate(chunks) if c.start_ms <= middle < c.end_ms),
                      len(chunks) - 1)
        part = audio[start * SAMPLE_RATE // 1000: end * SAMPLE_RATE // 1000]
        for raw in engine.transcribe(part, chunks[number].language, task, None, vad=False):
            seg = to_segment(raw, number, start)
            store.append_segment(seg)
            segments.append(seg)


def run_pass(engine: Engine, audio: np.ndarray, chunks: Sequence[Chunk], task: str,
             store: PassStore, progress: Progress, spans: Sequence[Span] = (),
             note: Callable[[str], None] = lambda _msg: None) -> list[Segment]:
    """Transcribe every chunk, skipping what the store already has; returns all segments.

    With `spans` (where the VAD heard speech), speech that came back without text is
    transcribed again at the end (see fill_skipped_speech).
    """
    segments, chunks_done, _ = store.load()
    for number, chunk in enumerate(chunks):
        if number in chunks_done:
            continue
        mine = [s for s in segments if s.chunk == number]
        begin = max(chunk.start_ms, mine[-1].end_ms) if mine else chunk.start_ms
        prompt = mine[-1].text[-PROMPT_CHARS:] if mine else None
        if chunk.end_ms - begin >= MIN_REMAINING_MS:
            part = audio[begin * SAMPLE_RATE // 1000: chunk.end_ms * SAMPLE_RATE // 1000]
            for raw in engine.transcribe(part, chunk.language, task, prompt):
                seg = to_segment(raw, number, begin)
                store.append_segment(seg)
                segments.append(seg)
                progress.update(seg.end_ms)
        store.mark_chunk_done(number)
        progress.update(chunk.end_ms)
    if spans:
        fill_skipped_speech(engine, audio, chunks, task, store, segments, spans, note)
    store.mark_finished()
    return segments
