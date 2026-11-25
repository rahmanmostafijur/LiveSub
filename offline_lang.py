"""Whole-file language planning: several speech samples across the file, dominant language,
and per-block detection for files that mix languages. Pure logic: the model is a callable."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

Span = tuple[int, int]                       # (start_ms, end_ms)
Detect = Callable[[Sequence[Span]], dict[str, float]]   # speech spans -> language probabilities

SAMPLE_MS = 30_000          # Whisper's language detector looks at 30 s
MIN_SAMPLE_MS = 3_000       # less speech than this is not enough to judge a language
DEFAULT_SAMPLES = 6
AGREEMENT_MIN = 0.8         # share of samples that must agree for a single-language file
BLOCK_MS = 180_000          # block length when a file mixes languages
EDGE_SEARCH_MS = 30_000     # a block edge is moved to the longest silence within this distance
UNSURE = 0.5                # block detections below this probability are not trusted
FLIP_SURE = 0.85            # an isolated odd block below this probability follows its neighbours


@dataclass(frozen=True)
class Chunk:
    start_ms: int
    end_ms: int
    language: str


@dataclass(frozen=True)
class Plan:
    chunks: tuple[Chunk, ...]
    dominant: str
    agreement: float        # share of samples that voted for `dominant` (1.0 when forced)
    samples: int            # detection windows used (0 when forced or no speech)
    mixed: bool
    forced: bool = False


def speech_ms(spans: Sequence[Span]) -> int:
    return sum(e - s for s, e in spans)


def collect_window(spans: Sequence[Span], offset_ms: int, length_ms: int) -> list[Span]:
    """Spans covering `length_ms` of speech starting `offset_ms` into the speech time."""
    out, seen = [], 0
    end_at = offset_ms + length_ms
    for start, end in spans:
        lo, hi = max(offset_ms, seen), min(end_at, seen + end - start)
        if hi > lo:
            out.append((start + lo - seen, start + hi - seen))
        seen += end - start
        if seen >= end_at:
            break
    return out


def pick_windows(spans: Sequence[Span], count: int, length_ms: int = SAMPLE_MS) -> list[list[Span]]:
    """Up to `count` windows of speech spread evenly over the speech in `spans`."""
    total = speech_ms(spans)
    if total < MIN_SAMPLE_MS or count < 1:
        return []
    count = min(count, max(1, total // length_ms))
    windows = []
    for i in range(count):
        centre = int(total * (i + 0.5) / count)
        offset = min(max(0, centre - length_ms // 2), max(0, total - length_ms))
        windows.append(collect_window(spans, offset, length_ms))
    return windows


def aggregate(votes: Sequence[dict[str, float]]) -> tuple[str, float]:
    """Dominant language (highest mean probability) and the share of samples led by it."""
    totals: dict[str, float] = {}
    for probs in votes:
        for lang, p in probs.items():
            totals[lang] = totals.get(lang, 0.0) + p
    dominant = max(totals, key=totals.__getitem__)
    leaders = [max(v, key=v.__getitem__) for v in votes]
    return dominant, leaders.count(dominant) / len(votes)


def block_edges(spans: Sequence[Span], total_ms: int, block_ms: int = BLOCK_MS) -> list[int]:
    """Boundaries between blocks of about `block_ms`, each placed in the nearest long pause."""
    edges = [0]
    gaps = [((a_end + b_start) // 2, b_start - a_end)
            for (_, a_end), (b_start, _) in zip(spans, spans[1:])]
    target = block_ms
    while target < total_ms - block_ms // 2:
        near = [g for g in gaps if abs(g[0] - target) <= EDGE_SEARCH_MS and g[0] > edges[-1]]
        edge = max(near, key=lambda g: g[1])[0] if near else target
        edges.append(edge)
        target = edge + block_ms
    edges.append(total_ms)
    return edges


def smooth_blocks(langs: list[str | None], confs: list[float], fallback: str) -> list[str]:
    """Fill unsure blocks from their neighbours and absorb isolated, weakly detected flips."""
    out: list[str] = []
    for i, (lang, conf) in enumerate(zip(langs, confs)):
        if lang is None or conf < UNSURE:
            lang = out[-1] if out else next(
                (l for l, c in zip(langs[i + 1:], confs[i + 1:]) if l and c >= UNSURE), fallback)
        out.append(lang)
    for i in range(1, len(out) - 1):
        if out[i - 1] == out[i + 1] != out[i] and confs[i] < FLIP_SURE:
            out[i] = out[i - 1]
    return out


def merge_blocks(edges: Sequence[int], langs: Sequence[str]) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for (start, end), lang in zip(zip(edges, edges[1:]), langs):
        if chunks and chunks[-1].language == lang:
            chunks[-1] = Chunk(chunks[-1].start_ms, end, lang)
        else:
            chunks.append(Chunk(start, end, lang))
    return tuple(chunks)


def _blocks_plan(spans: Sequence[Span], total_ms: int, detect: Detect, dominant: str
                 ) -> tuple[Chunk, ...]:
    edges = block_edges(spans, total_ms)
    langs: list[str | None] = []
    confs: list[float] = []
    for start, end in zip(edges, edges[1:]):
        inside = [(max(s, start), min(e, end)) for s, e in spans if e > start and s < end]
        windows = pick_windows(inside, 1)
        if not windows:
            langs.append(None)
            confs.append(0.0)
            continue
        probs = detect(windows[0])
        top = max(probs, key=probs.__getitem__)
        langs.append(top)
        confs.append(probs[top])
    return merge_blocks(edges, smooth_blocks(langs, confs, dominant))


def plan_language(total_ms: int, spans: Sequence[Span], detect: Detect, *,
                  forced: str = "", samples: int = DEFAULT_SAMPLES,
                  on_note: Callable[[str], None] = lambda _msg: None) -> Plan:
    """Decide which language to transcribe each part of the file in."""
    if forced:
        return Plan((Chunk(0, total_ms, forced),), forced, 1.0, 0, False, forced=True)
    windows = pick_windows(spans, samples)
    if not windows:
        return Plan((Chunk(0, total_ms, "en"),), "en", 0.0, 0, False)
    votes = []
    for number, window in enumerate(windows, start=1):
        on_note(f"Detecting language: sample {number}/{len(windows)}")
        votes.append(detect(window))
    dominant, agreement = aggregate(votes)
    if agreement >= AGREEMENT_MIN or total_ms <= BLOCK_MS:
        return Plan((Chunk(0, total_ms, dominant),), dominant, agreement, len(windows), False)
    on_note("Samples disagree: checking the language block by block")
    chunks = _blocks_plan(spans, total_ms, detect, dominant)
    return Plan(chunks, dominant, agreement, len(windows), len(chunks) > 1)
