"""Subtitle cues from word timestamps. Pure functions on integer milliseconds.

A cue has at most two lines of about 42 columns, lasts 1-7 s, ends at sentence ends and
pauses, and never overlaps its neighbours (minimum gap between cues).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence

MAX_LINE_COLS = 42          # display columns per line (wide CJK characters count 2)
MIN_CUE_MS = 1000           # short cues are held on screen this long if the next cue allows
MAX_CUE_MS = 7000
PAUSE_MS = 600              # a silence this long between words always ends a cue
MIN_GAP_MS = 80             # smallest gap between two cues (about 2 frames)
HARD_MIN_CUE_MS = 200       # a cue squeezed by its neighbour is never shorter than this
MAX_WORD_MS = 2500          # one word lasting longer is a timestamp glitch (music, silence)
MAX_CPS = 20                # reading speed: characters per second
SENTENCE_MIN_MS = 1200      # a sentence end closes the cue once the cue is this long ...
SENTENCE_MIN_COLS = 20      # ... or this wide; shorter sentences are joined with the next
CLAUSE_SPLIT_COLS = 50      # at this width a comma-like mark is a good place to end the cue

SENTENCE_END = ".?!…。？！।॥؟۔"
CLAUSE_END = ",;:，、；：—–"
_CLOSERS = "\"'”’»)]}"
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr", "prof", "no", "etc"}


@dataclass(frozen=True)
class Word:
    """One word as Whisper wrote it: `text` keeps its leading space (none for CJK)."""

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _char_width(ch: str) -> int:
    if unicodedata.category(ch) in ("Mn", "Me", "Cf") or unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Columns the text takes: combining marks 0, wide CJK 2, everything else 1."""
    return sum(_char_width(ch) for ch in text)


def ends_sentence(text: str) -> bool:
    """True if the word ends with sentence punctuation (not an abbreviation like 'Dr.')."""
    stripped = text.strip().rstrip(_CLOSERS)
    if not stripped or stripped[-1] not in SENTENCE_END:
        return False
    if stripped[-1] == "." and stripped[:-1].lower().lstrip("\"'“(") in _ABBREVIATIONS:
        return False
    return True


def ends_clause(text: str) -> bool:
    stripped = text.strip().rstrip(_CLOSERS)
    return bool(stripped) and stripped[-1] in CLAUSE_END + SENTENCE_END


def break_lines(tokens: Sequence[str], max_cols: int = MAX_LINE_COLS) -> tuple[str, ...] | None:
    """Split tokens into one or two balanced lines; None if they cannot fit in two."""
    whole = "".join(tokens).strip()
    if display_width(whole) <= max_cols:
        return (whole,)
    best: tuple[float, tuple[str, str]] | None = None
    for i in range(1, len(tokens)):
        first, second = "".join(tokens[:i]).strip(), "".join(tokens[i:]).strip()
        w1, w2 = display_width(first), display_width(second)
        if not first or not second or w1 > max_cols or w2 > max_cols:
            continue
        score = abs(w1 - w2) - (6 if ends_clause(first) else 0) + (1 if w1 > w2 else 0)
        if best is None or score < best[0]:
            best = (score, (first, second))
    return best[1] if best else None


def _fits(words: Sequence[Word], max_cols: int) -> bool:
    return len(words) == 1 or break_lines([w.text for w in words], max_cols) is not None


def _width(words: Sequence[Word]) -> int:
    return display_width("".join(w.text for w in words).strip())


def clean_words(words: Sequence[Word]) -> list[Word]:
    """Drop empty words, order by time, fix inverted times and cap absurdly long words."""
    cleaned = []
    for w in sorted((w for w in words if w.text.strip()), key=lambda w: (w.start_ms, w.end_ms)):
        end = min(max(w.end_ms, w.start_ms), w.start_ms + MAX_WORD_MS)
        cleaned.append(Word(w.text, w.start_ms, end))
    return cleaned


def group_words(words: Sequence[Word], max_cols: int = MAX_LINE_COLS,
                max_ms: int = MAX_CUE_MS, pause_ms: int = PAUSE_MS) -> list[list[Word]]:
    """Split time-ordered words into cue-sized groups."""
    groups: list[list[Word]] = []
    cur: list[Word] = []

    def flush(count: int | None = None) -> None:
        nonlocal cur
        count = len(cur) if count is None else count
        if count:
            groups.append(cur[:count])
        cur = cur[count:]

    for word in words:
        if cur:
            too_long = word.end_ms - cur[0].start_ms > max_ms
            if word.start_ms - cur[-1].end_ms >= pause_ms:
                flush()
            elif too_long or not _fits(cur + [word], max_cols):
                split = _last_clause_end(cur)
                flush(split if split else None)
                if cur and (word.end_ms - cur[0].start_ms > max_ms
                            or not _fits(cur + [word], max_cols)):
                    flush()
        cur.append(word)
        if ends_sentence(word.text) and (
                cur[-1].end_ms - cur[0].start_ms >= SENTENCE_MIN_MS
                or _width(cur) >= SENTENCE_MIN_COLS):
            flush()
        elif ends_clause(word.text) and _width(cur) >= CLAUSE_SPLIT_COLS:
            flush()
    flush()
    return groups


def _last_clause_end(words: Sequence[Word]) -> int:
    """Number of words up to the last clause mark in the second half of the cue, or 0."""
    total = _width(words)
    for i in range(len(words) - 1, 0, -1):
        if ends_clause(words[i - 1].text) and _width(words[:i]) * 2 >= total * 0.8:
            return i
    return 0


def fix_timing(cues: Sequence[Cue], *, min_ms: int = MIN_CUE_MS, max_ms: int = MAX_CUE_MS,
               gap_ms: int = MIN_GAP_MS) -> list[Cue]:
    """Order cues, remove overlaps, keep a minimum gap, hold short cues up to min duration."""
    ordered = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    starts = [c.start_ms for c in ordered]
    ends = [max(c.end_ms, c.start_ms + HARD_MIN_CUE_MS) for c in ordered]
    for i in range(1, len(ordered)):
        if starts[i] < ends[i - 1] + gap_ms:
            # the earlier cue gives way; only if that would shrink it below the minimum
            # does the later cue start a little late instead
            ends[i - 1] = max(starts[i] - gap_ms, starts[i - 1] + HARD_MIN_CUE_MS)
            starts[i] = max(starts[i], ends[i - 1] + gap_ms)
            ends[i] = max(ends[i], starts[i] + HARD_MIN_CUE_MS)
    result = []
    for i, cue in enumerate(ordered):
        want = max(min_ms, min(len(" ".join(cue.lines)) * 1000 // MAX_CPS, max_ms))
        limit = starts[i] + max_ms
        if i + 1 < len(ordered):
            limit = min(limit, starts[i + 1] - gap_ms)
        end = max(ends[i], min(starts[i] + want, limit))
        result.append(Cue(starts[i], min(end, starts[i] + max_ms), cue.lines))
    return result


def build_cues(words: Sequence[Word], *, max_cols: int = MAX_LINE_COLS) -> list[Cue]:
    """Words with timestamps -> subtitle cues with valid, non-overlapping timing."""
    cues = []
    for group in group_words(clean_words(words), max_cols):
        lines = break_lines([w.text for w in group], max_cols)
        if lines is None:  # a single over-long word: keep it on one line
            lines = ("".join(w.text for w in group).strip(),)
        cues.append(Cue(group[0].start_ms, group[-1].end_ms, lines))
    return fix_timing(cues)


def validate_cues(cues: Sequence[Cue], *, max_ms: int = MAX_CUE_MS,
                  gap_ms: int = MIN_GAP_MS) -> list[str]:
    """Human-readable problems in a cue list (empty when it is valid)."""
    problems = []
    for i, cue in enumerate(cues):
        if cue.end_ms <= cue.start_ms:
            problems.append(f"cue {i + 1}: end is not after start")
        if cue.end_ms - cue.start_ms > max_ms:
            problems.append(f"cue {i + 1}: longer than {max_ms} ms")
        if not 1 <= len(cue.lines) <= 2:
            problems.append(f"cue {i + 1}: {len(cue.lines)} lines")
        if i and cue.start_ms < cues[i - 1].end_ms + gap_ms:
            problems.append(f"cue {i + 1}: overlaps or touches cue {i}")
    return problems
