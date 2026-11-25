"""Clean Whisper segments (filler hallucinations, repetition loops), then build cues."""
from __future__ import annotations

from typing import Callable, Sequence

from offline_cues import Cue, Word, build_cues, fix_timing
from offline_state import Segment
from textproc import (MAX_COMPRESSION_RATIO, collapse_repeats, compression_ratio,
                      is_hallucination, normalize)

AMBIGUOUS_FILLERS = {"you", "bye"}   # also real one-word lines in films: kept when confident
CONFIDENT = 0.7
LOOP_RUN = 3                         # the same segment text this many times in a row is a loop
EDGE_PAUSE_MS = 600                  # pause next to a first/last word that marks it as stretched
EDGE_BASE_MS = 250                   # speaking time of a word: base + per letter
EDGE_PER_LETTER_MS = 90
EDGE_TOUCH_MS = 40                   # a re-timed word ends/starts this close to its neighbour


def is_filler(seg: Segment) -> bool:
    """Known Whisper inventions ('Thanks for watching', 'Amara.org', noise) via textproc."""
    if not is_hallucination(seg.text):
        return False
    return not (normalize(seg.text) in AMBIGUOUS_FILLERS and seg.prob >= CONFIDENT)


def collapse_word_loops(words: Sequence[Word]) -> tuple[Word, ...]:
    """Collapse a repeated phrase inside a segment, keeping the first copy's timestamps.

    Only segments Whisper itself would call repetitive (high compression ratio) are touched,
    so genuine repeats such as 'No, no, no!' survive.
    """
    text = "".join(w.text for w in words)
    if compression_ratio(text) < MAX_COMPRESSION_RATIO:
        return tuple(words)
    kept, position = [], 0
    for token in collapse_repeats(text).split():
        while position < len(words) and words[position].text.strip() != token:
            position += 1
        if position < len(words):
            kept.append(words[position])
            position += 1
    return tuple(kept)


def expected_word_ms(word: Word) -> int:
    """Rough speaking time of a word, from its letter count."""
    return EDGE_BASE_MS + EDGE_PER_LETTER_MS * sum(ch.isalnum() for ch in word.text)


def repair_edge_words(words: Sequence[Word]) -> tuple[Word, ...]:
    """Re-time a first/last word that Whisper stretched into a pause.

    Whisper sometimes starts a segment's first word at the end of the previous segment, so
    it covers the silence before the real speech (or the last word runs on into the silence
    after it). Such a word is abnormally long and separated from its neighbour by a clear
    pause; it is moved to sit directly against that neighbour.
    """
    fixed = list(words)
    if len(fixed) < 2:
        return tuple(fixed)
    first, second = fixed[0], fixed[1]
    length = expected_word_ms(first)
    if second.start_ms - first.end_ms >= EDGE_PAUSE_MS and first.end_ms - first.start_ms > 2 * length:
        end = second.start_ms - EDGE_TOUCH_MS
        fixed[0] = Word(first.text, end - length, end)
    prev, last = fixed[-2], fixed[-1]
    length = expected_word_ms(last)
    if last.start_ms - prev.end_ms >= EDGE_PAUSE_MS and last.end_ms - last.start_ms > 2 * length:
        start = prev.end_ms + EDGE_TOUCH_MS
        fixed[-1] = Word(last.text, start, start + length)
    return tuple(fixed)


def drop_segment_loops(segments: Sequence[Segment]) -> list[Segment]:
    """Keep a single copy of a segment text that repeats LOOP_RUN or more times in a row."""
    result: list[Segment] = []
    i = 0
    while i < len(segments):
        key, j = normalize(segments[i].text), i
        while j + 1 < len(segments) and normalize(segments[j + 1].text) == key:
            j += 1
        result.extend(segments[i:i + 1] if j - i + 1 >= LOOP_RUN else segments[i:j + 1])
        i = j + 1
    return result


def clean_segments(segments: Sequence[Segment]) -> list[Segment]:
    ordered = sorted((s for s in segments if not is_filler(s)), key=lambda s: s.start_ms)
    return drop_segment_loops(ordered)


def cues_from_segments(segments: Sequence[Segment], mask: Callable[[str], str] | None = None
                       ) -> list[Cue]:
    """Cues for all chunks (each chunk is cut on its own so languages never share a cue)."""
    cues: list[Cue] = []
    for chunk in sorted({s.chunk for s in segments}):
        words: list[Word] = []
        for seg in clean_segments([s for s in segments if s.chunk == chunk]):
            words.extend(collapse_word_loops(repair_edge_words(seg.words)))
        cues.extend(build_cues(words))
    cues = fix_timing(cues)
    if mask is not None:
        cues = [Cue(c.start_ms, c.end_ms, tuple(mask(line) for line in c.lines)) for c in cues]
    return cues
