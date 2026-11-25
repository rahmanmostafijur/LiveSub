"""Decide what the overlay shows: merge sentence pieces, wrap, and expire text.

Pure logic (no GUI): time and text measurement are passed in, so it is testable.
"""
from __future__ import annotations

from typing import Callable

SENTENCE_END = ".?!。？！।"           # '।' is the Bengali/Hindi full stop
TRAILING_CLOSERS = "\"')]}»”’"
ELLIPSES = ("...", "…")             # Whisper marks an unfinished thought this way

Measure = Callable[[str], int]       # text -> width in pixels


def ends_sentence(text: str) -> bool:
    """True if text ends with . ? ! (ignoring closing quotes), but not with '...'."""
    stripped = text.rstrip().rstrip(TRAILING_CLOSERS).rstrip()
    if not stripped or stripped.endswith(ELLIPSES):
        return False
    return stripped[-1] in SENTENCE_END


def starts_lowercase(text: str) -> bool:
    """True if the first letter is lower-case (Whisper's sign of a continued sentence).

    Scripts without letter case (Korean, Hindi, ...) never count.
    """
    for ch in text:
        if ch.isalpha():
            return ch.islower() and ch.upper() != ch
    return False


def wrap(text: str, measure: Measure, max_width: int) -> list[str]:
    """Greedy word wrap by pixel width; words wider than a line are split."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if measure(candidate) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = ""
        while measure(word) > max_width and len(word) > 1:  # e.g. CJK without spaces
            cut = len(word)
            while cut > 1 and measure(word[:cut]) > max_width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines


class SubtitleComposer:
    """Holds the current subtitle and applies merge / wrap / clear rules."""

    def __init__(self, measure: Measure, max_width: int, max_lines: int,
                 merge_window_s: float, clear_after_s: float) -> None:
        self._measure = measure
        self._max_width = max_width
        self._max_lines = max_lines
        self._merge_window = merge_window_s
        self._clear_after = clear_after_s
        self._text = ""
        self._updated = float("-inf")

    def add(self, text: str, now: float) -> list[str]:
        """Add a new piece; continue the current sentence or start a new subtitle.

        A piece continues the subtitle when it arrives within the merge window and
        the subtitle has no closing . ? ! - or the piece starts in lower case.
        """
        text = text.strip()
        if not text:
            return self.lines(now)
        recent = now - self._updated <= self._merge_window
        continues = not ends_sentence(self._text) or starts_lowercase(text)
        if self.lines(now) and recent and continues:
            self._text = f"{self._text} {text}"
        else:
            self._text = text
        self._updated = now
        # keep only what fits, so a long run of pieces scrolls instead of growing
        self._text = " ".join(wrap(self._text, self._measure,
                                   self._max_width)[-self._max_lines:])
        return self.lines(now)

    def lines(self, now: float) -> list[str]:
        """Lines to show now (empty once the subtitle has expired)."""
        if not self._text or now - self._updated > self._clear_after:
            return []
        return wrap(self._text, self._measure, self._max_width)[-self._max_lines:]

    def clear(self) -> None:
        """Forget the current subtitle (e.g. when the mode changes)."""
        self._text = ""
        self._updated = float("-inf")
