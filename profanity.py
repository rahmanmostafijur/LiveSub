"""Mask profanity and slurs in subtitle text, e.g. 'fucking' -> 'f*****g'."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from config import ProfanityConfig, app_dir, resource_path

log = logging.getLogger(__name__)

DEFAULT_WORDS_FILE = "profanity_words.txt"
# endings that still count as the listed word ('fuck' -> 'fucking', 'fuckers')
SUFFIXES = ("s", "es", "ed", "er", "ers", "ing", "in", "in'")
MASK_CHAR = "*"


def mask_word(word: str) -> str:
    """Keep the first and last letter, replace the rest: 'shit' -> 's**t'."""
    if len(word) <= 2:
        return MASK_CHAR * len(word)
    return word[0] + MASK_CHAR * (len(word) - 2) + word[-1]


def load_words(path: Path) -> list[str]:
    """Read one word per line; blank lines and '#' comments are ignored."""
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip().lower()
        if entry:
            words.append(entry)
    return words


class ProfanityFilter:
    """Whole-word, case-insensitive masking of a configurable word list."""

    def __init__(self, words: list[str], enabled: bool = True) -> None:
        self.enabled = enabled and bool(words)
        stems = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
        suffixes = "|".join(re.escape(s) for s in sorted(SUFFIXES, key=len, reverse=True))
        self._pattern = re.compile(
            r"(?<![\w])(?:" + "|".join(re.escape(s) for s in stems) + r")(?:" + suffixes
            + r")?(?![\w])", re.IGNORECASE) if stems else None

    @classmethod
    def from_config(cls, cfg: ProfanityConfig) -> "ProfanityFilter":
        """Build from config: word file (built-in by default) plus extra words."""
        if not cfg.enabled:
            return cls([], enabled=False)
        path = Path(cfg.words_file) if cfg.words_file else resource_path(DEFAULT_WORDS_FILE)
        if not path.is_absolute():
            path = app_dir() / path
        try:
            words = load_words(path)
        except OSError as exc:
            log.error("Profanity word list %s could not be read (%s); filter uses only "
                      "extra_words", path, exc)
            words = []
        words += [w for w in cfg.extra_words.split(",") if w.strip()]
        return cls(words, enabled=True)

    def __call__(self, text: str) -> str:
        """Return text with listed words masked (unchanged if disabled)."""
        if not self.enabled or self._pattern is None:
            return text
        return self._pattern.sub(lambda m: mask_word(m.group(0)), text)
