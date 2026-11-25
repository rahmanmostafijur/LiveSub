"""Text clean-up for Whisper output: overlap removal, repeat collapse, junk filter."""
from __future__ import annotations

import re
import zlib

MAX_OVERLAP_WORDS = 8          # longest repeated phrase removed at a segment boundary
REPEAT_COLLAPSE_MIN = 3        # n-grams repeated this many times in a row are collapsed
MAX_NGRAM = 8
MAX_COMPRESSION_RATIO = 2.4    # Whisper's own threshold for repetitive output
# (min length, max distinct letters): e.g. 'ლლლლ' or 'हेहेहेहे' is noise, not speech
JUNK_RULES = ((4, 1), (6, 2))

# Phrases Whisper often invents on music, noise or silence (compared normalized).
HALLUCINATION_EXACT = {
    "you", "thank you for watching", "thanks for watching", "please subscribe",
    "subscribe to my channel", "like and subscribe", "bye", "music",
}
HALLUCINATION_SUBSTRINGS = (
    "amara.org", "subtitles by", "transcribed by", "radio-canada",
    "ご視聴ありがとうございました", "字幕由", "thanks for watching",
    "thank you for watching", "don't forget to subscribe",
)

_WORD = re.compile(r"[\w']+", re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case text with punctuation removed and spaces collapsed."""
    return " ".join(_WORD.findall(text.lower()))


def compression_ratio(text: str) -> float:
    """Raw/zlib size ratio; high values mean repetitive text."""
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def _key(words: list[str]) -> list[str]:
    return [w.lower().strip(",.!?") for w in words]


def collapse_repeats(text: str) -> str:
    """Collapse a word sequence repeated 3+ times in a row to a single copy."""
    words = text.split()
    for n in range(1, MAX_NGRAM + 1):
        out: list[str] = []
        i = 0
        while i < len(words):
            chunk = _key(words[i:i + n])
            count = 1
            while _key(words[i + count * n:i + (count + 1) * n]) == chunk:
                count += 1
            if len(chunk) == n and count >= REPEAT_COLLAPSE_MIN:
                out.extend(words[i:i + n])
                i += count * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def strip_broken(text: str) -> str:
    """Remove replacement characters left by a cut-off multi-byte token."""
    return text.replace("�", "").strip()


def is_hallucination(text: str) -> bool:
    """True for empty output, known filler phrases, or one-symbol noise like 'ლლლლლლ'."""
    norm = normalize(text)
    if not norm:
        return True
    if norm in HALLUCINATION_EXACT:
        return True
    letters = norm.replace(" ", "")
    if any(len(letters) >= length and len(set(letters)) <= distinct
           for length, distinct in JUNK_RULES):
        return True
    lowered = text.lower()
    return any(s in lowered for s in HALLUCINATION_SUBSTRINGS)


def echoes_context(context: str, text: str) -> bool:
    """True if `text` merely repeats (part of) the context Whisper was given."""
    norm = normalize(text)
    return bool(norm) and norm in normalize(context)


def remove_overlap(previous: str, current: str, max_words: int = MAX_OVERLAP_WORDS) -> str:
    """Drop the start of `current` that repeats the end of `previous`."""
    prev_words = normalize(previous).split()
    cur_words = current.split()
    cur_norm = [normalize(w) for w in cur_words]
    limit = min(max_words, len(prev_words), len(cur_words))
    for n in range(limit, 0, -1):
        if prev_words[-n:] == cur_norm[:n]:
            return " ".join(cur_words[n:])
    return current
