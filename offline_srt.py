"""SRT formatting, writing and read-back verification. All times are integer milliseconds."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Sequence

from offline_cues import Cue

BOM = b"\xef\xbb\xbf"
_TIME = r"(\d+):(\d{2}):(\d{2}),(\d{3})"
_TIMING_LINE = re.compile(rf"^{_TIME} --> {_TIME}\s*$")


def seconds_to_ms(seconds: float) -> int:
    """Whole milliseconds, rounded half up; negative input becomes 0."""
    return max(0, math.floor(seconds * 1000 + 0.5))


def format_timestamp(ms: int) -> str:
    """'HH:MM:SS,mmm'. Works on integers, so a rounded-up 999.6 ms can never print as ',1000'."""
    ms = max(0, int(ms))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_timestamp(text: str) -> int:
    """Inverse of format_timestamp."""
    match = re.fullmatch(_TIME, text.strip())
    if not match:
        raise ValueError(f"bad SRT timestamp: {text!r}")
    h, m, s, ms = (int(g) for g in match.groups())
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def format_srt(cues: Sequence[Cue]) -> str:
    """The SRT document (LF newlines) for numbered cues."""
    blocks = []
    for number, cue in enumerate(cues, start=1):
        timing = f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}"
        blocks.append("\n".join((str(number), timing, *cue.lines)))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_srt(text: str) -> list[Cue]:
    """Read an SRT document back into cues (used to verify what was written)."""
    cues: list[Cue] = []
    if not text.strip():
        return cues
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError(f"malformed SRT block: {block[:60]!r}")
        match = _TIMING_LINE.match(lines[1])
        if not match:
            raise ValueError(f"malformed SRT timing line: {lines[1]!r}")
        g = [int(x) for x in match.groups()]
        start = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        cues.append(Cue(start, end, tuple(lines[2:])))
    return cues


def write_srt(path: Path, cues: Sequence[Cue]) -> None:
    """Write UTF-8 without BOM (CRLF line ends, the SRT convention), then read it back.

    Raises ValueError if the file on disk does not decode to exactly the cues written.
    """
    data = format_srt(cues).replace("\n", "\r\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    verify_srt(path, cues)


def verify_srt(path: Path, expected: Sequence[Cue]) -> None:
    """Check a written SRT: no BOM, strict UTF-8, no replacement characters, same cues."""
    raw = path.read_bytes()
    if raw.startswith(BOM):
        raise ValueError(f"{path} starts with a BOM")
    text = raw.decode("utf-8")  # strict: raises on invalid bytes
    if "�" in text:
        raise ValueError(f"{path} contains replacement characters")
    if parse_srt(text) != list(expected):
        raise ValueError(f"{path} does not read back as the cues that were written")
