"""Console progress bar with percent, elapsed time and estimated remaining time."""
from __future__ import annotations

import sys
import time
from typing import Callable, TextIO

BAR_WIDTH = 28
REDRAW_S = 0.5
LOG_EVERY_PERCENT = 10      # when output is not a terminal: one line per this many percent


def format_duration(seconds: float) -> str:
    """'H:MM:SS' (whole seconds, rounded down)."""
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Progress:
    """Tracks position (ms) in `total_ms` of audio; `start_ms` is where a resumed run began."""

    def __init__(self, label: str, total_ms: int, start_ms: int = 0, *,
                 stream: TextIO | None = None, clock: Callable[[], float] = time.monotonic,
                 is_tty: bool | None = None) -> None:
        self.label, self.total_ms, self.start_ms = label, max(1, total_ms), start_ms
        self._stream = stream or sys.stderr
        self._clock = clock
        self._tty = self._stream.isatty() if is_tty is None else is_tty
        self._began = clock()
        self._last_draw = -REDRAW_S
        self._last_logged = -1
        self.position_ms = start_ms

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._began

    @property
    def processed_s(self) -> float:
        """Seconds of audio handled in this run (excludes what a resumed run already had)."""
        return max(0, self.position_ms - self.start_ms) / 1000

    @property
    def fraction(self) -> float:
        return min(1.0, self.position_ms / self.total_ms)

    def eta_s(self) -> float | None:
        """Remaining seconds from this run's speed, None until there is something to go on."""
        done = self.position_ms - self.start_ms
        if done <= 0 or self.elapsed_s <= 0:
            return None
        return self.elapsed_s * (self.total_ms - self.position_ms) / done

    def real_time_factor(self) -> float | None:
        """Processing time divided by audio time (1.0 = as long as the audio)."""
        return self.elapsed_s / self.processed_s if self.processed_s > 0 else None

    def line(self) -> str:
        eta = self.eta_s()
        filled = int(BAR_WIDTH * self.fraction)
        return (f"{self.label} [{'#' * filled}{'-' * (BAR_WIDTH - filled)}] "
                f"{self.fraction * 100:5.1f}%  elapsed {format_duration(self.elapsed_s)}  "
                f"ETA {format_duration(eta) if eta is not None else '--:--:--'}")

    def update(self, position_ms: int) -> None:
        self.position_ms = max(self.position_ms, min(position_ms, self.total_ms))
        if self._tty:
            if self.elapsed_s - self._last_draw >= REDRAW_S:
                self._draw()
        else:
            step = int(self.fraction * 100) // LOG_EVERY_PERCENT
            if step > self._last_logged:
                self._last_logged = step
                self._stream.write(self.line() + "\n")
                self._stream.flush()

    def _draw(self) -> None:
        self._last_draw = self.elapsed_s
        self._stream.write("\r" + self.line())
        self._stream.flush()

    def finish(self) -> None:
        """Show 100% and end the line."""
        self.position_ms = self.total_ms
        if self._tty:
            self._draw()
            self._stream.write("\n")
        else:
            self._stream.write(self.line() + "\n")
        self._stream.flush()
