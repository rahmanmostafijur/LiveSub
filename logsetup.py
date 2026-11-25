"""Logging that never breaks on non-ASCII text (Chinese, Hindi, Bengali, ...).

Windows consoles, pipes and IDE runners often use a legacy code page such as
cp1252 in strict mode; writing CJK or Indic text then raises inside the log
handler, prints "--- Logging error ---" and garbles the surrounding lines.
Here every console write falls back to escaped text, and an optional UTF-8
log file keeps the exact text.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_FILE_MAX_BYTES = 1_000_000
LOG_FILE_BACKUPS = 1


def _escape_for(stream: TextIO, text: str) -> str:
    """Re-encode text so it fits the stream's encoding (unknown chars -> \\uXXXX)."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    return text.encode(encoding, "backslashreplace").decode(encoding, "replace")


def make_stream_safe(stream: TextIO) -> None:
    """Switch a strict text stream to escaping instead of raising on unencodable text."""
    if isinstance(stream, io.TextIOWrapper) and stream.errors == "strict":
        try:
            stream.reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass  # stream unusable for reconfigure; SafeStreamHandler still protects logging


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that writes escaped text when the stream cannot encode a message."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record) + self.terminator
            try:
                self.stream.write(message)
            except UnicodeError:
                self.stream.write(_escape_for(self.stream, message))
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _file_handler(log_file: Path) -> tuple[logging.Handler | None, Path | None]:
    """UTF-8 rotating log; falls back to %LOCALAPPDATA%\\LiveSub if not writable."""
    fallback = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LiveSub" / log_file.name
    for path in (log_file, fallback):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return RotatingFileHandler(path, maxBytes=LOG_FILE_MAX_BYTES,
                                       backupCount=LOG_FILE_BACKUPS, encoding="utf-8"), path
        except OSError:
            continue
    return None, None


def install_exception_hooks() -> None:
    """Log uncaught exceptions from the main thread and from worker threads."""
    def main_hook(exc_type, exc, tb):
        logging.getLogger("livesub").critical("Uncaught error", exc_info=(exc_type, exc, tb))

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread else "?"
        logging.getLogger("livesub").error(
            "Uncaught error in thread %s", name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def setup_logging(level: int = logging.INFO, log_file: Path | None = None,
                  stream: TextIO | None = None) -> Path | None:
    """Configure the root logger: safe console output plus an optional UTF-8 file.

    Returns the log file actually used (it may be the fallback location), or None.
    """
    for std in (sys.stdout, sys.stderr):
        if std is not None:
            make_stream_safe(std)
    target = stream if stream is not None else sys.stderr
    if target is not None:
        make_stream_safe(target)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT)
    if target is not None:
        console = SafeStreamHandler(target)
        console.setFormatter(formatter)
        root.addHandler(console)
    used: Path | None = None
    if log_file is not None:
        file_handler, used = _file_handler(log_file)
        if file_handler is not None:
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
    root.setLevel(level)
    if log_file is not None and used != log_file:
        logging.getLogger(__name__).warning("Cannot write %s; logging to %s", log_file, used)
    return used
