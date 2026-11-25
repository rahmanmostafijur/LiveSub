"""LiveSub: live subtitles for any audio playing on this PC.

Pipeline (each stage in its own thread, connected by queues):
    capture (WASAPI loopback) -> segmenter (VAD) -> transcriber (Whisper) -> overlay (Tk)
A supervisor restarts dead threads; hotkeys and the tray icon control the overlay.
"""
from __future__ import annotations

import argparse
import ctypes
import logging
import os
import queue
import signal
import sys
import threading
import time

from config import MODES, AppConfig, app_dir, load_config

SEGMENT_QUEUE_SIZE = 20
SUBTITLE_QUEUE_SIZE = 20
MUTEX_NAME = "Local\\LiveSubSingleInstance"
ERROR_ALREADY_EXISTS = 183
LOG_FILE = "livesub.log"

log = logging.getLogger("livesub")


def build_arg_parser() -> argparse.ArgumentParser:
    """Command-line options; anything not given comes from config.json."""
    from transcribe import _language_arg

    parser = argparse.ArgumentParser(description="Live subtitles for system audio")
    parser.add_argument("--mode", choices=MODES, type=str.lower,
                        help="start in 'original' or 'english' (translation) mode")
    parser.add_argument("--language", type=_language_arg,
                        help="force a language code such as en, ko, hi, or 'auto'")
    parser.add_argument("--model", help="Whisper model, e.g. small or base")
    parser.add_argument("--exit-after", type=float, metavar="SECONDS",
                        help="quit automatically after this many seconds (for testing)")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    return parser


def ensure_std_streams() -> bool:
    """A windowed .exe has no console: give libraries a harmless stdout/stderr.

    Returns True if a console is attached.
    """
    has_console = sys.stderr is not None
    if not has_console:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return has_console


def enable_ctrl_c() -> None:
    """Re-enable Ctrl+C for this console.

    Windows passes an "ignore Ctrl+C" flag from parent to child processes, so when
    LiveSub is started by some launchers or IDE runners, Ctrl+C would do nothing.
    """
    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    except (AttributeError, OSError):
        pass


def show_message(text: str) -> None:
    """Message box for errors the user must see even without a console."""
    try:
        ctypes.windll.user32.MessageBoxW(None, text, "LiveSub", 0x40)
    except (AttributeError, OSError):
        pass


def acquire_single_instance():
    """Return a mutex handle, or None if LiveSub is already running."""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return handle


class App:
    """Owns the pipeline threads, overlay, hotkeys, tray and supervisor."""

    def __init__(self, cfg: AppConfig, args: argparse.Namespace, has_console: bool) -> None:
        from capture import AudioCapture, make_audio_queue
        from models import approx_size_text, is_downloaded
        from profanity import ProfanityFilter
        from segmenter import SegmenterWorker
        from transcribe import Transcriber, config_from_args

        self.cfg = cfg
        self.args = args
        self.has_console = has_console
        self.stop_event = threading.Event()
        self.tcfg = config_from_args(args, cfg.transcribe)
        self.model_present = is_downloaded(self.tcfg.model, self.tcfg.model_path)
        self.model_size = approx_size_text(self.tcfg.model)
        self.audio_q = make_audio_queue(cfg.audio)
        self.segment_q: queue.Queue = queue.Queue(maxsize=SEGMENT_QUEUE_SIZE)
        self.subtitle_q: queue.Queue = queue.Queue(maxsize=SUBTITLE_QUEUE_SIZE)
        self.transcriber = Transcriber(self.segment_q, self.subtitle_q, self.tcfg,
                                       text_filter=ProfanityFilter.from_config(cfg.profanity))
        self.capture = AudioCapture(self.audio_q, cfg.audio)
        self.segmenter = SegmenterWorker(self.audio_q, self.segment_q, cfg.vad,
                                         cfg.audio.sample_rate)
        self.log_path = app_dir() / LOG_FILE
        self.overlay = None
        self.hotkeys = None
        self.tray = None
        self.supervisor = None

    # -- status shown on the overlay ---------------------------------------------

    def status(self) -> str | None:
        """Overlay message while the model is not ready yet (or failed)."""
        if self.transcriber.error:
            return "LiveSub: speech model could not be loaded - see livesub.log"
        if not self.transcriber.ready.is_set():
            if not self.model_present:
                return (f"LiveSub: downloading speech model ({self.model_size}, "
                        "first run only)...")
            return "LiveSub: loading speech model..."
        return None

    def health_report(self) -> str:
        from health import process_memory_mb

        now = time.monotonic()

        def ago(t: float) -> str:
            return f"{now - t:.0f}s ago" if t else "never"

        return (f"memory {process_memory_mb():.0f} MB | audio {ago(self.capture.last_audio)} | "
                f"subtitle {ago(self.transcriber.last_output)} | capture reopens "
                f"{self.capture.restarts} | queues audio {self.audio_q.qsize()} "
                f"segments {self.segment_q.qsize()} subtitles {self.subtitle_q.qsize()}")

    def _can_restart(self, name: str) -> bool:
        if self.stop_event.is_set():
            return False
        if name == "transcriber":   # do not loop on a model that cannot load
            return self.transcriber.ready.is_set() and not self.transcriber.error
        return True

    # -- lifecycle ------------------------------------------------------------------

    def run(self) -> int:
        """Start everything, run the overlay until quit, then stop in order."""
        from health import Supervisor
        from overlay import SubtitleOverlay, enable_dpi_awareness

        if not self.model_present:
            log.info("First run: downloading Whisper model '%s' (%s) into %s. This needs an "
                     "internet connection once; later starts work offline.",
                     self.tcfg.model, self.model_size, self.tcfg.model_path)
        self.transcriber.start()
        self.capture.start()
        self.segmenter.start()
        self.supervisor = Supervisor({"capture": self.capture, "segmenter": self.segmenter,
                                      "transcriber": self.transcriber},
                                     self._can_restart, self.health_report)
        self.supervisor.start()
        try:
            enable_dpi_awareness()
            self.overlay = SubtitleOverlay(self.cfg.overlay, self.subtitle_q, self.transcriber,
                                           self.status, self.stop_event)
            self._start_controls()
            threading.Thread(target=self._report_model_failure, daemon=True).start()
            self.overlay.run()
        except ValueError as exc:  # e.g. an invalid hotkey in config.json
            log.error("Startup failed: %s", exc)
            if not self.has_console:
                show_message(f"LiveSub could not start:\n{exc}")
            return 2
        finally:
            self.shutdown()
        return 0

    def _start_controls(self) -> None:
        from hotkeys import GlobalHotkeys
        from tray import TrayIcon

        ov = self.cfg.overlay
        self.hotkeys = GlobalHotkeys({"toggle": ov.hotkey_toggle, "mode": ov.hotkey_mode,
                                      "language": ov.hotkey_language, "quit": ov.hotkey_quit},
                                     self.overlay.actions.put)
        self.hotkeys.start()
        log.info("Hotkeys: %s show/hide | %s original/english | %s language | %s quit",
                 ov.hotkey_toggle, ov.hotkey_mode, ov.hotkey_language, ov.hotkey_quit)
        if ov.tray_icon:
            self.tray = TrayIcon(self.overlay.actions, lambda: self.transcriber.mode,
                                 lambda: self.transcriber.language,
                                 lambda: self.overlay.visible, ov.language_choices,
                                 self.log_path)
            if self.tray.start():
                log.info("Tray icon ready (right-click it for settings)")

    def _report_model_failure(self) -> None:
        self.transcriber.ready.wait()
        if self.transcriber.error and not self.has_console:
            show_message("LiveSub could not load the speech model.\n\n"
                         f"{self.transcriber.error}\n\nOn the first start an internet "
                         "connection is needed to download it (once). Details: livesub.log")

    def shutdown(self) -> None:
        """Stop all threads; safe to call more than once."""
        log.info("Shutting down...")
        self.stop_event.set()
        for part in (self.supervisor, self.tray, self.hotkeys):
            if part:
                part.stop()
        self.capture.stop()
        self.segmenter.stop()
        self.transcriber.stop()
        log.info("Stopped")


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python main.py` and the packaged LiveSub.exe."""
    from logsetup import install_exception_hooks, setup_logging

    has_console = ensure_std_streams()
    args = build_arg_parser().parse_args(argv)
    log_path = setup_logging(logging.DEBUG if args.debug else logging.INFO,
                             log_file=app_dir() / LOG_FILE,
                             stream=sys.stderr if has_console else None)
    install_exception_hooks()
    try:
        return _run_app(args, has_console, log_path)
    except Exception as exc:  # anything unexpected must be visible, even without console
        log.exception("LiveSub failed to start")
        if not has_console:
            show_message(f"LiveSub stopped because of an error:\n{exc}\n\nDetails: {log_path}")
        return 1


def _run_app(args: argparse.Namespace, has_console: bool, log_path) -> int:
    mutex = acquire_single_instance()
    if mutex is None:
        log.error("LiveSub is already running (look for its tray icon)")
        if not has_console:
            show_message("LiveSub is already running.\nUse its tray icon or Ctrl+Alt+Q to quit it.")
        return 1
    try:
        cfg = load_config()
    except ValueError as exc:
        log.error("Invalid config.json: %s", exc)
        if not has_console:
            show_message(f"config.json is invalid:\n{exc}")
        return 2

    app = App(cfg, args, has_console)
    app.log_path = log_path or app_dir() / LOG_FILE
    log.info("LiveSub starting | model %s | mode %s | language %s | profanity filter %s",
             app.tcfg.model, app.tcfg.mode, app.tcfg.language or "auto-detect",
             "on" if cfg.profanity.enabled else "off")
    if has_console:
        enable_ctrl_c()
    for sig in (signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            signal.signal(sig, lambda *_: app.stop_event.set())
    if args.exit_after:
        timer = threading.Timer(args.exit_after, app.stop_event.set)
        timer.daemon = True
        timer.start()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
