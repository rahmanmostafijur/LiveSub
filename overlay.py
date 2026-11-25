"""Borderless, transparent, always-on-top, click-through subtitle window (tkinter).

Runs on the main thread. Every POLL_MS it drains the subtitle queue and the
action queue (filled by hotkeys and the tray menu), updates the composer, and
redraws. A small notice row above the subtitles shows mode/language changes.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Protocol

from config import OverlayConfig, language_name
from subtitle_text import SubtitleComposer

log = logging.getLogger(__name__)

POLL_MS = 50
TOPMOST_REFRESH_S = 2.0       # some players steal "topmost"; re-assert it regularly
TRANSPARENT_KEY = "#010101"   # near-black: anti-aliased edges blend into the outline
PADDING_PX = 10
NOTICE_FONT_RATIO = 0.55
OUTLINE_DIRECTIONS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020   # mouse clicks pass through
WS_EX_TOOLWINDOW = 0x00000080    # no taskbar button / Alt+Tab entry
WS_EX_NOACTIVATE = 0x08000000    # never takes keyboard focus


class Controller(Protocol):
    """What the overlay needs from the transcriber."""

    @property
    def mode(self) -> str: ...
    @property
    def language(self) -> str: ...
    def toggle_mode(self) -> str: ...
    def set_mode(self, mode: str) -> None: ...
    def set_language(self, language: str) -> None: ...


def enable_dpi_awareness() -> None:
    """Crisp text and true pixel sizes on scaled displays (call before creating Tk)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass  # older Windows: text is slightly blurry but works


def make_click_through(root: tk.Tk) -> None:
    """Add layered/transparent/tool-window styles to the Tk top-level window."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT
                              | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    except (AttributeError, OSError) as exc:
        log.warning("Could not make the overlay click-through: %s", exc)


def describe(mode: str, language: str) -> str:
    """Notice text such as 'English subtitles · from Korean'."""
    mode_txt = "English subtitles" if mode == "english" else "Original language"
    return f"{mode_txt} · {language_name(language)}"


class SubtitleOverlay:
    """The on-screen subtitle window and its event loop."""

    def __init__(self, cfg: OverlayConfig, subtitles: queue.Queue, controller: Controller,
                 status: Callable[[], str | None], stop_event: threading.Event) -> None:
        self._cfg = cfg
        self._subtitles = subtitles
        self._ctl = controller
        self._status_fn = status
        self._stop = stop_event
        self.actions: queue.Queue[str] = queue.Queue()   # filled by hotkeys / tray
        self.visible = True
        self.notice = ""
        self._notice_until = 0.0
        self._detected: str | None = None
        self._drawn: tuple | None = None
        self._next_topmost = 0.0

        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("LiveSub")
        self.root.overrideredirect(True)
        self.root.configure(bg=TRANSPARENT_KEY)
        weight = "bold" if cfg.bold else "normal"
        self._font = tkfont.Font(root=self.root, family=cfg.font_family, size=cfg.font_size,
                                 weight=weight)
        self._notice_font = tkfont.Font(root=self.root, family=cfg.font_family, weight="bold",
                                        size=max(10, int(cfg.font_size * NOTICE_FONT_RATIO)))
        self._line_h = self._font.metrics("linespace")
        self._notice_h = self._notice_font.metrics("linespace")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self._text_w = int(screen_w * cfg.width_ratio)
        margin = PADDING_PX + cfg.outline_px
        self._width = self._text_w + 2 * margin
        self._height = self._notice_h + cfg.max_lines * self._line_h + 2 * margin
        x = (screen_w - self._width) // 2
        y = screen_h - self._height - cfg.bottom_margin_px
        self.root.geometry(f"{self._width}x{self._height}+{x}+{y}")
        self._canvas = tk.Canvas(self.root, width=self._width, height=self._height,
                                 bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self._canvas.pack(fill="both", expand=True)
        self.composer = SubtitleComposer(self._font.measure, self._text_w, cfg.max_lines,
                                         cfg.merge_window_s, cfg.clear_after_s)
        self.root.update_idletasks()
        make_click_through(self.root)
        self.root.attributes("-transparentcolor", TRANSPARENT_KEY)
        self.root.deiconify()
        self._raise()

    # -- public -------------------------------------------------------------

    def run(self) -> None:
        """Run the GUI loop until quit (hotkey, tray, Ctrl+C or stop_event)."""
        self.root.after(POLL_MS, self._poll)
        try:
            self.root.mainloop()
        finally:
            try:
                self.root.destroy()
            except tk.TclError:
                pass

    def show_notice(self, text: str, now: float | None = None) -> None:
        """Show a short message in the notice row above the subtitles."""
        now = time.monotonic() if now is None else now
        self.notice = text
        self._notice_until = now + self._cfg.status_s
        log.info("Notice: %s", text)

    def tick(self, now: float | None = None) -> list[str]:
        """One update step: handle actions and new subtitles, redraw; returns shown lines."""
        now = time.monotonic() if now is None else now
        self._handle_actions(now)
        while True:
            try:
                subtitle = self._subtitles.get_nowait()
            except queue.Empty:
                break
            if self._is_stale(subtitle):
                log.debug("Dropped subtitle from before a mode/language switch: %s",
                          subtitle.text)
                continue
            log.info("[%s %s %.1fs] %s", subtitle.mode, subtitle.language,
                     subtitle.latency_s, subtitle.text)
            self._note_detected_language(subtitle.language, now)
            self.composer.add(subtitle.text, now)
        lines = self._current_lines(now)
        notice = self.notice if now < self._notice_until else ""
        if self.visible and (lines, notice) != self._drawn:
            log.debug("Overlay shows: %s", " / ".join(lines))
            self._draw(lines, notice)
        if now >= self._next_topmost and self.visible:
            self._raise()
            self._next_topmost = now + TOPMOST_REFRESH_S
        return lines

    # -- internals ------------------------------------------------------------

    def _poll(self) -> None:
        if self._stop.is_set():
            self.root.quit()
            return
        try:
            self.tick()
        except Exception:
            log.exception("Overlay update failed")
        self.root.after(POLL_MS, self._poll)

    def _handle_actions(self, now: float) -> None:
        while True:
            try:
                action = self.actions.get_nowait()
            except queue.Empty:
                return
            name, _, arg = action.partition(":")
            try:
                self._dispatch(name, arg, now)
            except ValueError as exc:
                log.warning("Ignored action %r: %s", action, exc)

    def _dispatch(self, name: str, arg: str, now: float) -> None:
        if name == "quit":
            log.info("Quit requested")
            self._stop.set()
        elif name == "toggle":
            self._set_visible(not self.visible)
        elif name in ("show", "hide"):
            self._set_visible(name == "show")
        elif name == "mode":
            self._ctl.set_mode(arg) if arg else self._ctl.toggle_mode()
            self.composer.clear()
            self.show_notice(describe(self._ctl.mode, self._ctl.language), now)
        elif name == "language":
            if arg:
                self._ctl.set_language(arg)
            else:
                self._cycle_language()
            self._detected = None
            self.composer.clear()
            self.show_notice(describe(self._ctl.mode, self._ctl.language), now)

    def _is_stale(self, subtitle) -> bool:
        """Subtitle decoded before the user switched mode or forced another language."""
        if subtitle.mode != self._ctl.mode:
            return True
        forced = self._ctl.language
        return bool(forced) and subtitle.language != forced

    def _cycle_language(self) -> None:
        """Move to the next language choice, skipping codes the transcriber rejects."""
        choices = self._cfg.language_choices
        current = self._ctl.language
        index = choices.index(current) if current in choices else -1
        for step in range(1, len(choices) + 1):
            code = choices[(index + step) % len(choices)] or "auto"
            try:
                self._ctl.set_language(code)
                return
            except ValueError as exc:
                log.warning("Skipping language '%s' from overlay.languages: %s", code, exc)

    def _note_detected_language(self, language: str, now: float) -> None:
        """With auto-detect, briefly show when the detected language changes."""
        if self._ctl.language or language == self._detected:
            return
        if self._detected is not None:
            self.show_notice(f"Detected: {language_name(language)}", now)
        self._detected = language

    def _current_lines(self, now: float) -> list[str]:
        status = self._status_fn()
        if status:
            return [status]
        return self.composer.lines(now)

    def _set_visible(self, visible: bool) -> None:
        self.visible = visible
        log.info("Overlay %s", "shown" if visible else "hidden")
        if visible:
            self.root.deiconify()
            self._drawn = None
            self._raise()
        else:
            self.root.withdraw()

    def _raise(self) -> None:
        self.root.attributes("-topmost", True)
        self.root.lift()

    def _draw_text(self, x: int, y: int, text: str, font: tkfont.Font) -> None:
        for step in range(1, self._cfg.outline_px + 1):
            for dx, dy in OUTLINE_DIRECTIONS:
                self._canvas.create_text(x + dx * step, y + dy * step, text=text, font=font,
                                         fill=self._cfg.outline_color, anchor="n")
        self._canvas.create_text(x, y, text=text, font=font, fill=self._cfg.text_color,
                                 anchor="n")

    def _draw(self, lines: list[str], notice: str) -> None:
        self._canvas.delete("all")
        self._drawn = (list(lines), notice)
        center = self._width // 2
        top = PADDING_PX + self._cfg.outline_px
        if notice:
            self._draw_text(center, top, notice, self._notice_font)
        first_row = self._cfg.max_lines - len(lines)       # bottom-align the subtitles
        for i, line in enumerate(lines):
            y = top + self._notice_h + (first_row + i) * self._line_h
            self._draw_text(center, y, line, self._font)
