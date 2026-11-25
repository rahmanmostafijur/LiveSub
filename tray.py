"""System tray icon with a settings menu (show/hide, mode, language, log, quit).

Menu clicks only put action strings into the overlay's queue, exactly like the
hotkeys, so all changes happen on the GUI thread.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Callable

from config import language_name

log = logging.getLogger(__name__)

ICON_SIZE = 64


def make_icon_image():
    """A simple subtitle-style icon drawn with Pillow (no image files needed)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 8, 62, 56), radius=10, fill=(32, 46, 74, 255))
    draw.rounded_rectangle((12, 30, 52, 36), radius=3, fill=(255, 255, 255, 255))
    draw.rounded_rectangle((18, 41, 46, 47), radius=3, fill=(255, 210, 80, 255))
    return image


class TrayIcon:
    """Tray icon running in its own thread."""

    def __init__(self, actions: queue.Queue[str], get_mode: Callable[[], str],
                 get_language: Callable[[], str], is_visible: Callable[[], bool],
                 languages: list[str], log_file: Path) -> None:
        self._actions = actions
        self._get_mode = get_mode
        self._get_language = get_language
        self._is_visible = is_visible
        self._languages = languages
        self._log_file = log_file
        self._icon = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Show the icon; returns False (and logs why) if the tray is unavailable."""
        try:
            import pystray

            self._icon = pystray.Icon("LiveSub", make_icon_image(), "LiveSub",
                                      self._build_menu(pystray))
        except Exception as exc:
            log.warning("Tray icon unavailable (%s); use the hotkeys instead", exc)
            return False
        self._thread = threading.Thread(target=self._icon.run, name="Tray", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Remove the icon and end its thread."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception as exc:
                log.debug("Tray stop: %s", exc)
        if self._thread:
            self._thread.join(2.0)

    def _send(self, action: str) -> Callable:
        return lambda icon, item: self._actions.put(action)

    def _build_menu(self, pystray):
        item, menu = pystray.MenuItem, pystray.Menu
        mode_items = [
            item("Original language", self._send("mode:original"), radio=True,
                 checked=lambda i: self._get_mode() == "original"),
            item("English translation", self._send("mode:english"), radio=True,
                 checked=lambda i: self._get_mode() == "english"),
        ]
        language_items = [
            item(language_name(code), self._send(f"language:{code or 'auto'}"), radio=True,
                 checked=lambda i, code=code: self._get_language() == code)
            for code in self._languages
        ]
        return menu(
            item("Show subtitles", self._send("toggle"), default=True,
                 checked=lambda i: self._is_visible()),
            item("Mode", menu(*mode_items)),
            item("Language", menu(*language_items)),
            menu.SEPARATOR,
            item("Open log file", lambda icon, i: self._open_log()),
            item("Quit LiveSub", self._send("quit")),
        )

    def _open_log(self) -> None:
        try:
            os.startfile(self._log_file)  # opens in the default text editor
        except OSError as exc:
            log.warning("Could not open %s: %s", self._log_file, exc)
