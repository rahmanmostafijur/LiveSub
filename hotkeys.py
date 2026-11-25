"""System-wide hotkeys on Windows via RegisterHotKey (no extra packages).

The overlay never has keyboard focus (it is click-through), so hotkeys must be
global. A dedicated thread owns the registrations and its message loop; fired
actions are handed to a callback (the overlay queues them for the GUI thread).
"""
from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable

log = logging.getLogger(__name__)

MODIFIERS = {"alt": 0x0001, "ctrl": 0x0002, "control": 0x0002, "shift": 0x0004,
             "win": 0x0008}
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
NAMED_KEYS = {"space": 0x20, "enter": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09,
              "home": 0x24, "end": 0x23, "insert": 0x2D, "delete": 0x2E,
              "pageup": 0x21, "pagedown": 0x22}


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+s' -> (modifier flags, virtual-key code)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    *mods, key = parts
    flags = 0
    for mod in mods:
        if mod not in MODIFIERS:
            raise ValueError(f"unknown modifier '{mod}' in hotkey '{spec}'")
        flags |= MODIFIERS[mod]
    if not flags:
        raise ValueError(f"hotkey '{spec}' needs at least one of ctrl/alt/shift/win")
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key in NAMED_KEYS:
        vk = NAMED_KEYS[key]
    else:
        raise ValueError(f"unknown key '{key}' in hotkey '{spec}'")
    return flags, vk


class GlobalHotkeys:
    """Register hotkeys -> actions; callback(action) runs on the hotkey thread."""

    def __init__(self, bindings: dict[str, str], callback: Callable[[str], None]) -> None:
        self._bindings = {action: parse_hotkey(spec) for action, spec in bindings.items()}
        self._specs = bindings
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._started = threading.Event()

    def start(self) -> None:
        """Start the hotkey thread (returns once registrations were attempted)."""
        self._thread = threading.Thread(target=self._run, name="Hotkeys", daemon=True)
        self._thread.start()
        self._started.wait(2.0)

    def stop(self) -> None:
        """Unregister hotkeys and end the message loop."""
        if self._thread and self._thread.is_alive() and self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2.0)

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        ids: dict[int, str] = {}
        for hotkey_id, (action, (mods, vk)) in enumerate(self._bindings.items(), start=1):
            if user32.RegisterHotKey(None, hotkey_id, mods | MOD_NOREPEAT, vk):
                ids[hotkey_id] = action
                log.info("Hotkey %s -> %s", self._specs[action], action)
            else:
                log.warning("Hotkey %s for '%s' is already used by another program; "
                            "change it in config.json", self._specs[action], action)
        self._started.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in ids:
                    try:
                        self._callback(ids[msg.wParam])
                    except Exception:
                        log.exception("Hotkey action '%s' failed", ids[msg.wParam])
        finally:
            for hotkey_id in ids:
                user32.UnregisterHotKey(None, hotkey_id)
