"""Overlay, hotkey, tray and masking integration tests (Windows desktop required for Tk)."""
import queue
import threading

import numpy as np
import pytest

from config import OverlayConfig, TranscribeConfig
from hotkeys import GlobalHotkeys, parse_hotkey
from profanity import ProfanityFilter
from segmenter import SAMPLE_RATE, Segment
from transcribe import Recognition, Subtitle, Transcriber


# -- hotkeys ------------------------------------------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("ctrl+alt+s", (0x0003, ord("S"))),
    ("Ctrl+Shift+F9", (0x0006, 0x78)),
    ("win+alt+space", (0x0009, 0x20)),
])
def test_parse_hotkey(spec, expected):
    assert parse_hotkey(spec) == expected


@pytest.mark.parametrize("spec", ["", "s", "ctrl+", "ctrl+alt+bogus", "hyper+s"])
def test_parse_hotkey_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_hotkey(spec)


def test_hotkey_thread_starts_and_stops_cleanly():
    hk = GlobalHotkeys({"x": "ctrl+alt+shift+f24"}, lambda action: None)
    hk.start()
    hk.stop()
    assert not hk._thread.is_alive()


# -- masking through the transcriber -------------------------------------------

class OneShotEngine:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def recognize(self, audio, task, prompt):
        self.prompts.append(prompt)
        return Recognition(self.text, "en", 0.9, -0.2, 0.01)


def test_transcriber_masks_subtitle_but_keeps_raw_context():
    engine = OneShotEngine("what the fuck,")
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, TranscribeConfig(), lambda _c: engine,
                         text_filter=ProfanityFilter(["fuck"]))
    worker.start()
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    in_q.put(Segment(audio, 0.0, 1.0))
    assert out_q.get(timeout=2).text == "what the f**k,"
    in_q.put(Segment(audio, 1.5, 2.5))
    out_q.get(timeout=2)
    worker.stop()
    assert engine.prompts[1] == "what the fuck,"    # Whisper context stays unmasked


# -- tray menu ------------------------------------------------------------------

def test_tray_menu_items_send_actions():
    pystray = pytest.importorskip("pystray")
    from tray import TrayIcon

    actions: queue.Queue = queue.Queue()
    tray = TrayIcon(actions, lambda: "original", lambda: "", lambda: True,
                    ["", "en", "ko"], None)
    menu = tray._build_menu(pystray)
    by_text = {item.text: item for item in menu.items if item.text}
    by_text["Show subtitles"](None)
    by_text["Mode"].submenu.items[1](None)                 # English translation
    korean = [i for i in by_text["Language"].submenu.items if i.text == "Korean"][0]
    korean(None)
    by_text["Quit LiveSub"](None)
    sent = [actions.get_nowait() for _ in range(4)]
    assert sent == ["toggle", "mode:english", "language:ko", "quit"]
    auto = by_text["Language"].submenu.items[0]
    assert auto.text == "Auto-detect" and auto.checked


# -- overlay window -------------------------------------------------------------

class FakeController:
    def __init__(self):
        self.mode = "original"
        self.language = ""

    def toggle_mode(self):
        self.mode = "english" if self.mode == "original" else "original"
        return self.mode

    def set_mode(self, mode):
        self.mode = mode

    def set_language(self, language):
        if language not in ("auto", "en", "ko", "hi", "ml", "ta", "bn"):
            raise ValueError(f"unknown language {language}")
        self.language = "" if language == "auto" else language


@pytest.fixture
def overlay():
    tk = pytest.importorskip("tkinter")
    from overlay import SubtitleOverlay

    subs: queue.Queue = queue.Queue()
    ctl = FakeController()
    try:
        ov = SubtitleOverlay(OverlayConfig(merge_window_s=4, clear_after_s=3), subs, ctl,
                             lambda: None, threading.Event())
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    yield ov, subs, ctl
    ov.root.destroy()


def subtitle(text: str, language: str = "en") -> Subtitle:
    return Subtitle(text, language, "original", 0.0, 1.0, 1.0, 1.0)


def test_overlay_merges_unfinished_sentences_and_clears(overlay):
    ov, subs, _ = overlay
    subs.put(subtitle("Four score and seven years ago"))
    assert ov.tick(now=100.0) == ["Four score and seven years ago"]
    subs.put(subtitle("our fathers brought forth."))
    lines = ov.tick(now=101.0)
    assert " ".join(lines) == "Four score and seven years ago our fathers brought forth."
    assert len(lines) <= 2
    subs.put(subtitle("A new nation."))
    assert ov.tick(now=102.0) == ["A new nation."]     # previous ended with '.'
    assert ov.tick(now=106.0) == []                     # cleared after silence


def test_overlay_mode_toggle_and_quit_actions(overlay):
    ov, _, ctl = overlay
    ov.actions.put("mode")
    ov.tick(now=10.0)
    assert ctl.mode == "english" and ov.notice == "English subtitles · Auto-detect"
    ov.actions.put("mode:original")
    ov.tick(now=10.1)
    assert ctl.mode == "original"
    ov.actions.put("toggle")
    ov.tick(now=10.2)
    assert ov.visible is False
    ov.actions.put("show")
    ov.tick(now=10.3)
    assert ov.visible is True
    ov.actions.put("quit")
    ov.tick(now=10.4)
    assert ov._stop.is_set()


def test_overlay_language_hotkey_cycles_and_tray_sets(overlay):
    ov, _, ctl = overlay
    ov.actions.put("language")                 # auto -> en
    ov.tick(now=1.0)
    assert ctl.language == "en" and ov.notice == "Original language · English"
    ov.actions.put("language")                 # en -> ko
    ov.tick(now=1.1)
    assert ctl.language == "ko"
    ov.actions.put("language:auto")
    ov.tick(now=1.2)
    assert ctl.language == "" and ov.notice.endswith("Auto-detect")


def test_overlay_drops_subtitles_from_before_a_switch(overlay):
    ov, subs, ctl = overlay
    ov.actions.put("mode:english")
    subs.put(subtitle("Old original-mode text"))          # decoded before the switch
    assert ov.tick(now=1.0) == []
    ov.actions.put("language:ko")
    subs.put(Subtitle("late english line", "en", "english", 0, 1, 1, 1))
    assert ov.tick(now=1.1) == []


def test_language_cycle_skips_invalid_codes():
    tk = pytest.importorskip("tkinter")
    from overlay import SubtitleOverlay

    ctl = FakeController()
    try:
        ov = SubtitleOverlay(OverlayConfig(languages="auto,en,xx,ko"), queue.Queue(), ctl,
                             lambda: None, threading.Event())
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    try:
        ov.actions.put("language")
        ov.tick(now=1.0)
        ov.actions.put("language")
        ov.tick(now=1.1)
        assert ctl.language == "ko"                      # 'xx' was skipped
    finally:
        ov.root.destroy()


def test_overlay_announces_detected_language_change(overlay):
    ov, subs, _ = overlay
    subs.put(subtitle("Hello there.", "en"))
    ov.tick(now=1.0)
    assert ov.notice == ""                      # first detection: no notice
    subs.put(subtitle("안녕하세요.", "ko"))
    ov.tick(now=2.0)
    assert ov.notice == "Detected: Korean"


def test_overlay_shows_status_until_model_ready(overlay):
    ov, subs, _ = overlay
    ready = {"status": "LiveSub: loading speech model..."}
    ov._status_fn = lambda: ready["status"]
    assert ov.tick(now=1.0) == ["LiveSub: loading speech model..."]
    ready["status"] = None
    subs.put(subtitle("Hello."))
    assert ov.tick(now=2.0) == ["Hello."]
