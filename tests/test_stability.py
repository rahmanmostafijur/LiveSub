"""Capture watchdog/recovery, supervisor restarts, and app status messages."""
import argparse
import queue
import time

import pytest

from capture import AudioCapture, CaptureStalled
from config import AppConfig, AudioConfig
from health import Supervisor, process_memory_mb


def make_capture(**overrides) -> AudioCapture:
    cfg = AudioConfig(**{"retry_delay_s": 0.0, "watchdog_s": 10.0,
                         "digital_silence_s": 30.0, **overrides})
    return AudioCapture(queue.Queue(), cfg)


def test_watchdog_raises_when_no_audio_data_arrives():
    cap = make_capture()
    with pytest.raises(CaptureStalled):
        cap._check_watchdog(now=100.0, opened=80.0, last_data=89.0, last_sound=89.0,
                            had_sound=True)


def test_watchdog_raises_on_long_digital_silence():
    cap = make_capture()
    with pytest.raises(CaptureStalled):
        cap._check_watchdog(now=100.0, opened=60.0, last_data=99.9, last_sound=65.0,
                            had_sound=True)


def test_watchdog_quiet_while_audio_flows():
    make_capture()._check_watchdog(now=100.0, opened=0.0, last_data=99.9, last_sound=99.0,
                                   had_sound=True)


def test_capture_reopens_after_device_errors_and_stalls(monkeypatch):
    cap = make_capture()
    calls = []

    def fake_session(stop):
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise OSError("device unplugged")
        if len(calls) == 2:
            raise CaptureStalled("no audio")
        cap._stop.set()

    monkeypatch.setattr(cap, "_session", fake_session)
    cap._run()
    assert len(calls) == 3 and cap.restarts == 2


def test_device_watcher_needs_a_real_change_to_switch():
    from capture import DeviceWatcher

    watcher = DeviceWatcher(5.0)
    watcher.current = "Speakers (Realtek)"
    # names that never match the loopback name must not cause a reopen loop
    assert watcher.changed_from("Some Other Name [Loopback]", "Speakers (Realtek)") is None
    watcher.current = "Headphones (USB)"
    assert watcher.changed_from("Speakers (Realtek) [Loopback]",
                                "Speakers (Realtek)") == "Headphones (USB)"


def test_stuck_capture_is_revived(monkeypatch):
    import threading

    cap = make_capture()
    spawned = []
    monkeypatch.setattr(cap, "_spawn", lambda: spawned.append(True))
    blocker = threading.Event()
    cap._thread = threading.Thread(target=blocker.wait, daemon=True)
    cap._thread.start()
    old_stop = cap._stop
    cap.heartbeat = time.monotonic() - 60
    assert cap.stuck(30.0)
    sup = Supervisor({"capture": cap}, lambda name: True, lambda: "")
    sup.check()
    blocker.set()
    assert spawned and old_stop.is_set() and sup.restarts["capture"] == 1


def test_log_file_falls_back_when_folder_not_writable(tmp_path, monkeypatch):
    import logging

    from logsetup import setup_logging

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        used = setup_logging(log_file=blocker / "livesub.log", stream=None)
        assert used == tmp_path / "local" / "LiveSub" / "livesub.log"
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved:
            root.addHandler(h)


class FakeWorker:
    def __init__(self, alive=True):
        self.alive = alive
        self.starts = 0

    @property
    def running(self):
        return self.alive

    def start(self):
        self.starts += 1
        self.alive = True


def test_supervisor_restarts_dead_worker_only():
    dead, alive = FakeWorker(alive=False), FakeWorker()
    sup = Supervisor({"segmenter": dead, "capture": alive}, lambda name: True, lambda: "ok")
    sup.check()
    assert dead.starts == 1 and alive.starts == 0 and sup.restarts["segmenter"] == 1


def test_supervisor_respects_can_restart():
    dead = FakeWorker(alive=False)
    Supervisor({"transcriber": dead}, lambda name: False, lambda: "").check()
    assert dead.starts == 0


def test_process_memory_is_reported():
    assert process_memory_mb() > 10


def test_app_status_messages_for_first_run_and_failure():
    from main import App

    args = argparse.Namespace(mode=None, model=None, language=None)
    app = App(AppConfig(), args, has_console=True)
    app.model_present = False
    assert "downloading speech model" in app.status()
    assert "first run" in app.status()
    app.model_present = True
    assert app.status() == "LiveSub: loading speech model..."
    app.transcriber.ready.set()
    assert app.status() is None
    app.transcriber.error = "boom"
    assert "could not be loaded" in app.status()
    assert "memory" in app.health_report()
