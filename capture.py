"""System-audio capture from the default speaker via WASAPI loopback.

Runs in its own thread and pushes 16 kHz mono float32 chunks into a queue.
Recovers from device errors, follows changes of the default speaker, and a
watchdog reopens the stream if it silently stops delivering audio (this can
happen when players reset the audio device while switching videos).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np
import pyaudiowpatch as pyaudio
import sounddevice as sd

from audio_dsp import StreamResampler, to_mono, write_wav
from config import AudioConfig
from queues import put_drop_oldest

log = logging.getLogger(__name__)

RAW_QUEUE_BLOCKS = 100       # raw device blocks buffered between callback and thread
RAW_GET_TIMEOUT_S = 0.2      # how long the worker waits for a raw block
DROP_LOG_INTERVAL_S = 5.0    # rate limit for "queue full" warnings


class LoopbackNotFoundError(RuntimeError):
    """Raised when no WASAPI loopback device exists for the default speaker."""


class CaptureStalled(RuntimeError):
    """The stream stopped delivering usable audio; the session is reopened."""


def find_default_loopback(pa: pyaudio.PyAudio) -> dict[str, Any]:
    """Return the loopback device info that mirrors the default WASAPI speaker."""
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise LoopbackNotFoundError("WASAPI is not available on this system") from exc

    out_index = wasapi["defaultOutputDevice"]
    if out_index < 0:
        raise LoopbackNotFoundError("No default output device is set in Windows")
    speaker = pa.get_device_info_by_index(out_index)
    if speaker.get("isLoopbackDevice"):
        return speaker

    for loopback in pa.get_loopback_device_info_generator():
        if speaker["name"] in loopback["name"]:
            return loopback
    raise LoopbackNotFoundError(f"No loopback device found for '{speaker['name']}'")


def query_default_speaker_name() -> str | None:
    """Ask Windows for the current default speaker using a fresh device scan.

    Uses sounddevice's separate PortAudio instance so the open capture stream
    is not disturbed. Returns None if the query fails.
    """
    try:
        try:
            sd._terminate()   # re-scan devices; PortAudio caches them at init
        finally:
            sd._initialize()  # always re-init, or every later query would fail
        for api in sd.query_hostapis():
            if "WASAPI" in api["name"]:
                index = api["default_output_device"]
                return sd.query_devices(index)["name"] if index >= 0 else None
    except Exception as exc:  # device scans can fail transiently during changes
        log.debug("Default speaker query failed: %s", exc)
    return None


class DeviceWatcher:
    """Background thread that tracks the default speaker name.

    Kept out of the capture thread so a slow device rescan can never stall audio.
    """

    def __init__(self, interval_s: float) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current: str | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DeviceWatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(self._interval + 2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            name = query_default_speaker_name()
            if name:
                self.current = name

    def changed_from(self, device_name: str, baseline: str | None) -> str | None:
        """New default speaker if it changed since the session opened (`baseline`)
        and differs from the captured device, else None."""
        current = self.current
        if current and current != baseline and current not in device_name:
            return current
        return None


class AudioCapture:
    """Background loopback capture that feeds a thread-safe queue."""

    def __init__(self, out_queue: queue.Queue[np.ndarray], cfg: AudioConfig) -> None:
        self._out = out_queue
        self._cfg = cfg
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watcher = DeviceWatcher(cfg.device_check_interval_s)
        self._last_drop_log = 0.0
        self.device_name: str | None = None
        self.last_audio = 0.0            # monotonic time of the last non-silent chunk
        self.heartbeat = 0.0             # updated by the capture loop; stale = stuck
        self.restarts = 0                # sessions reopened (errors, watchdog, changes)

    def start(self) -> None:
        """Start the capture and device-watcher threads."""
        if self.running:
            return
        self._spawn()

    def _spawn(self) -> None:
        # each generation gets its own stop event and watcher, so a thread stuck in
        # a driver call can be abandoned and replaced (see revive)
        self._stop = threading.Event()
        self._watcher = DeviceWatcher(self._cfg.device_check_interval_s)
        self._watcher.start()
        self.heartbeat = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(self._stop,),
                                        name="AudioCapture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the threads to stop and wait for them."""
        self._stop.set()
        self._watcher.stop()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("Capture thread did not stop within %.1fs", timeout)

    @property
    def running(self) -> bool:
        """True while the capture thread is alive."""
        return bool(self._thread and self._thread.is_alive())

    def stuck(self, limit_s: float) -> bool:
        """True if the thread is alive but has not made progress for limit_s."""
        return self.running and time.monotonic() - self.heartbeat > limit_s

    def revive(self) -> None:
        """Abandon a thread blocked inside the audio driver and start a fresh one."""
        log.error("Audio capture stopped responding; starting a new capture thread")
        self._stop.set()
        self._watcher.stop()
        self.restarts += 1
        self._spawn()

    def _run(self, stop: threading.Event | None = None) -> None:
        """Keep a capture session alive until stopped, reopening on errors."""
        stop = stop or self._stop
        while not stop.is_set():
            self.heartbeat = time.monotonic()
            delay = self._cfg.retry_delay_s
            try:
                self._session(stop)
                delay = 0.0
            except CaptureStalled as exc:
                log.debug("Reopening capture: %s", exc)
                delay = 0.0
            except LoopbackNotFoundError as exc:
                log.error("%s. Retrying in %.0fs", exc, delay)
            except OSError as exc:
                log.warning("Audio device error (%s). Reopening in %.0fs", exc, delay)
            except Exception:
                log.exception("Unexpected capture error. Reopening in %.0fs", delay)
            if stop is not self._stop:
                return            # this thread was replaced by revive()
            if not stop.is_set():
                self.restarts += 1
            self.heartbeat = time.monotonic()
            stop.wait(delay)
        log.info("Audio capture stopped")

    def _session(self, stop: threading.Event) -> None:
        """Open the current default loopback device and stream until it changes."""
        pa = pyaudio.PyAudio()
        try:
            device = find_default_loopback(pa)
            channels = int(device["maxInputChannels"])
            rate = int(device["defaultSampleRate"])
            block = max(1, rate * self._cfg.block_ms // 1000)
            raw: queue.Queue[bytes] = queue.Queue(maxsize=RAW_QUEUE_BLOCKS)

            def on_audio(in_data: bytes | None, frame_count: int,
                         time_info: dict, status: int) -> tuple[None, int]:
                if in_data:
                    try:
                        raw.put_nowait(in_data)
                    except queue.Full:
                        pass  # worker is stalled; drop rather than block the audio thread
                return None, pyaudio.paContinue

            stream = pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                             input=True, input_device_index=device["index"],
                             frames_per_buffer=block, stream_callback=on_audio)
            if device["name"] != self.device_name:
                log.info("Capturing '%s' (%d Hz, %d ch)", device["name"], rate, channels)
            self.device_name = device["name"]
            try:
                if stop is self._stop:   # a replaced (revived) thread just cleans up
                    self._pump(stream, raw, channels,
                               StreamResampler(rate, self._cfg.sample_rate),
                               device["name"], stop)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()

    def _pump(self, stream: pyaudio.Stream, raw: queue.Queue[bytes], channels: int,
              resampler: StreamResampler, device_name: str, stop: threading.Event) -> None:
        """Convert raw blocks; return or raise when the session must be reopened."""
        opened = last_data = last_sound = time.monotonic()
        had_sound = False
        watcher = self._watcher
        baseline = watcher.current       # default speaker when this session opened
        while not stop.is_set():
            try:
                data = raw.get(timeout=RAW_GET_TIMEOUT_S)
            except queue.Empty:
                data = None
            now = self.heartbeat = time.monotonic()
            if data:
                last_data = now
                samples = np.frombuffer(data, dtype=np.float32)
                if np.any(samples):
                    last_sound = self.last_audio = now
                    had_sound = True
                chunk = resampler.process(to_mono(samples, channels))
                if chunk.size:
                    self._emit(chunk)

            if not stream.is_active():
                raise OSError("capture stream stopped unexpectedly")
            self._check_watchdog(now, opened, last_data, last_sound, had_sound)
            new_default = watcher.changed_from(device_name, baseline)
            if new_default:
                log.info("Default speaker changed to '%s', switching", new_default)
                return

    def _check_watchdog(self, now: float, opened: float, last_data: float,
                        last_sound: float, had_sound: bool) -> None:
        """Raise CaptureStalled if the stream stopped delivering usable audio."""
        if now - last_data > self._cfg.watchdog_s:
            if had_sound:
                log.info("No audio for %.0fs after playback; reopening capture",
                         self._cfg.watchdog_s)
            raise CaptureStalled(f"no audio data for {self._cfg.watchdog_s:.0f}s")
        if now - max(last_sound, opened) > self._cfg.digital_silence_s:
            raise CaptureStalled(f"only digital silence for {self._cfg.digital_silence_s:.0f}s")

    def _emit(self, chunk: np.ndarray) -> None:
        """Push a chunk; if the consumer lags, drop the oldest audio."""
        if not put_drop_oldest(self._out, chunk):
            return
        now = time.monotonic()
        if now - self._last_drop_log > DROP_LOG_INTERVAL_S:
            self._last_drop_log = now
            log.warning("Audio queue full; dropping oldest audio (consumer too slow)")


def make_audio_queue(cfg: AudioConfig) -> queue.Queue[np.ndarray]:
    """Create a bounded queue sized to hold cfg.queue_seconds of audio."""
    blocks = max(1, int(cfg.queue_seconds * 1000 / cfg.block_ms))
    return queue.Queue(maxsize=blocks)


def _self_test(seconds: float, wav_path: str) -> None:
    """Capture for a few seconds, log levels, and save a 16 kHz WAV for listening."""
    from config import load_config

    cfg = load_config().audio
    audio_q = make_audio_queue(cfg)
    capture = AudioCapture(audio_q, cfg)
    capture.start()
    collected: list[np.ndarray] = []
    second_buf: list[np.ndarray] = []
    end = time.monotonic() + seconds
    next_report = time.monotonic() + 1.0
    try:
        while time.monotonic() < end:
            try:
                chunk = audio_q.get(timeout=0.2)
                collected.append(chunk)
                second_buf.append(chunk)
            except queue.Empty:
                pass
            if time.monotonic() >= next_report:
                next_report += 1.0
                joined = np.concatenate(second_buf) if second_buf else np.zeros(0)
                rms = float(np.sqrt(np.mean(joined ** 2))) if joined.size else 0.0
                bar = "#" * min(40, int(rms * 200))
                log.info("received %5.2fs audio  rms=%.4f  %s",
                         joined.size / cfg.sample_rate, rms, bar)
                second_buf.clear()
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        capture.stop()

    audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    write_wav(wav_path, audio, cfg.sample_rate)
    log.info("Saved %.1fs of audio to %s", audio.size / cfg.sample_rate, wav_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test system-audio capture")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", default="capture_test.wav")
    args = parser.parse_args()
    from logsetup import setup_logging

    setup_logging()
    _self_test(args.seconds, args.out)
