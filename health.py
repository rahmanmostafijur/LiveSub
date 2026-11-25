"""Supervisor: restarts dead pipeline threads and logs a periodic health line."""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes
from typing import Callable, Protocol

log = logging.getLogger(__name__)

STUCK_LIMIT_S = 30.0   # a worker with no heartbeat for this long is considered hung


class Worker(Protocol):
    @property
    def running(self) -> bool: ...
    def start(self) -> None: ...


class _MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def process_memory_mb() -> float:
    """Working-set size of this process in MB (0 if unavailable)."""
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32")   # private instance: own prototypes
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD)
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                              ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError, ctypes.ArgumentError):
        return 0.0
    return counters.WorkingSetSize / 1e6 if ok else 0.0


class Supervisor:
    """Every `interval_s` restart workers that died; log health every `report_s`."""

    def __init__(self, workers: dict[str, Worker], can_restart: Callable[[str], bool],
                 report: Callable[[], str], interval_s: float = 5.0,
                 report_s: float = 60.0) -> None:
        self._workers = workers
        self._can_restart = can_restart
        self._report = report
        self._interval = interval_s
        self._report_s = report_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts: dict[str, int] = {name: 0 for name in workers}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="Supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(self._interval + 1.0)

    def check(self) -> None:
        """Restart workers that died; revive ones stuck without progress."""
        for name, worker in self._workers.items():
            stuck = getattr(worker, "stuck", None)
            if stuck is not None and stuck(STUCK_LIMIT_S) and self._can_restart(name):
                self.restarts[name] += 1
                worker.revive()
            elif not worker.running and self._can_restart(name):
                self.restarts[name] += 1
                log.error("%s thread stopped unexpectedly; restarting (restart #%d)",
                          name, self.restarts[name])
                worker.start()

    def _run(self) -> None:
        next_report = time.monotonic() + self._report_s
        while not self._stop.wait(self._interval):
            try:
                self.check()
                if time.monotonic() >= next_report:
                    next_report += self._report_s
                    log.info("Health: %s", self._report())
            except Exception:
                log.exception("Supervisor check failed")
