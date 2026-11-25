"""Logging must never fail on Chinese/Hindi/Bengali text, whatever the console encoding."""
import io
import logging
import threading

import pytest

from logsetup import SafeStreamHandler, setup_logging

SAMPLES = ["zh: 阿叔 你好", "hi: नमस्ते दुनिया", "bn: আশা ভালোবাসা"]


class StrictLegacyStream:
    """Mimics a cp1252 console/pipe that raises on unencodable text and cannot be reconfigured."""

    encoding = "cp1252"

    def __init__(self):
        self.lines = []

    def write(self, text):
        text.encode(self.encoding)  # raises UnicodeEncodeError like a strict stream
        self.lines.append(text)

    def flush(self):
        pass


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)


@pytest.fixture
def no_logging_errors(monkeypatch):
    def fail(self, record):
        raise AssertionError(f"logging error for {record.getMessage()!r}")
    monkeypatch.setattr(logging.Handler, "handleError", fail)


def test_strict_legacy_stream_gets_escaped_lines(no_logging_errors):
    stream = StrictLegacyStream()
    setup_logging(stream=stream)
    for text in SAMPLES:
        logging.getLogger("t").info(text)
    output = "".join(stream.lines).splitlines()
    assert len(output) == 3                       # no merged lines
    assert "\\u963f\\u53d4" in output[0]          # 阿叔 escaped, not lost
    assert all("Logging error" not in line for line in output)


def test_strict_textiowrapper_is_made_safe(no_logging_errors):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    setup_logging(stream=stream)
    for text in SAMPLES:
        logging.getLogger("t").info(text)
    stream.flush()
    lines = raw.getvalue().decode("cp1252").splitlines()
    assert len(lines) == 3 and stream.errors == "backslashreplace"


def test_utf8_log_file_keeps_exact_text(tmp_path, no_logging_errors):
    log_file = tmp_path / "livesub.log"
    setup_logging(log_file=log_file, stream=StrictLegacyStream())
    for text in SAMPLES:
        logging.getLogger("t").info(text)
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_file.read_text(encoding="utf-8")
    assert all(text in content for text in SAMPLES)


def test_concurrent_threads_never_merge_lines(no_logging_errors):
    stream = StrictLegacyStream()
    handler = SafeStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("threads")
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    def worker(n):
        for i in range(50):
            logger.info("%d-%d %s", n, i, SAMPLES[i % 3])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.removeHandler(handler)
    lines = "".join(stream.lines).splitlines()
    assert len(lines) == 200
    assert all(line.split(" ", 1)[0].count("-") == 1 for line in lines)
