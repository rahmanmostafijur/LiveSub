"""Tests for config loading and validation."""
import json

import pytest

from config import AppConfig, load_config


def write(tmp_path, data) -> object:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_file_returns_defaults(tmp_path):
    assert load_config(tmp_path / "nope.json") == AppConfig()


def test_overrides_are_applied_and_coerced(tmp_path):
    cfg = load_config(write(tmp_path, {"audio": {"block_ms": 50, "queue_seconds": 10}}))
    assert cfg.audio.block_ms == 50
    assert cfg.audio.queue_seconds == 10.0
    assert isinstance(cfg.audio.queue_seconds, float)
    assert cfg.audio.sample_rate == 16000


def test_unknown_keys_are_ignored(tmp_path):
    cfg = load_config(write(tmp_path, {"audio": {"bogus": 1}, "other": {}}))
    assert cfg == AppConfig()


def test_wrong_type_raises(tmp_path):
    with pytest.raises(ValueError, match="audio.block_ms"):
        load_config(write(tmp_path, {"audio": {"block_ms": "fast"}}))


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_config(path)
