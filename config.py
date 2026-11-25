"""Application settings, loaded from config.json with safe defaults."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODES = ("original", "english")
LANGUAGE_NAMES = {"": "Auto-detect", "en": "English", "ko": "Korean", "hi": "Hindi",
                  "ml": "Malayalam", "ta": "Tamil", "bn": "Bengali", "ja": "Japanese",
                  "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German",
                  "ar": "Arabic", "ur": "Urdu", "te": "Telugu", "mr": "Marathi"}


def language_name(code: str) -> str:
    """Readable name for a language code ('' = auto-detect)."""
    return LANGUAGE_NAMES.get(code, code.upper())


def app_dir() -> Path:
    """Folder holding config.json and models (the .exe folder when packaged)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def resource_path(name: str) -> Path:
    """A bundled file: the copy next to the app if present (user-editable),
    otherwise the one packed inside the PyInstaller build."""
    local = app_dir() / name
    if local.exists():
        return local
    return Path(getattr(sys, "_MEIPASS", app_dir())) / name


DEFAULT_CONFIG_PATH = app_dir() / "config.json"


@dataclass(frozen=True)
class AudioConfig:
    """Capture settings."""

    sample_rate: int = 16000          # Whisper expects 16 kHz mono
    block_ms: int = 100               # size of each chunk pushed to the queue
    queue_seconds: float = 30.0       # audio buffered before the oldest is dropped
    device_check_interval_s: float = 5.0  # how often to look for a new default speaker
    retry_delay_s: float = 3.0        # wait before reopening after a device error
    watchdog_s: float = 10.0          # reopen capture if no audio data arrives this long
    digital_silence_s: float = 30.0   # reopen capture if only exact zeros arrive this long


@dataclass(frozen=True)
class VadConfig:
    """Speech segmentation settings (times in seconds)."""

    threshold: float = 0.5        # speech probability that starts a segment
    neg_threshold: float = 0.35   # probability below which a frame counts as silence
    min_speech_s: float = 0.25    # segments with less new speech are discarded
    min_segment_s: float = 2.0    # after this length, a short pause ends the segment
    max_segment_s: float = 4.0    # hard limit; longer speech is cut here
    split_silence_s: float = 0.2  # pause that ends a segment once min_segment_s is reached
    end_silence_s: float = 0.5    # pause that always ends a segment
    pad_s: float = 0.15           # trailing silence kept so the last word is not clipped
    pre_roll_s: float = 0.3       # audio kept before speech onset
    overlap_s: float = 0.4        # repeated at the next segment's start after a hard cut
    cut_search_s: float = 1.0     # window before max_segment_s searched for a quiet cut point
    gap_flush_s: float = 0.6      # no audio for this long (playback paused) closes a segment

    def __post_init__(self) -> None:
        if not 0.0 < self.neg_threshold <= self.threshold < 1.0:
            raise ValueError("vad: need 0 < neg_threshold <= threshold < 1")
        if not 0.0 < self.min_segment_s < self.max_segment_s:
            raise ValueError("vad: need 0 < min_segment_s < max_segment_s")
        if self.overlap_s + self.cut_search_s >= self.max_segment_s:
            raise ValueError("vad: overlap_s + cut_search_s must be below max_segment_s")
        if self.split_silence_s > self.end_silence_s:
            raise ValueError("vad: split_silence_s must not exceed end_silence_s")
        for name in ("min_speech_s", "pad_s", "pre_roll_s", "overlap_s",
                     "cut_search_s", "gap_flush_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"vad: {name} must not be negative")


@dataclass(frozen=True)
class TranscribeConfig:
    """Whisper settings."""

    model: str = "small"            # tiny | base | small | medium, or a model folder path
    model_dir: str = "models"       # download/cache folder, relative to the app folder
    compute_type: str = "int8"
    cpu_threads: int = 4            # 0 = let CTranslate2 decide
    mode: str = "original"          # original | english (Whisper translate)
    language: str = ""              # "" or "auto" = detect, or a code such as "bn" to force it
    language_min_duration_s: float = 1.5   # shorter segments never change the language
    language_min_confidence: float = 0.5   # less certain detections are ignored
    beam_size: int = 1              # 1 = greedy (fastest); 5 = slightly better, much slower
    min_window_s: float = 6.0       # encoder window; shorter = faster (Whisper default is 30)
    window_margin_s: float = 2.0    # silence kept after the audio inside the window
    use_context: bool = True        # give the previous subtitle to Whisper as context
    context_max_gap_s: float = 3.0  # only use context if speech continued within this gap
    language_smoothing: float = 0.5 # weight of the newest segment in language detection
    max_lag_s: float = 8.0          # segments waiting longer than this are skipped
    merge_max_s: float = 10.0       # when behind, queued segments are merged up to this length
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0

    def __post_init__(self) -> None:
        language = self.language.strip().lower()
        object.__setattr__(self, "language", "" if language == "auto" else language)
        if self.language and not self.language.isalpha():
            raise ValueError(f"transcribe: invalid language code {self.language!r}")
        if not 0.0 <= self.language_min_confidence < 1.0 or self.language_min_duration_s < 0:
            raise ValueError("transcribe: language_min_confidence must be in [0, 1) "
                             "and language_min_duration_s >= 0")
        if self.mode not in MODES:
            raise ValueError(f"transcribe: mode must be one of {MODES}")
        if self.beam_size < 1 or self.cpu_threads < 0:
            raise ValueError("transcribe: beam_size must be >= 1 and cpu_threads >= 0")
        if not 1.0 <= self.min_window_s <= 30.0:
            raise ValueError("transcribe: min_window_s must be between 1 and 30")
        if not 0.0 < self.merge_max_s <= 30.0 - self.window_margin_s:
            raise ValueError("transcribe: merge_max_s + window_margin_s must not exceed 30")
        if not 0.0 < self.language_smoothing <= 1.0:
            raise ValueError("transcribe: language_smoothing must be in (0, 1]")

    @property
    def model_path(self) -> Path:
        """Absolute model download folder."""
        path = Path(self.model_dir)
        return path if path.is_absolute() else app_dir() / path


@dataclass(frozen=True)
class OverlayConfig:
    """Subtitle window, merging and hotkeys."""

    font_family: str = "Segoe UI"
    font_size: int = 30
    bold: bool = True
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_px: int = 2
    max_lines: int = 2
    width_ratio: float = 0.8        # max text width as a share of the screen width
    bottom_margin_px: int = 90      # distance from the bottom of the screen
    merge_window_s: float = 4.0     # an unfinished sentence is continued within this time
    clear_after_s: float = 4.0      # subtitle disappears after this much silence
    status_s: float = 2.0           # how long notices like "English mode" stay visible
    hotkey_toggle: str = "ctrl+alt+s"
    hotkey_mode: str = "ctrl+alt+m"
    hotkey_language: str = "ctrl+alt+l"   # cycles through `languages`
    hotkey_quit: str = "ctrl+alt+q"
    languages: str = "auto,en,ko,hi,ml,ta,bn"   # offered by the hotkey and the tray menu
    tray_icon: bool = True

    @property
    def language_choices(self) -> list[str]:
        """Language codes to cycle through; '' stands for auto-detect."""
        codes = [c.strip().lower() for c in self.languages.split(",") if c.strip()]
        return ["" if c == "auto" else c for c in codes] or [""]

    def __post_init__(self) -> None:
        if not 1 <= self.max_lines <= 5 or not 8 <= self.font_size <= 120:
            raise ValueError("overlay: max_lines must be 1-5 and font_size 8-120")
        if not 0.2 <= self.width_ratio <= 1.0:
            raise ValueError("overlay: width_ratio must be between 0.2 and 1.0")
        if self.merge_window_s < 0 or self.clear_after_s <= 0 or self.outline_px < 0:
            raise ValueError("overlay: merge_window_s/outline_px must be >= 0, "
                             "clear_after_s > 0")


@dataclass(frozen=True)
class ProfanityConfig:
    """Masking of profanity and slurs in subtitles and logs."""

    enabled: bool = True
    words_file: str = ""            # "" = built-in profanity_words.txt
    extra_words: str = ""           # comma-separated additions, e.g. "darn, heck"


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    profanity: ProfanityConfig = field(default_factory=ProfanityConfig)


def _build_section(cls: type, values: dict[str, Any], section: str) -> Any:
    """Create a section dataclass from a dict, ignoring unknown keys with a warning."""
    known = {f.name for f in fields(cls)}
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            log.warning("Unknown config key '%s.%s' ignored", section, key)
            continue
        default = getattr(defaults, key)
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Config '{section}.{key}' must be a number, got {value!r}")
            value = type(default)(value)
        elif not isinstance(value, type(default)):
            raise ValueError(
                f"Config '{section}.{key}' must be {type(default).__name__}, got {value!r}"
            )
        kwargs[key] = value
    return replace(defaults, **kwargs)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load config.json if present; otherwise return defaults."""
    if not path.exists():
        log.info("No %s found, using defaults", path.name)
        return AppConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    defaults = AppConfig()
    section_names = {f.name for f in fields(AppConfig)}
    kwargs: dict[str, Any] = {}
    for name, values in raw.items():
        if name not in section_names:
            log.warning("Unknown config section '%s' ignored", name)
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{name}' must be an object")
        section_cls = type(getattr(defaults, name))
        kwargs[name] = _build_section(section_cls, values, name)
    return replace(defaults, **kwargs)
