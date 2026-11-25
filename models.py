"""Whisper model files: where they come from, whether they are downloaded, sizes."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# download sizes (GB), so the first-run message needs no network call
APPROX_SIZE_GB = {"tiny": 0.08, "base": 0.15, "small": 0.49, "medium": 1.53,
                  "large-v3-turbo": 1.62, "large-v3": 3.09}


def repo_for(model: str) -> str:
    """Hugging Face repository faster-whisper downloads a model name from."""
    from faster_whisper.utils import _MODELS

    return _MODELS.get(model, model)


def is_downloaded(model: str, model_dir: Path) -> bool:
    """True if the model is already available offline (same lookup WhisperModel uses)."""
    if Path(model).is_dir():
        return True
    from faster_whisper.utils import download_model

    try:
        download_model(model, local_files_only=True, cache_dir=str(model_dir))
        return True
    except Exception:
        return False


def approx_size_text(model: str) -> str:
    """'about 0.49 GB' for known models, otherwise 'unknown size'."""
    size = APPROX_SIZE_GB.get(model)
    return f"about {size:.2f} GB" if size else "unknown size"


def download_size_gb(model: str) -> float | None:
    """Exact download size from the Hugging Face API, or None if offline."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_for(model), files_metadata=True)
        return sum((s.size or 0) for s in info.siblings) / 1e9
    except Exception as exc:
        log.warning("Could not get size of '%s': %s", model, exc)
        return None
