"""ffmpeg helpers: find it, decode any media file to 16 kHz mono audio, burn subtitles in."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from config import app_dir
from offline_progress import Progress

SAMPLE_RATE = 16000
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts",
                    ".mpg", ".mpeg", ".3gp", ".ogv", ".mts", ".m2ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

FFMPEG_HELP = """\
ffmpeg was not found. It is needed to read video files and to burn subtitles in.
Install it once, then run this command again:
  - Windows 10/11:  winget install Gyan.FFmpeg      (then open a new terminal)
  - or download a "release essentials" build from https://www.gyan.dev/ffmpeg/builds/ and
    either add its bin folder to PATH or copy ffmpeg.exe to {local}"""

BURN_STYLE = "FontSize=18,Outline=1.2,Shadow=0.5,MarginV=18"


class MediaError(RuntimeError):
    """A media file could not be read or converted (message is meant for the user)."""


def local_ffmpeg_dir() -> Path:
    return app_dir() / "ffmpeg"


def find_ffmpeg() -> str | None:
    """ffmpeg on PATH, else ffmpeg.exe in the app's ffmpeg folder, else None."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for folder in (local_ffmpeg_dir(), local_ffmpeg_dir() / "bin"):
        candidate = folder / "ffmpeg.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def ffmpeg_help() -> str:
    return FFMPEG_HELP.format(local=local_ffmpeg_dir())


def _stderr_tail(text: str, lines: int = 4) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def extract_audio(ffmpeg: str, source: Path, audio_track: int = 0) -> np.ndarray:
    """Decode a media file's audio track to float32 mono 16 kHz samples."""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(source),
           "-map", f"0:a:{audio_track}", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = _stderr_tail(proc.stderr.decode("utf-8", "replace"))
        if any(text in detail for text in ("matches no streams", "does not contain any stream",
                                           "for option 'map'")):
            raise MediaError(f"{source.name} has no audio track {audio_track}")
        raise MediaError(f"ffmpeg could not read {source.name}: {detail}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise MediaError(f"{source.name} contains no audio")
    return pcm.astype(np.float32) / 32768.0


def has_video(ffmpeg: str, source: Path) -> bool:
    """True if the file has a real video stream (cover art does not count)."""
    proc = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(source)],
                          capture_output=True)   # exits 1 after listing streams: expected
    info = proc.stderr.decode("utf-8", "replace")
    return any(re.search(r"Stream #\S+.*: Video:", line) and "attached pic" not in line
               for line in info.splitlines())


def burn_subtitles(ffmpeg: str, video: Path, srt: Path, output: Path, duration_ms: int) -> None:
    """Re-encode `video` with `srt` drawn into the picture (audio re-encoded to AAC).

    The SRT is copied to a temporary folder under a plain name and ffmpeg runs there,
    which avoids the filter-path escaping problems of Windows paths and special characters.
    """
    with tempfile.TemporaryDirectory(prefix="livesub_burn_") as tmp:
        shutil.copyfile(srt, Path(tmp) / "subs.srt")
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-i", str(video), "-vf", f"subtitles=subs.srt:force_style='{BURN_STYLE}'",
               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-progress", "pipe:1", "-nostats", str(output)]
        errors = Path(tmp) / "ffmpeg.err"
        progress = Progress("Burning subtitles", duration_ms)
        with errors.open("wb") as err_file:
            proc = subprocess.Popen(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=err_file,
                                    text=True, encoding="utf-8", errors="replace")
            try:
                for line in proc.stdout:
                    if line.startswith("out_time_us="):
                        value = line.split("=", 1)[1].strip()
                        if value.isdigit():
                            progress.update(int(value) // 1000)
                proc.wait()
            except BaseException:
                proc.kill()
                raise
        if proc.returncode != 0:
            output.unlink(missing_ok=True)
            raise MediaError("ffmpeg could not burn the subtitles in: "
                             + _stderr_tail(errors.read_text(encoding="utf-8", errors="replace")))
        progress.finish()
