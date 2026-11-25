"""Build the Windows app folder dist\\LiveSub\\ with PyInstaller.

    python build.py               # build; copy the downloaded model so it works offline
    python build.py --no-models   # smaller folder; the model downloads on first start
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist" / "LiveSub"
ICON = ROOT / "build_assets" / "livesub.ico"
EXTRA_FILES = ("profanity_words.txt", "README.md")
ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64)]


def make_icon() -> None:
    """Write the tray icon drawing as a multi-size .ico for the exe."""
    from tray import make_icon_image

    ICON.parent.mkdir(exist_ok=True)
    make_icon_image().save(ICON, sizes=ICON_SIZES)


def run_pyinstaller() -> None:
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    str(ROOT / "LiveSub.spec")], cwd=ROOT, check=True)


def copy_extras() -> None:
    """User-editable files go next to the exe (not inside _internal)."""
    for name in EXTRA_FILES:
        shutil.copy2(ROOT / name, DIST / name)
    config = DIST / "config.json"
    if not config.exists():
        config.write_text('{\n  "transcribe": {"language": "auto", "mode": "original"}\n}\n',
                          encoding="utf-8")


def copy_model() -> None:
    """Copy the configured Whisper model (if downloaded) into dist\\LiveSub\\models."""
    from config import load_config
    from models import repo_for

    model = load_config().transcribe.model
    folder = "models--" + repo_for(model).replace("/", "--")
    source = ROOT / "models" / folder
    if not source.exists():
        print(f"Model '{model}' not downloaded yet; the app will download it on first start")
        return
    target = DIST / "models" / folder
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    print(f"Copied model '{model}' to {target}")


def folder_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dist\\LiveSub with PyInstaller")
    parser.add_argument("--no-models", action="store_true",
                        help="do not copy the model; it downloads on first start")
    args = parser.parse_args()
    make_icon()
    run_pyinstaller()
    copy_extras()
    if not args.no_models:
        copy_model()
    print(f"\nBuilt {DIST} ({folder_size_mb(DIST):.0f} MB). Start LiveSub.exe; "
          "use LiveSub-console.exe to see the log in a console window.")


if __name__ == "__main__":
    main()
