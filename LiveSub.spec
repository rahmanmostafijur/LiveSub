# PyInstaller spec for LiveSub (one-folder build). Run via: python build.py
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("faster_whisper")          # bundled Silero VAD model
datas += [("profanity_words.txt", ".")]
binaries = collect_dynamic_libs("ctranslate2")        # Whisper inference DLLs
hiddenimports = ["pystray._win32"]                    # tray backend is imported dynamically
excludes = ["pytest", "matplotlib", "IPython", "pandas", "scipy"]

a = Analysis(["main.py"], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)

common = dict(exclude_binaries=True, upx=False, icon="build_assets/livesub.ico")
app = EXE(pyz, a.scripts, [], name="LiveSub", console=False, **common)
debug = EXE(pyz, a.scripts, [], name="LiveSub-console", console=True, **common)

COLLECT(app, debug, a.binaries, a.datas, name="LiveSub", upx=False)
