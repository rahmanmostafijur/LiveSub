# LiveSub

Live subtitles for anything playing on a Windows PC (PotPlayer, YouTube, Netflix, ...).
LiveSub records the system audio, detects speech, transcribes it with Whisper on the CPU,
and shows the text in a transparent, click-through overlay at the bottom of the screen.
It can show the spoken language or translate to English. Everything runs offline after
the first model download; no GPU or API key is needed.

```
capture.py  ->  segmenter.py  ->  transcribe.py  ->  overlay.py
(WASAPI loopback)  (Silero VAD)    (faster-whisper)    (tkinter window)
```

## Using the packaged app

1. Copy the `LiveSub` folder anywhere (for example `C:\Apps\LiveSub`).
2. Start `LiveSub.exe`. A small subtitle icon appears in the tray (click the `^` arrow
   in the taskbar if it is hidden).
3. Play a video. Subtitles appear at the bottom of the screen about 1-2 s after speech.

`LiveSub-console.exe` is the same app with a console window that shows the log, useful
for troubleshooting. Only one copy of LiveSub can run at a time.

**First start:** if the folder has no `models` subfolder, LiveSub downloads the speech
model (`small`, about 0.49 GB) once. The overlay shows "downloading speech model..."
during this; an internet connection is needed only this one time.

## Controls

| Hotkey | Action |
|---|---|
| Ctrl+Alt+S | show / hide subtitles |
| Ctrl+Alt+M | original language / English translation |
| Ctrl+Alt+L | next language: Auto-detect, English, Korean, Hindi, Malayalam, Tamil, Bengali |
| Ctrl+Alt+Q | quit (Ctrl+C in the console also quits) |

Right-click the tray icon for the same settings: Show subtitles, Mode, Language, Open log
file, Quit. Changes of mode or language are shown briefly above the subtitles. With
auto-detect, a change of the detected language is shown as "Detected: ...".

## Running from source

```
cd D:\LiveSub
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Options (they override `config.json`):

| Option | Example | Meaning |
|---|---|---|
| `--mode` | `--mode english` | start in English translation mode |
| `--language` | `--language ko` | force a language; `auto` to detect |
| `--model` | `--model base` | faster but less accurate model |
| `--debug` | | more detailed log |

## Configuration (`config.json`, next to the exe or main.py)

List only what you want to change, for example:

```json
{
  "transcribe": { "model": "small", "mode": "original", "language": "auto" },
  "overlay": { "font_size": 34, "bottom_margin_px": 120, "languages": "auto,en,ko,hi" },
  "profanity": { "enabled": true, "extra_words": "darn, heck" }
}
```

- `transcribe.language`: `auto` (default) or a code such as `en`, `ko`, `hi`, `ml`, `ta`, `bn`.
  Auto-detect ignores very short or unsure segments and follows a new language after two
  clear segments, so it settles quickly when you switch videos.
- `transcribe.model`: `small` (default) or `base` (faster, less accurate).
- `overlay.merge_window_s`: a piece that does not end a sentence (or the next piece starts
  in lower case) is added to the current subtitle if it arrives within this time.
- `overlay.clear_after_s`: subtitle disappears after this much silence.
- `overlay.font_family`, `font_size`, `max_lines`, `width_ratio`, `bottom_margin_px`,
  `hotkey_*`, `languages`, `tray_icon`.
- `profanity.enabled`, `words_file`, `extra_words`: masks words such as `f**k` in the
  overlay and the log. The list is `profanity_words.txt` (editable).
- `audio.watchdog_s`: the capture is reopened if no audio arrives for this long, which
  recovers from players that reset the audio device when switching videos.

All settings and defaults are documented in `config.py`.

## Offline subtitles for a video or audio file (`offline.py`)

Besides the live overlay, LiveSub can turn a video or audio file into an accurate `.srt`
subtitle file. It runs on the CPU, needs no internet after the one-time model download, and
does not touch the live app. Speed is not a goal here: it uses Whisper's normal 30 second
window with beam search, so it is slow but careful (see "How long it takes" below).

```
python offline.py movie.mkv                                   # large-v3, language detected
python offline.py movie.mkv --model medium                    # faster, a little less accurate
python offline.py movie.mkv --language hi --also-translate    # Hindi .srt + English .en.srt
python offline.py movie.mkv --task translate                  # English subtitles only
python offline.py movie.mkv --burn                            # also a copy with subtitles burned in
python offline.py --batch D:\Movies --resume                  # every video in a folder
```

The `.srt` is written next to the input (`movie.srt`, English as `movie.en.srt`) as UTF-8
without BOM, so Bengali, Korean, Hindi, Tamil and Malayalam text opens correctly in players.

| Option | Meaning |
|---|---|
| `--model small\|medium\|large-v3` | default `large-v3` (int8, CPU). A missing model is downloaded into `models\` after showing its size and asking y/N (`--yes` skips the question) |
| `--language auto\|en\|ko\|hi\|ml\|ta\|bn\|...` | default `auto`: 6 speech samples spread over the whole file vote for the dominant language |
| `--task transcribe\|translate` | `translate` = English subtitles |
| `--also-translate` | write the original-language file and the English `.en.srt` (a second full pass; skipped for English audio) |
| `--output out.srt` | output name (English goes to `out.en.srt`) |
| `--resume` / `--restart` | continue an interrupted run / discard it and start over |
| `--mask` | mask profanity (`f**k`) with the live app's word list; off by default |
| `--burn` | also write `movie.subbed.mp4` with the subtitles drawn into the picture (ffmpeg re-encodes the video, which takes a while) |
| `--batch FOLDER` | all video/audio files in the folder; files that already have an `.srt` are skipped (`--overwrite` redoes them) |
| `--audio-track N` | use another audio track (default 0 = first) |
| `--lang-samples N`, `--threads N` | language samples (default 6), CPU threads (default automatic) |

**ffmpeg** is needed to read video files and for `--burn`. It is found on `PATH` or in
`D:\LiveSub\ffmpeg\`. If it is missing the program says so; install it with
`winget install Gyan.FFmpeg` and open a new terminal.

**Interrupted runs.** Every finished piece is saved as it is produced to a folder
`movie.partial\` next to the output. After Ctrl+C, a crash or a power cut, run the same
command with `--resume`. Without `--resume` the program refuses to overwrite the partial
result. The folder is deleted when the subtitles are complete.

**What is done for accuracy and sync.** Standard `transcribe()` with 30 s windows, beam size
5, previous text as context, VAD filtering tuned to keep soft word starts, and word
timestamps. Subtitles are built from the word times: at most 2 lines of about 42 columns,
1-7 s per cue, split at sentence ends and pauses, an 80 ms minimum gap, never overlapping.
A first/last word that Whisper stretched into a pause is moved back next to the speech.
Whisper's known filler inventions ("Thanks for watching", "Amara.org") and repetition loops
are removed with the same `textproc.py` rules as the live app. Speech that the model skipped
without any warning is found with the VAD and transcribed again on its own.

**Mixed-language files.** If the six samples disagree, the file is checked block by block
(about 3 minutes each) and every section is transcribed in its own language. The switch is
found to within one block, so a language change can be placed up to about 3 minutes off.

**Limits.** Subtitles are normally within about half a second of the speech (a worst case of
about 2 s was seen on fast read speech); Whisper timestamps are not frame-exact. Hindi was
about 3 times slower than English with `large-v3`; Korean was not.

### How long it takes

Measured on this PC (Ryzen 5 2400G, 4 cores, int8). "Real-time factor" is processing time
divided by audio length. Speed varies by up to 50% between runs on this machine.

| Model | Audio tested | Real-time factor | A 2 hour movie takes about |
|---|---|---|---|
| `medium` | English | 1.4 - 2.0 | 3 - 4 hours |
| `medium` | Korean | 1.4 | 3 hours |
| `large-v3` | English | 2.4 - 2.7 | 5 - 5.5 hours |
| `large-v3` | Korean | 1.9 | 4 hours |
| `large-v3` | Hindi | 7.0 | 14 hours (Hindi was far slower than Korean in the same model) |

The test clips were mostly speech (about 80%). A film with music and silence usually takes
25-40% less. Add 2-4 minutes for language detection, and double the time for
`--also-translate` (a second full pass). Accuracy in the tests: on English read speech the
word error rate was 0.075 for `medium` and 0.131 for `large-v3` (`large-v3` invented a few
extra words in two places); on Korean the character error rate was 0.105 for `medium` and
0.031 for `large-v3`. So `large-v3` is clearly better for non-English speech, and `medium`
is a reasonable choice for English. Close other heavy programs while it runs, and use
`--resume` if you have to stop it.

## Logs and health

Everything is logged to `livesub.log` (UTF-8, next to the exe/main.py, rotated at 1 MB).
Once a minute a `Health:` line shows memory use, time since the last audio and subtitle,
and how often the capture was reopened. If a worker thread ever stops, it is restarted
automatically and logged as an error.

## Building the exe

```
.\venv\Scripts\activate
pip install -r requirements.txt
python build.py              # dist\LiveSub\ including the downloaded model (~790 MB)
python build.py --no-models  # dist\LiveSub\ without model (~300 MB, downloads on first start)
```

The build is a one-folder app: `dist\LiveSub\LiveSub.exe` plus an `_internal` folder.
`config.json`, `profanity_words.txt`, `models\` and `livesub.log` live next to the exe so
they can be edited and kept when the app is updated.

**One-folder vs one-file.** One-folder (used here) starts in about 3-5 s because nothing
has to be unpacked. A one-file exe (`--onefile`) is a single ~300 MB file, but every start
first unpacks it to a temporary folder (10-20 s on a normal disk, and the temp copy is
deleted again at exit). One-file builds are also flagged by antivirus programs more often.
The model (0.5-3 GB) should stay outside the exe either way, so a one-file build would
still need the `models` folder next to it.

## Testing the parts separately

```
python capture.py --seconds 10          # records capture_test.wav from system audio
python segmenter.py --seconds 30        # writes speech segments to segments_test\
python transcribe.py --seconds 60       # prints subtitles in the console (Enter = switch mode)
python benchmark.py                     # compares Whisper models on this CPU
python -m pytest -q                     # unit tests
```

## Troubleshooting

- **No subtitles:** audio must play through the default Windows output device. The log
  shows which device is captured; a new default device is picked up within seconds.
- **Subtitles lag behind:** use `--model base`, or close other heavy programs.
- **A hotkey does nothing:** another program uses it (the log says so); change
  `overlay.hotkey_*` in `config.json`.
- **Wrong language:** pick it with Ctrl+Alt+L or the tray menu, or set
  `transcribe.language`.
- **"LiveSub is already running":** quit the other copy from its tray icon or with Ctrl+Alt+Q.
- **Bengali in original mode** is written in the wrong script by the `small` model;
  use English mode for Bengali videos.
