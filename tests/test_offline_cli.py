import shutil
import subprocess

import pytest

import offline
import offline_engine
from offline_audio import MediaError, extract_audio, find_ffmpeg
from offline_state import SAMPLE_RATE, RawSegment

FFMPEG = find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

SCRIPT = [(1.0, 3.0, "Hello there my friend."), (4.0, 6.0, "How are you today."),
          (7.0, 9.0, "I am fine, thank you.")]


def args_for(*argv):
    return offline.parse_args([str(a) for a in argv])


def make_tone(path, seconds=10):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", str(path)], check=True)


class FakeBackend:
    """Engine stand-in: plays SCRIPT; `crash_after` segments, then the 'user' presses Ctrl+C."""

    def __init__(self, language="en", crash_after=None, total_s=10.0):
        self.language, self.crash_after, self.total_s = language, crash_after, total_s
        self.emitted, self.tasks = 0, []

    def speech_spans(self, audio):
        return [(1000, 3000), (4000, 6000), (7000, 9000)]     # where SCRIPT has speech

    def detector(self, audio):
        return lambda spans: {self.language: 0.95, "xx": 0.01}

    def transcribe(self, audio, language, task, prompt, vad=True):
        self.tasks.append(task)
        offset = self.total_s - len(audio) / SAMPLE_RATE
        for start, end, text in SCRIPT:
            if start >= offset - 1e-6:
                if self.crash_after is not None and self.emitted >= self.crash_after:
                    raise KeyboardInterrupt
                self.emitted += 1
                parts = text.split()
                step = (end - start) / len(parts)
                words = tuple((" " + w, start - offset + i * step, start - offset + (i + 1) * step, 0.9)
                              for i, w in enumerate(parts))
                yield RawSegment(start - offset, end - offset, text, words)


def context(backend):
    ctx = offline.Context(FFMPEG or "ffmpeg", "small", 0, None)
    ctx._backend = backend
    return ctx


# --- arguments and file names ---------------------------------------------------------

def test_defaults_match_the_spec():
    args = args_for("movie.mkv")

    assert (args.model, args.language, args.task) == ("large-v3", "auto", "transcribe")
    assert not (args.mask or args.burn or args.resume or args.also_translate)


@pytest.mark.parametrize("argv", [[], ["a.mp4", "--batch", "x"], ["--batch", "x", "--output", "o.srt"],
                                  ["a.mp4", "--resume", "--restart"], ["a.mp4", "--model", "tiny"]])
def test_invalid_combinations_are_rejected(argv):
    with pytest.raises(SystemExit):
        offline.parse_args(argv)


def test_output_names(tmp_path):
    movie = tmp_path / "My Movie.mkv"

    assert offline.output_paths(movie, args_for(movie)) == {"transcribe": tmp_path / "My Movie.srt"}
    assert offline.output_paths(movie, args_for(movie, "--task", "translate")) == {
        "translate": tmp_path / "My Movie.en.srt"}
    both = offline.output_paths(movie, args_for(movie, "--also-translate"))
    assert both == {"transcribe": tmp_path / "My Movie.srt", "translate": tmp_path / "My Movie.en.srt"}
    custom = offline.output_paths(movie, args_for(movie, "--output", tmp_path / "out.srt",
                                                  "--also-translate"))
    assert custom == {"transcribe": tmp_path / "out.srt", "translate": tmp_path / "out.en.srt"}


def test_unknown_language_code_is_refused():
    offline.check_language("auto")
    offline.check_language("ml")
    with pytest.raises(offline.UsageError, match="Unknown language"):
        offline.check_language("klingon")


def test_batch_collects_only_media_files(tmp_path):
    for name in ("a.mp4", "b.MKV", "c.mp3", "notes.txt", "a.srt"):
        (tmp_path / name).write_bytes(b"x")

    files = offline.collect_inputs(args_for("--batch", tmp_path))

    assert [f.name for f in files] == ["a.mp4", "b.MKV", "c.mp3"]
    with pytest.raises(offline.UsageError):
        offline.collect_inputs(args_for(tmp_path / "missing.mp4"))


# --- model download prompt ------------------------------------------------------------

def test_missing_model_shows_size_and_downloads_nothing_without_consent(monkeypatch, capsys):
    monkeypatch.setattr(offline_engine, "is_downloaded", lambda name, folder: False)
    monkeypatch.setattr(offline_engine, "download_size_gb", lambda name: 3.09)
    monkeypatch.setattr(offline_engine, "_confirmed", lambda question: False)

    assert offline_engine.ensure_model("large-v3", assume_yes=False) is False
    out = capsys.readouterr().out
    assert "3.09 GB" in out and "models" in out and "Nothing was downloaded" in out


def test_present_model_needs_no_prompt(monkeypatch):
    monkeypatch.setattr(offline_engine, "is_downloaded", lambda name, folder: True)
    monkeypatch.setattr(offline_engine, "_confirmed",
                        lambda q: pytest.fail("must not ask when the model is on disk"))

    assert offline_engine.ensure_model("small", assume_yes=False) is True


def test_no_answer_counts_as_no(monkeypatch):
    class Closed:
        def isatty(self):
            return True
    monkeypatch.setattr(offline_engine.sys, "stdin", Closed())
    monkeypatch.setattr("builtins.input", lambda q: (_ for _ in ()).throw(EOFError))

    assert offline_engine._confirmed("Download? [y/N] ") is False


# --- ffmpeg and a whole file ----------------------------------------------------------

@needs_ffmpeg
def test_extract_audio_returns_16k_mono_float_samples(tmp_path):
    make_tone(tmp_path / "tone.wav", seconds=3)

    audio = extract_audio(FFMPEG, tmp_path / "tone.wav")

    assert audio.dtype.name == "float32" and abs(len(audio) - 3 * SAMPLE_RATE) < 400
    assert 0.1 < abs(audio).max() <= 1.0


@needs_ffmpeg
def test_unreadable_and_missing_track_give_clear_errors(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    make_tone(tmp_path / "tone.wav", seconds=1)

    with pytest.raises(MediaError, match="could not read"):
        extract_audio(FFMPEG, tmp_path / "bad.mp4")
    with pytest.raises(MediaError, match="no audio track 3"):
        extract_audio(FFMPEG, tmp_path / "tone.wav", audio_track=3)


@needs_ffmpeg
def test_full_run_writes_srt_and_cleans_up_state(tmp_path):
    make_tone(tmp_path / "talk.wav")
    args = args_for(tmp_path / "talk.wav")

    report = offline.process_file(args.input, args, context(FakeBackend()))

    srt = (tmp_path / "talk.srt").read_text(encoding="utf-8")
    assert "Hello there my friend." in srt and "I am fine, thank you." in srt
    assert "00:00:01,000 --> " in srt
    assert not (tmp_path / "talk.partial").exists()
    assert report.status == "done"


@needs_ffmpeg
def test_interrupted_run_leaves_partial_and_resume_finishes_it(tmp_path):
    make_tone(tmp_path / "talk.wav")
    first = args_for(tmp_path / "talk.wav")
    with pytest.raises(KeyboardInterrupt):
        offline.process_file(first.input, first, context(FakeBackend(crash_after=2)))
    assert (tmp_path / "talk.partial").exists() and not (tmp_path / "talk.srt").exists()

    with pytest.raises(offline.UsageError, match="--resume"):    # no silent overwrite
        offline.process_file(first.input, first, context(FakeBackend()))

    resumed = args_for(tmp_path / "talk.wav", "--resume")
    offline.process_file(resumed.input, resumed, context(FakeBackend()))
    resumed_srt = (tmp_path / "talk.srt").read_text(encoding="utf-8")
    assert not (tmp_path / "talk.partial").exists()

    straight = tmp_path / "straight"
    straight.mkdir()
    shutil.copy(tmp_path / "talk.wav", straight / "talk.wav")
    plain = args_for(straight / "talk.wav")
    offline.process_file(plain.input, plain, context(FakeBackend()))
    assert resumed_srt == (straight / "talk.srt").read_text(encoding="utf-8")


@needs_ffmpeg
def test_resume_refuses_a_changed_input_and_restart_discards(tmp_path):
    make_tone(tmp_path / "talk.wav")
    first = args_for(tmp_path / "talk.wav")
    with pytest.raises(KeyboardInterrupt):
        offline.process_file(first.input, first, context(FakeBackend(crash_after=1)))
    make_tone(tmp_path / "talk.wav", seconds=11)                 # the file changed

    resume = args_for(tmp_path / "talk.wav", "--resume")
    with pytest.raises(offline.UsageError, match="input_size"):
        offline.process_file(resume.input, resume, context(FakeBackend(total_s=11.0)))

    restart = args_for(tmp_path / "talk.wav", "--restart")
    offline.process_file(restart.input, restart, context(FakeBackend(total_s=11.0)))
    assert (tmp_path / "talk.srt").exists()


@needs_ffmpeg
def test_also_translate_writes_two_files_for_foreign_audio_and_one_for_english(tmp_path):
    make_tone(tmp_path / "hindi.wav")
    make_tone(tmp_path / "english.wav")
    foreign = args_for(tmp_path / "hindi.wav", "--also-translate")
    backend = FakeBackend(language="hi")

    offline.process_file(foreign.input, foreign, context(backend))

    assert (tmp_path / "hindi.srt").exists() and (tmp_path / "hindi.en.srt").exists()
    assert backend.tasks == ["transcribe", "translate"]

    english = args_for(tmp_path / "english.wav", "--also-translate")
    offline.process_file(english.input, english, context(FakeBackend(language="en")))
    assert (tmp_path / "english.srt").exists() and not (tmp_path / "english.en.srt").exists()


@needs_ffmpeg
def test_translate_task_writes_only_the_english_file(tmp_path):
    make_tone(tmp_path / "hindi.wav")
    args = args_for(tmp_path / "hindi.wav", "--task", "translate")
    backend = FakeBackend(language="hi")

    offline.process_file(args.input, args, context(backend))

    assert backend.tasks == ["translate"]
    assert (tmp_path / "hindi.en.srt").exists() and not (tmp_path / "hindi.srt").exists()


@needs_ffmpeg
def test_batch_skips_finished_files_and_keeps_going_after_a_failure(tmp_path):
    make_tone(tmp_path / "a.wav")
    make_tone(tmp_path / "b.wav")
    (tmp_path / "b.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nold\n", encoding="utf-8")
    args = args_for("--batch", tmp_path)

    skipped = offline.process_file(tmp_path / "b.wav", args, context(FakeBackend()))
    done = offline.process_file(tmp_path / "a.wav", args, context(FakeBackend()))

    assert skipped.status.startswith("skipped") and "old" in (tmp_path / "b.srt").read_text()
    assert done.status == "done" and (tmp_path / "a.srt").exists()
