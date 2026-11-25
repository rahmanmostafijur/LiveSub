"""Tests for text clean-up and the transcription worker (fake engine, no model)."""
import queue
import time

import numpy as np
import pytest

from config import TranscribeConfig
from segmenter import SAMPLE_RATE, Segment
from textproc import (collapse_repeats, echoes_context, is_hallucination, remove_overlap,
                      strip_broken)
from transcribe import (LanguageTracker, Recognition, Transcriber, batch_segments,
                        merge_segments)


def seg(start: float, end: float, overlap: float = 0.0, age: float = 0.0) -> Segment:
    audio = np.ones(int(round((end - start) * SAMPLE_RATE)), dtype=np.float32)
    return Segment(audio, start, end, overlap, created_at=time.monotonic() - age)


# -- textproc ---------------------------------------------------------------

def test_collapse_repeats_keeps_one_copy_of_looping_phrase():
    text = "the path is dark, the path is dark, the path is dark, the path is dark"
    assert collapse_repeats(text) == "the path is dark,"


def test_collapse_repeats_leaves_double_words_alone():
    assert collapse_repeats("no no, I said go") == "no no, I said go"


@pytest.mark.parametrize("text", ["", "  ", "Thanks for watching!", "you",
                                  "Subtitles by the Amara.org community", "ლლლლლლლ",
                                  "ლლლლ", "टी टी टी टी", "Hehehehe"])
def test_hallucinations_are_detected(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", ["Thank you for joining us.", "Asha Asha Asha",
                                  "Where are you?", "No no."])
def test_real_speech_is_kept(text):
    assert not is_hallucination(text)


def test_remove_overlap_drops_repeated_words():
    assert remove_overlap("we went to the market", "the Market and bought fish") \
        == "and bought fish"
    assert remove_overlap("hello there", "general Kenobi") == "general Kenobi"


def test_echoes_context_and_strip_broken():
    assert echoes_context("I don't know the truth.", "I don't know the truth")
    assert not echoes_context("I don't know.", "Who are you?")
    assert strip_broken("আশা�") == "আশা"


# -- language tracking ------------------------------------------------------

def tracker() -> LanguageTracker:
    return LanguageTracker("", 0.5, min_duration_s=1.5, min_confidence=0.5)


def test_language_tracker_ignores_one_unsure_segment():
    t = tracker()
    for _ in range(3):
        t.update({"bn": 0.6, "hi": 0.3}, 3.0)
    assert t.update({"mi": 0.25, "bn": 0.2}, 3.0)[0] == "bn"


def test_language_tracker_follows_confident_change():
    t = tracker()
    t.update({"bn": 0.6, "hi": 0.3}, 3.0)
    t.update({"en": 0.99}, 3.0)
    assert t.update({"en": 0.99}, 3.0)[0] == "en"


def test_short_segment_reuses_established_language():
    t = tracker()
    t.update({"bn": 0.8}, 3.0)
    # a confident but very short clip ("Hey. Hey.") must not switch the language
    assert t.update({"en": 0.95}, 0.8) == ("bn", pytest.approx(0.8))
    assert t.update({"zh": 0.9}, 1.2)[0] == "bn"
    assert t.current == "bn"


def test_low_confidence_detection_is_not_remembered():
    t = tracker()
    t.update({"bn": 0.7}, 3.0)
    for _ in range(5):
        assert t.update({"zh": 0.45, "bn": 0.3}, 4.0)[0] == "bn"


def test_without_history_short_segment_uses_best_guess_but_not_stored():
    t = tracker()
    assert t.update({"en": 0.9}, 0.7)[0] == "en"
    assert t.current is None
    assert t.update({"bn": 0.8}, 2.0)[0] == "bn"


def test_language_tracker_forced():
    assert LanguageTracker("JA", 0.5).update({"en": 0.9}, 0.5)[0] == "ja"


def test_parse_language_and_overrides():
    from transcribe import apply_overrides, parse_language

    assert parse_language("BN") == "bn"
    assert parse_language("auto") == ""
    with pytest.raises(ValueError):
        parse_language("bengali")
    cfg = TranscribeConfig(language="bn")
    assert apply_overrides(cfg, language="").language == ""          # --language auto
    assert apply_overrides(cfg, language=None).language == "bn"      # flag not given
    assert apply_overrides(cfg, language="hi", model="base").model == "base"


def test_config_language_auto_normalized_and_file_loaded(tmp_path):
    import json

    from config import load_config

    assert TranscribeConfig(language="Auto").language == ""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"transcribe": {"language": "bn"}}), encoding="utf-8")
    assert load_config(path).transcribe.language == "bn"
    with pytest.raises(ValueError):
        TranscribeConfig(language="b-n")


@pytest.mark.parametrize("argv,mode,language", [
    (["--mode", "english", "--language", "bn"], "english", "bn"),
    (["--language", "bn", "--mode", "English"], "english", "bn"),
    (["--mode", "original"], "original", "bn"),          # language from config
    (["--mode", "english", "--language", "auto"], "english", ""),
    ([], "original", "bn"),                              # nothing given: config values
])
def test_cli_mode_and_language_combine(argv, mode, language):
    from transcribe import build_arg_parser, config_from_args

    base = TranscribeConfig(language="bn")
    cfg = config_from_args(build_arg_parser().parse_args(argv), base)
    assert (cfg.mode, cfg.language) == (mode, language)


def test_cli_rejects_unknown_mode():
    from transcribe import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--mode", "spanish"])


def test_worker_starts_in_cli_mode_and_enter_toggle_still_works():
    from transcribe import build_arg_parser, config_from_args

    cfg = config_from_args(build_arg_parser().parse_args(["--mode", "english"]),
                           TranscribeConfig())
    engine = FakeEngine(["hola", "hello"])
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, cfg, lambda _c: engine)
    worker.start()
    in_q.put(seg(0, 2))
    assert out_q.get(timeout=2).mode == "english"
    assert worker.toggle_mode() == "original"     # what pressing Enter calls
    in_q.put(seg(10, 12))
    assert out_q.get(timeout=2).mode == "original"
    worker.stop()
    assert [c[1] for c in engine.calls] == ["translate", "transcribe"]


def test_project_config_uses_auto_detect():
    from config import DEFAULT_CONFIG_PATH, load_config

    assert load_config(DEFAULT_CONFIG_PATH).transcribe.language == ""


def test_tracker_switches_when_new_video_rejects_current_language():
    t = tracker()
    for _ in range(3):
        t.update({"en": 0.95}, 3.0)
    # a Malayalam video starts: moderately confident, but English is clearly gone
    assert t.update({"ml": 0.42, "ta": 0.3, "en": 0.02}, 3.0)[0] == "en"   # 1st: wait
    assert t.update({"ml": 0.45, "ta": 0.3, "en": 0.01}, 3.0)[0] == "ml"   # 2nd: switch


def test_tracker_does_not_switch_on_short_disagreements():
    t = tracker()
    t.update({"en": 0.95}, 3.0)
    for _ in range(4):
        assert t.update({"ko": 0.9, "en": 0.01}, 1.0)[0] == "en"


def test_set_forced_resets_history():
    t = tracker()
    t.update({"en": 0.95}, 3.0)
    t.set_forced("ko")
    assert t.update({"en": 0.99}, 3.0)[0] == "ko"
    t.set_forced("")
    assert t.current is None


def test_transcriber_set_language_reaches_engine_and_drops_context():
    class LanguageEngine(FakeEngine):
        def __init__(self, texts):
            super().__init__(texts)
            self.languages = []

        def set_language(self, code):
            self.languages.append(code)

    engine = LanguageEngine(["first part,", "second"])
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, TranscribeConfig(), lambda _c: engine)
    worker.start()
    in_q.put(seg(0, 2))
    out_q.get(timeout=2)
    worker.set_language("ko")
    assert worker.language == "ko"
    in_q.put(seg(2.5, 4))
    out_q.get(timeout=2)
    worker.stop()
    assert engine.languages == ["", "ko"]      # config language at start, then the switch
    assert engine.calls[1][2] is None          # old-language context not reused
    with pytest.raises(ValueError):
        worker.set_language("klingon")


# -- batching / merging -----------------------------------------------------

def test_batch_segments_respects_max_length():
    batches = batch_segments([seg(0, 3), seg(3, 6), seg(6, 9), seg(9, 12)], 7.0)
    assert [len(b) for b in batches] == [2, 2]


def test_merge_removes_overlap_and_pads_gaps():
    a, b, c = seg(0, 2), seg(1.6, 4, overlap=0.4), seg(5, 6)
    merged = merge_segments([a, b, c])
    expected = 2.0 + (2.4 - 0.4) + 0.3 + 1.0
    assert len(merged) == pytest.approx(expected * SAMPLE_RATE, abs=2)


# -- worker -------------------------------------------------------------------

class FakeEngine:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def recognize(self, audio, task, prompt):
        self.calls.append((len(audio), task, prompt))
        return Recognition(self.texts.pop(0), "en", 0.9, -0.2, 0.01)


def run_worker(segments, texts, **cfg_overrides):
    engine = FakeEngine(texts)
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, TranscribeConfig(**cfg_overrides), lambda _c: engine)
    for s in segments:
        in_q.put(s)
    worker.start()
    deadline = time.monotonic() + 3
    while worker.processed < len(segments) and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()
    results = []
    while not out_q.empty():
        results.append(out_q.get_nowait())
    return engine, results


def test_worker_emits_subtitle():
    engine, subs = run_worker([seg(0, 2)], ["Hello there."])
    assert [s.text for s in subs] == ["Hello there."]
    assert subs[0].mode == "original" and engine.calls[0][1] == "transcribe"


def test_worker_passes_previous_text_as_context():
    engine = FakeEngine(["First part,", "second part."])
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, TranscribeConfig(), lambda _c: engine)
    worker.start()
    in_q.put(seg(0, 2))
    out_q.get(timeout=2)
    in_q.put(seg(2.5, 4))
    out_q.get(timeout=2)
    worker.stop()
    assert engine.calls[0][2] is None and engine.calls[1][2] == "First part,"


def test_context_not_reused_after_long_real_pause():
    engine = FakeEngine(["first part,", "new video"])
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, TranscribeConfig(context_max_gap_s=3.0),
                         lambda _c: engine)
    worker.start()
    in_q.put(seg(0, 2, age=7.0))               # spoken 7 s ago
    out_q.get(timeout=2)
    in_q.put(seg(2.1, 3.0))                    # stream time continuous, wall clock not
    out_q.get(timeout=2)
    worker.stop()
    assert engine.calls[1][2] is None


def test_worker_skips_stale_segments():
    engine, subs = run_worker([seg(0, 2, age=30)], ["late"], max_lag_s=8)
    assert subs == [] and engine.calls == []


def test_worker_filters_hallucination():
    _, subs = run_worker([seg(0, 2)], ["Thanks for watching!"])
    assert subs == []


def test_worker_translate_mode_and_toggle():
    engine = FakeEngine(["one", "two"])
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = Transcriber(in_q, out_q, TranscribeConfig(), lambda _c: engine)
    assert worker.toggle_mode() == "english"
    worker.start()
    in_q.put(seg(0, 2))
    out_q.get(timeout=2)
    worker.set_mode("original")
    in_q.put(seg(10, 12))
    out_q.get(timeout=2)
    worker.stop()
    assert [c[1] for c in engine.calls] == ["translate", "transcribe"]
    with pytest.raises(ValueError):
        worker.set_mode("klingon")


def test_worker_reports_model_load_failure():
    def boom(_cfg):
        raise RuntimeError("no network")
    worker = Transcriber(queue.Queue(), queue.Queue(), TranscribeConfig(), boom)
    worker.start()
    assert worker.ready.wait(2)
    assert "no network" in worker.error


def test_transcribe_config_validation():
    with pytest.raises(ValueError):
        TranscribeConfig(mode="spanish")
    with pytest.raises(ValueError):
        TranscribeConfig(min_window_s=40)
