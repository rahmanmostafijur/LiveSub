import numpy as np
import pytest

from offline_clean import (clean_segments, collapse_word_loops, cues_from_segments,
                           drop_segment_loops, is_filler, repair_edge_words)
from offline_cues import Word, validate_cues
from offline_lang import Chunk
from offline_progress import Progress
from offline_state import (SAMPLE_RATE, PassStore, RawSegment, StateDir, find_skipped_speech,
                           last_position_ms, plan_matches, run_pass, to_segment, words_from_text)


class _Sink:
    def write(self, _):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


def raw(start, end, text, prob=0.9):
    """A RawSegment whose words are spread evenly over start..end (seconds)."""
    parts = text.split()
    step = (end - start) / len(parts)
    words = tuple((" " + p, start + i * step, start + (i + 1) * step, prob) for i, p in enumerate(parts))
    return RawSegment(start, end, text, words)


def seg(start_ms, end_ms, text, chunk=0, prob=0.9):
    return to_segment(raw(start_ms / 1000, end_ms / 1000, text, prob), chunk, 0)


class FakeEngine:
    """Plays a script of segments (absolute seconds); can die after `crash_after` segments."""

    def __init__(self, script, crash_after=None, hidden=()):
        self.script, self.crash_after = script, crash_after
        self.hidden = set(hidden)                 # texts the model "skips" unless re-checked
        self.calls, self.emitted = [], 0

    def transcribe(self, audio, language, task, prompt, vad=True):
        offset_s = audio[0] / SAMPLE_RATE         # the test audio is a sample-index ramp
        self.calls.append({"offset_s": offset_s, "language": language, "prompt": prompt,
                           "len_s": len(audio) / SAMPLE_RATE, "vad": vad})
        for start, end, text in self.script:
            if vad and text in self.hidden:
                continue
            if start >= offset_s - 1e-6 and end <= offset_s + len(audio) / SAMPLE_RATE + 1e-6:
                if self.crash_after is not None and self.emitted >= self.crash_after:
                    raise KeyboardInterrupt
                self.emitted += 1
                yield raw(start - offset_s, end - offset_s, text)


SCRIPT = [(1.0, 3.0, "First sentence here."), (4.0, 6.0, "Second one follows."),
          (8.0, 10.0, "Third sentence now."), (12.0, 14.0, "Fourth and last one.")]


def ramp(seconds):
    return np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32)


def run(engine, store, chunks=(Chunk(0, 16000, "en"),)):
    return run_pass(engine, ramp(16), list(chunks), "transcribe", store,
                    Progress("t", 16000, is_tty=False, stream=_Sink()))


# --- resume ---------------------------------------------------------------------------

def test_uninterrupted_run_stores_every_segment(tmp_path):
    store = PassStore(tmp_path / "t.jsonl")

    segments = run(FakeEngine(SCRIPT), store)

    assert [s.text for s in segments] == [t for _, _, t in SCRIPT]
    loaded, chunks_done, finished = store.load()
    assert [s.text for s in loaded] == [s.text for s in segments]
    assert chunks_done == {0} and finished


def test_interrupted_run_resumes_without_repeating_or_losing_segments(tmp_path):
    store = PassStore(tmp_path / "t.jsonl")
    with pytest.raises(KeyboardInterrupt):
        run(FakeEngine(SCRIPT, crash_after=2), store)
    saved, _, finished = store.load()
    assert [s.text for s in saved] == ["First sentence here.", "Second one follows."]
    assert not finished

    resumed_engine = FakeEngine(SCRIPT)
    segments = run(resumed_engine, store)

    assert [s.text for s in segments] == [t for _, _, t in SCRIPT]
    assert resumed_engine.calls[0]["offset_s"] == 6.0           # continues after the last segment
    assert resumed_engine.calls[0]["prompt"] == "Second one follows."
    assert [(s.start_ms, s.end_ms) for s in segments] == [(1000, 3000), (4000, 6000),
                                                         (8000, 10000), (12000, 14000)]


def test_resumed_result_equals_uninterrupted_result(tmp_path):
    clean = run(FakeEngine(SCRIPT), PassStore(tmp_path / "a.jsonl"))
    store = PassStore(tmp_path / "b.jsonl")
    with pytest.raises(KeyboardInterrupt):
        run(FakeEngine(SCRIPT, crash_after=3), store)

    resumed = run(FakeEngine(SCRIPT), store)

    assert cues_from_segments(resumed) == cues_from_segments(clean)


def test_finished_chunks_are_not_transcribed_again(tmp_path):
    store = PassStore(tmp_path / "t.jsonl")
    chunks = (Chunk(0, 7000, "en"), Chunk(7000, 16000, "hi"))
    with pytest.raises(KeyboardInterrupt):
        run(FakeEngine(SCRIPT, crash_after=3), store, chunks)

    engine = FakeEngine(SCRIPT)
    segments = run(engine, store, chunks)

    assert [c["language"] for c in engine.calls] == ["hi"]      # chunk 0 was finished
    assert {s.chunk for s in segments} == {0, 1}
    assert len(segments) == 4


def test_multi_language_chunks_are_transcribed_in_their_language(tmp_path):
    engine = FakeEngine(SCRIPT)

    run(engine, PassStore(tmp_path / "t.jsonl"), (Chunk(0, 7000, "ko"), Chunk(7000, 16000, "en")))

    assert [c["language"] for c in engine.calls] == ["ko", "en"]
    assert [c["offset_s"] for c in engine.calls] == [0.0, 7.0]


def test_damaged_last_line_is_dropped_and_file_repaired(tmp_path):
    store = PassStore(tmp_path / "t.jsonl")
    store.append_segment(seg(1000, 3000, "kept"))
    with store.path.open("ab") as fh:
        fh.write(b'{"k":"seg","c":0,"s":4000,"e":50')            # power cut mid-write

    segments, _, _ = store.load()

    assert [s.text for s in segments] == ["kept"]
    store.append_segment(seg(4000, 5000, "after"))               # file must be appendable again
    assert [s.text for s in store.load()[0]] == ["kept", "after"]


def test_non_latin_text_survives_the_partial_file(tmp_path):
    store = PassStore(tmp_path / "t.jsonl")
    store.append_segment(seg(0, 2000, "আমি তোমাকে ভালোবাসি"))

    assert store.load()[0][0].text == "আমি তোমাকে ভালোবাসি"
    assert "আমি" in store.path.read_text(encoding="utf-8")


def test_plan_round_trip_and_mismatch_detection(tmp_path):
    state = StateDir(tmp_path / "movie.partial")
    meta = {"input_size": 10, "input_mtime_ns": 5, "model": "medium", "language": "auto",
            "audio_track": 0}
    chunks = [Chunk(0, 5000, "hi"), Chunk(5000, 9000, "en")]

    state.write_plan(meta, chunks)

    saved_meta, saved_chunks = state.read_plan()
    assert saved_chunks == chunks
    assert plan_matches(saved_meta, meta) == []
    assert plan_matches(saved_meta, {**meta, "model": "large-v3"}) == ["model"]
    assert plan_matches(saved_meta, {**meta, "input_size": 11}) == ["input_size"]
    state.remove()
    assert not state.exists()


def test_unreadable_plan_is_reported_as_missing(tmp_path):
    state = StateDir(tmp_path / "x.partial")
    state.path.mkdir()
    (state.path / "plan.json").write_text("{not json", encoding="utf-8")

    assert state.read_plan() is None


def test_last_position():
    assert last_position_ms([]) == 0
    assert last_position_ms([seg(0, 1000, "a"), seg(2000, 4500, "b")]) == 4500


# --- speech the model skipped ---------------------------------------------------------

SPEECH = [(1000, 3000), (4000, 6000), (8000, 10000), (12000, 14000)]


def test_speech_without_text_is_found_and_covered_speech_is_not():
    covered = [seg(1000, 3000, "a"), seg(4000, 6000, "b"), seg(12000, 14000, "d")]

    gaps = find_skipped_speech(SPEECH, covered)

    assert gaps == [(7700, 10300)]                 # the third stretch, padded by 300 ms


def test_short_blips_are_not_rechecked_and_close_spans_form_one_stretch():
    assert find_skipped_speech([(1000, 1800)], []) == []                  # under 1.5 s
    assert find_skipped_speech([(1000, 1900), (2400, 3400)], []) == [(700, 3700)]


def test_partly_transcribed_stretch_is_left_alone():
    assert find_skipped_speech([(0, 10_000)], [seg(0, 6000, "most of it")]) == []


def test_skipped_utterance_is_recovered_without_the_vad_and_stored(tmp_path):
    store = PassStore(tmp_path / "t.jsonl")
    engine = FakeEngine(SCRIPT, hidden={"Third sentence now."})
    notes = []

    segments = run_pass(engine, ramp(16), [Chunk(0, 16000, "en")], "transcribe", store,
                        Progress("t", 16000, is_tty=False, stream=_Sink()), SPEECH, notes.append)

    assert "Third sentence now." in [s.text for s in segments]
    assert [c["vad"] for c in engine.calls] == [True, False]
    assert (engine.calls[1]["offset_s"], engine.calls[1]["language"]) == (7.7, "en")
    assert any("skipped" in n for n in notes)
    assert "Third sentence now." in [s.text for s in store.load()[0]]      # survives a resume


def test_nothing_is_rechecked_when_all_speech_has_text(tmp_path):
    engine = FakeEngine(SCRIPT)

    run_pass(engine, ramp(16), [Chunk(0, 16000, "en")], "transcribe", PassStore(tmp_path / "t.jsonl"),
             Progress("t", 16000, is_tty=False, stream=_Sink()), SPEECH)

    assert len(engine.calls) == 1


# --- segment conversion and cleaning ---------------------------------------------------

def test_to_segment_applies_offset_and_rounds_to_ms():
    s = to_segment(raw(1.2346, 2.0, "hello there"), chunk=2, offset_ms=10_000)

    assert (s.chunk, s.start_ms, s.end_ms) == (2, 11235, 12000)
    assert s.words[0].start_ms == 11235


def test_segment_without_word_times_gets_even_words():
    s = to_segment(RawSegment(0.0, 2.0, "one two", ()), 0, 0)

    assert [w.text for w in s.words] == [" one", " two"]
    assert words_from_text("", 0, 1000) == ()


@pytest.mark.parametrize("text", ["Thanks for watching!", "Subtitles by the Amara.org community",
                                  "Thank you for watching.", "Please subscribe"])
def test_known_hallucinations_are_dropped(text):
    assert is_filler(seg(0, 2000, text))
    kept = clean_segments([seg(0, 2000, text), seg(3000, 5000, "Real speech here")])
    assert [s.text for s in kept] == ["Real speech here"]


def test_short_real_lines_survive_when_confident():
    assert not is_filler(seg(0, 1000, "Bye.", prob=0.95))
    assert is_filler(seg(0, 1000, "Bye.", prob=0.3))


def test_segment_repeated_in_a_loop_is_reduced_to_one():
    loop = [seg(i * 2000, i * 2000 + 1500, "I don't know.") for i in range(5)]

    kept = drop_segment_loops(loop + [seg(20000, 21000, "Different.")])

    assert [s.text for s in kept] == ["I don't know.", "Different."]


def test_two_repeats_are_real_dialogue_and_kept():
    assert len(drop_segment_loops([seg(0, 1000, "Yes."), seg(2000, 3000, "Yes.")])) == 2


def test_repetition_loop_inside_a_segment_is_collapsed():
    text = " ".join(["Thank you very much."] * 8)
    words = tuple(Word(" " + w, i * 200, i * 200 + 200) for i, w in enumerate(text.split()))

    kept = collapse_word_loops(words)

    assert "".join(w.text for w in kept).strip() == "Thank you very much."
    assert kept[0].start_ms == 0


def test_genuine_short_repeat_is_not_touched():
    words = tuple(Word(" " + w, i * 200, i * 200 + 200) for i, w in enumerate("No, no, no!".split()))

    assert collapse_word_loops(words) == words


def test_stretched_first_word_is_moved_next_to_the_real_speech():
    words = (Word(" We", 27890, 28970), Word(" decided", 30830, 31230), Word(" to", 31230, 31450))

    fixed = repair_edge_words(words)

    assert fixed[0].end_ms == 30830 - 40
    assert fixed[0].end_ms - fixed[0].start_ms == 250 + 90 * 2
    assert fixed[1:] == words[1:]


def test_stretched_last_word_is_moved_next_to_the_speech():
    words = (Word(" see", 1000, 1300), Word(" you", 1300, 1600), Word(" later", 3500, 5200))

    fixed = repair_edge_words(words)

    assert fixed[-1].start_ms == 1600 + 40
    assert fixed[-1].end_ms < 2500


def test_normal_edge_words_are_left_alone():
    words = (Word(" Well", 1000, 1400), Word(" then", 2200, 2500))   # a pause, but a normal word

    assert repair_edge_words(words) == words


def test_cues_from_segments_never_overlap_and_chunks_are_separate():
    segments = [seg(1000, 3000, "Hello there friend.", chunk=0),
                seg(2900, 5000, "Second overlapping line.", chunk=1)]

    cues = cues_from_segments(segments)

    assert validate_cues(cues) == []
    assert [c.text for c in cues] == ["Hello there friend.", "Second overlapping line."]


def test_mask_is_applied_to_cue_text_only_when_given():
    segments = [seg(1000, 3000, "What the fuck is this.")]

    plain = cues_from_segments(segments)
    masked = cues_from_segments(segments, mask=lambda line: line.replace("fuck", "f**k"))

    assert "fuck" in plain[0].text and "f**k" in masked[0].text
