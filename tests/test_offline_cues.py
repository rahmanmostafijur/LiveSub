from offline_cues import (MAX_CUE_MS, MAX_LINE_COLS, MIN_GAP_MS, Cue, Word, break_lines,
                          build_cues, display_width, ends_sentence, fix_timing, validate_cues)


def speak(sentence: str, start_ms: int, per_word_ms: int = 300, gap_ms: int = 0):
    """Words of `sentence` spoken back to back, each with a leading space."""
    words, t = [], start_ms
    for raw in sentence.split():
        words.append(Word(" " + raw, t, t + per_word_ms))
        t += per_word_ms + gap_ms
    return words


def test_single_short_sentence_is_one_cue_with_word_timing():
    cues = build_cues(speak("Hello there my friend.", 5000))

    assert len(cues) == 1
    assert cues[0].lines == ("Hello there my friend.",)
    assert cues[0].start_ms == 5000
    assert cues[0].end_ms >= 5000 + 4 * 300


def test_splits_at_sentence_end():
    words = speak("I went home early today.", 0) + speak("Then I slept for hours.", 1500)

    cues = build_cues(words)

    assert [c.text for c in cues] == ["I went home early today.", "Then I slept for hours."]


def test_splits_at_a_pause_even_without_punctuation():
    words = speak("so we waited", 0) + speak("and nothing happened", 3000)

    cues = build_cues(words)

    assert [c.text for c in cues] == ["so we waited", "and nothing happened"]


def test_very_short_sentences_are_joined_instead_of_flashing():
    cues = build_cues(speak("Yes. I did.", 0, per_word_ms=200))

    assert len(cues) == 1
    assert cues[0].text == "Yes. I did."


def test_long_sentence_wraps_to_two_balanced_lines_within_limit():
    sentence = "This is a rather long sentence that will have to be wrapped onto two lines."

    cues = build_cues(speak(sentence, 0, per_word_ms=250))

    assert len(cues) == 1
    assert len(cues[0].lines) == 2
    assert all(display_width(line) <= MAX_LINE_COLS for line in cues[0].lines)
    assert abs(display_width(cues[0].lines[0]) - display_width(cues[0].lines[1])) < 15


def test_overlong_speech_is_split_into_several_cues():
    sentence = " ".join(["word"] * 60)

    cues = build_cues(speak(sentence, 0, per_word_ms=300))

    assert len(cues) > 1
    assert validate_cues(cues) == []
    assert all(len(c.lines) <= 2 for c in cues)


def test_cue_never_longer_than_max_even_for_slow_speech():
    cues = build_cues(speak("one two three four five six seven eight", 0, per_word_ms=2000))

    assert all(c.end_ms - c.start_ms <= MAX_CUE_MS for c in cues)


def test_short_cue_is_held_to_minimum_duration():
    cue = build_cues([Word(" Okay.", 1000, 1300)])[0]

    assert cue.end_ms - cue.start_ms == 1000


def test_minimum_duration_does_not_push_into_next_cue():
    words = [Word(" Okay.", 1000, 1300), Word(" Fine.", 2000, 2300)]   # 700 ms pause

    first, second = build_cues(words)

    assert first.end_ms == second.start_ms - MIN_GAP_MS     # held as long as allowed
    assert first.end_ms - first.start_ms < 1000


def test_overlapping_cues_are_separated_with_minimum_gap():
    cues = fix_timing([Cue(0, 3000, ("a",)), Cue(2500, 5000, ("b",))])

    assert validate_cues(cues) == []
    assert cues[1].start_ms == 2500           # the later cue keeps its onset
    assert cues[0].end_ms == 2500 - MIN_GAP_MS


def test_unsorted_and_identical_start_cues_are_ordered_and_separated():
    cues = fix_timing([Cue(4000, 5000, ("late",)), Cue(1000, 2000, ("x",)),
                       Cue(1000, 2500, ("y",))])

    assert validate_cues(cues) == []
    assert [c.lines[0] for c in cues][-1] == "late"


def test_fully_overlapping_words_still_give_valid_cues():
    words = speak("first sentence here.", 0) + speak("second one overlaps.", 400)

    assert validate_cues(build_cues(words)) == []


def test_zero_and_inverted_word_times_are_repaired():
    cues = build_cues([Word(" Hi", 2000, 2000), Word(" there.", 2100, 2050)])

    assert validate_cues(cues) == []
    assert cues[0].end_ms > cues[0].start_ms


def test_stretched_final_word_does_not_hold_cue_for_seconds():
    cue = build_cues([Word(" Goodbye.", 1000, 9000)])[0]

    assert cue.end_ms - cue.start_ms <= 2500


def test_abbreviation_is_not_a_sentence_end():
    assert not ends_sentence(" Dr.")
    assert ends_sentence(" done.")
    assert ends_sentence(' "Really?"')
    assert ends_sentence(" हाँ।")


def test_cjk_counts_double_width_and_wraps_without_spaces():
    chars = "これは日本語の長い文章です" * 3
    words = [Word(ch, i * 200, i * 200 + 200) for i, ch in enumerate(chars)]

    cues = build_cues(words)

    assert all(display_width(line) <= MAX_LINE_COLS for c in cues for line in c.lines)
    assert "".join("".join(c.lines) for c in cues) == chars


def test_combining_marks_do_not_count_toward_width():
    assert display_width("क्षि") < len("क्षि")
    assert display_width("한국어") == 6


def test_indic_text_is_kept_intact():
    text = "আমি তোমাকে ভালোবাসি।"
    cues = build_cues(speak(text, 0))

    assert " ".join(c.text for c in cues).replace("\n", " ") == text


def test_break_lines_returns_none_when_it_cannot_fit():
    assert break_lines([" word" * 1] * 1 + [" x" * 30] * 4) is None


def test_empty_input_gives_no_cues():
    assert build_cues([]) == []
    assert build_cues([Word("  ", 0, 100)]) == []
