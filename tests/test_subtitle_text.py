"""Tests for subtitle merging, wrapping and expiry."""
import pytest

from subtitle_text import SubtitleComposer, ends_sentence, wrap


def measure(text: str) -> int:
    return len(text) * 10          # 10 px per character


def composer(**kw) -> SubtitleComposer:
    args = dict(measure=measure, max_width=300, max_lines=2, merge_window_s=4.0,
                clear_after_s=4.0)
    args.update(kw)
    return SubtitleComposer(**args)


@pytest.mark.parametrize("text,expected", [
    ("It is for us.", True), ("Really?", True), ("Stop!", True), ('He said "go."', True),
    ("আমি ভালো আছি।", True), ("However,", False), ("and then", False),
    ("However...", False), ("Wait…", False), ("", False),
])
def test_ends_sentence(text, expected):
    assert ends_sentence(text) is expected


def test_unfinished_piece_is_continued():
    c = composer()
    c.add("Four score and seven years ago", 0.0)
    assert c.add("our fathers brought forth", 1.5) == [
        "Four score and seven years ago", "our fathers brought forth"]


def test_finished_sentence_is_replaced():
    c = composer()
    c.add("What they did here.", 0.0)
    assert c.add("It is for us", 1.0) == ["It is for us"]


def test_lowercase_piece_continues_even_after_period():
    c = composer()
    c.add("This is a disaster.", 0.0)
    lines = c.add("said one of them.", 1.5)
    assert " ".join(lines) == "This is a disaster. said one of them." and len(lines) == 2


def test_caseless_script_relies_on_punctuation_only():
    c = composer()
    c.add("안녕하세요.", 0.0)
    assert c.add("반갑습니다", 1.0) == ["반갑습니다"]


def test_piece_after_merge_window_starts_new_subtitle():
    c = composer(merge_window_s=2.0)
    c.add("and so", 0.0)
    assert c.add("we continue", 3.0) == ["we continue"]


def test_subtitle_clears_after_silence():
    c = composer(clear_after_s=4.0)
    c.add("Hello there", 0.0)
    assert c.lines(3.9) == ["Hello there"]
    assert c.lines(4.1) == []


def test_expired_text_is_not_merged_into():
    c = composer(merge_window_s=10.0, clear_after_s=2.0)
    c.add("unfinished", 0.0)
    assert c.add("fresh start", 5.0) == ["fresh start"]


def test_at_most_two_lines_and_latest_text_kept():
    c = composer()
    c.add("one two three four five six seven", 0.0)            # 33 chars -> 2 lines
    lines = c.add("eight nine ten eleven twelve thirteen", 1.0)
    assert len(lines) == 2
    assert lines[-1].endswith("thirteen")
    assert all(measure(line) <= 300 for line in lines)


def test_wrap_splits_long_words_without_spaces():
    lines = wrap("这是一个没有空格的很长很长很长的中文句子用来测试换行功能", measure, 100)
    assert all(measure(line) <= 100 for line in lines)
    assert "".join(lines) == "这是一个没有空格的很长很长很长的中文句子用来测试换行功能"


def test_clear_resets():
    c = composer()
    c.add("text", 0.0)
    c.clear()
    assert c.lines(0.5) == []
