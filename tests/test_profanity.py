"""Profanity/slur masking tests."""
import pytest

from config import ProfanityConfig
from profanity import ProfanityFilter, mask_word

WORDS = ["fuck", "shit", "bitch", "dick", "spic", "retard"]


@pytest.fixture
def pf() -> ProfanityFilter:
    return ProfanityFilter(WORDS)


def test_mask_word_keeps_first_and_last_letter():
    assert mask_word("shit") == "s**t"
    assert mask_word("ok") == "**"


@pytest.mark.parametrize("text,expected", [
    ("What the fuck is this?", "What the f**k is this?"),
    ("FUCKING hell", "F*****G hell"),
    ("Those bitches!", "Those b*****s!"),
    ("He fucked up, shit.", "He f****d up, s**t."),
    ("retarded idea", "r******d idea"),
])
def test_listed_words_and_endings_are_masked(pf, text, expected):
    assert pf(text) == expected


@pytest.mark.parametrize("text", [
    "The class assassinated nobody in Scunthorpe.",
    "Charles Dickens wrote spicy stories.",
    "Shitake mushrooms",            # not a listed word + suffix
    "আমি ভালো আছি",                  # non-Latin text untouched
])
def test_innocent_words_are_untouched(pf, text):
    assert pf(text) == text


def test_disabled_filter_changes_nothing():
    assert ProfanityFilter(WORDS, enabled=False)("shit") == "shit"


def test_from_config_uses_builtin_list_and_extra_words():
    pf = ProfanityFilter.from_config(ProfanityConfig(extra_words="darn, heck"))
    assert pf("well shit, darn it, heck") == "well s**t, d**n it, h**k"
    assert ProfanityFilter.from_config(ProfanityConfig(enabled=False))("shit") == "shit"


def test_missing_word_file_falls_back_to_extra_words(tmp_path):
    cfg = ProfanityConfig(words_file=str(tmp_path / "missing.txt"), extra_words="darn")
    assert ProfanityFilter.from_config(cfg)("darn shit") == "d**n shit"
