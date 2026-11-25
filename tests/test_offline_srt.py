import pytest

from offline_cues import Cue
from offline_srt import (BOM, format_srt, format_timestamp, parse_srt, parse_timestamp,
                         seconds_to_ms, verify_srt, write_srt)

SAMPLES = {
    "bengali": "আমি তোমাকে ভালোবাসি।",
    "korean": "안녕하세요, 만나서 반갑습니다.",
    "hindi": "मुझे आपसे मिलकर खुशी हुई।",
    "tamil": "உங்களை சந்தித்ததில் மகிழ்ச்சி.",
    "malayalam": "നിങ്ങളെ കണ്ടതിൽ സന്തോഷം.",
}


@pytest.mark.parametrize("ms,expected", [
    (0, "00:00:00,000"),
    (1, "00:00:00,001"),
    (999, "00:00:00,999"),
    (1000, "00:00:01,000"),
    (61_005, "00:01:01,005"),
    (3_599_999, "00:59:59,999"),
    (3_600_000, "01:00:00,000"),
    (7_265_432, "02:01:05,432"),
    (100 * 3_600_000, "100:00:00,000"),
])
def test_format_timestamp(ms, expected):
    assert format_timestamp(ms) == expected
    assert parse_timestamp(expected) == ms


def test_negative_timestamp_is_clamped_to_zero():
    assert format_timestamp(-50) == "00:00:00,000"


@pytest.mark.parametrize("seconds,ms", [
    (0.0, 0), (1.2344, 1234), (1.2346, 1235), (59.9996, 60_000),
    (3599.9999, 3_600_000), (0.0004, 0), (-1.0, 0), (7200.0, 7_200_000),
])
def test_seconds_to_ms_rounds_to_nearest(seconds, ms):
    assert seconds_to_ms(seconds) == ms


def test_rounding_up_across_a_second_never_prints_1000_ms():
    assert format_timestamp(seconds_to_ms(59.9996)) == "00:01:00,000"
    assert format_timestamp(seconds_to_ms(3599.9996)) == "01:00:00,000"


def test_format_srt_numbers_cues_and_separates_with_blank_lines():
    cues = [Cue(1000, 2500, ("Hello",)), Cue(3000, 4000, ("Second", "line"))]

    assert format_srt(cues) == (
        "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond\nline\n")


def test_format_srt_of_no_cues_is_empty():
    assert format_srt([]) == ""
    assert parse_srt("") == []


def test_parse_round_trips_crlf_documents():
    cues = [Cue(1000, 2500, ("a",)), Cue(3000, 4000, ("b", "c"))]

    assert parse_srt(format_srt(cues).replace("\n", "\r\n")) == cues


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_srt("1\nnot a timing\ntext\n")


def test_written_file_is_utf8_without_bom_and_crlf(tmp_path):
    path = tmp_path / "out.srt"

    write_srt(path, [Cue(0, 1000, ("Hello",))])

    raw = path.read_bytes()
    assert not raw.startswith(BOM)
    assert raw == b"1\r\n00:00:00,000 --> 00:00:01,000\r\nHello\r\n"


@pytest.mark.parametrize("language", SAMPLES)
def test_non_latin_scripts_survive_a_write_and_read_back(tmp_path, language):
    text = SAMPLES[language]
    cues = [Cue(1000, 3000, (text,)), Cue(3500, 5000, (text[:6], text[6:]))]
    path = tmp_path / "subs.srt"

    write_srt(path, cues)

    raw = path.read_bytes()
    assert not raw.startswith(BOM)
    assert text in raw.decode("utf-8")                  # strict decode, no mojibake
    assert parse_srt(path.read_text(encoding="utf-8")) == cues


def test_all_five_scripts_in_one_file(tmp_path):
    cues = [Cue(i * 2000, i * 2000 + 1500, (t,)) for i, t in enumerate(SAMPLES.values())]
    path = tmp_path / "mixed.srt"

    write_srt(path, cues)

    decoded = path.read_text(encoding="utf-8")
    assert all(t in decoded for t in SAMPLES.values())


def test_verify_detects_a_bom(tmp_path):
    path = tmp_path / "bom.srt"
    path.write_bytes(BOM + format_srt([Cue(0, 1000, ("x",))]).encode())

    with pytest.raises(ValueError, match="BOM"):
        verify_srt(path, [Cue(0, 1000, ("x",))])


def test_verify_detects_a_mismatch_and_invalid_utf8(tmp_path):
    path = tmp_path / "bad.srt"
    write_srt(path, [Cue(0, 1000, ("x",))])

    with pytest.raises(ValueError):
        verify_srt(path, [Cue(0, 1000, ("y",))])
    path.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\n\xff\xfe\n")
    with pytest.raises(UnicodeDecodeError):
        verify_srt(path, [])


def test_write_replaces_an_existing_file_without_leaving_temp_files(tmp_path):
    path = tmp_path / "out.srt"
    path.write_text("old")

    write_srt(path, [Cue(0, 1000, ("new",))])

    assert "new" in path.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["out.srt"]
