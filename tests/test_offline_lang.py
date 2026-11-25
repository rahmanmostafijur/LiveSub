import io

from offline_lang import (BLOCK_MS, Chunk, aggregate, block_edges, collect_window, pick_windows,
                          plan_language, smooth_blocks, speech_ms)
from offline_progress import Progress, format_duration

MINUTE = 60_000


def hour_of_speech():
    """Speech 50 s out of every 60 s for one hour."""
    return [(m * MINUTE, m * MINUTE + 50_000) for m in range(60)]


def by_time(timeline):
    """Fake detector: language is looked up from the time of the first span of the window."""
    seen = []

    def detect(spans):
        seen.append(list(spans))
        start = spans[0][0]
        for lo, hi, lang in timeline:
            if lo <= start < hi:
                return {lang: 0.95, "xx": 0.02}
        return {"en": 0.5}
    detect.seen = seen
    return detect


# --- sampling -------------------------------------------------------------------------

def test_windows_are_spread_across_the_whole_file_not_the_start():
    windows = pick_windows(hour_of_speech(), 6)

    starts = [w[0][0] for w in windows]
    assert len(windows) == 6
    assert starts == sorted(starts)
    assert starts[0] < 10 * MINUTE and starts[-1] > 45 * MINUTE
    assert all(speech_ms(w) <= 30_000 for w in windows)


def test_window_contains_only_speech():
    spans = [(0, 4000), (60_000, 80_000), (200_000, 230_000)]

    window = collect_window(spans, offset_ms=2000, length_ms=10_000)

    assert window == [(2000, 4000), (60_000, 68_000)]


def test_short_file_gets_a_single_window_and_tiny_speech_none():
    assert len(pick_windows([(0, 20_000)], 6)) == 1
    assert pick_windows([(0, 2000)], 6) == []
    assert pick_windows([], 6) == []


def test_aggregate_picks_highest_mean_probability_and_reports_agreement():
    votes = [{"hi": 0.9, "en": 0.1}, {"hi": 0.8, "en": 0.2}, {"en": 0.6, "hi": 0.4}]

    assert aggregate(votes) == ("hi", 2 / 3)


# --- planning -------------------------------------------------------------------------

def test_single_language_file_is_one_chunk_in_the_dominant_language():
    detect = by_time([(0, 10 ** 9, "ko")])

    plan = plan_language(60 * MINUTE, hour_of_speech(), detect)

    assert plan.chunks == (Chunk(0, 60 * MINUTE, "ko"),)
    assert plan.dominant == "ko" and plan.agreement == 1.0 and not plan.mixed
    assert plan.samples == 6 and len(detect.seen) == 6        # six samples, not one


def test_one_odd_sample_does_not_make_a_file_mixed():
    calls = iter([{"ko": 0.9}] * 3 + [{"ja": 0.7}] + [{"ko": 0.9}] * 2)

    plan = plan_language(60 * MINUTE, hour_of_speech(), lambda spans: next(calls))

    assert plan.dominant == "ko" and not plan.mixed and len(plan.chunks) == 1


def test_forced_language_skips_detection():
    detect = by_time([])

    plan = plan_language(10 * MINUTE, hour_of_speech(), detect, forced="bn")

    assert plan.chunks == (Chunk(0, 10 * MINUTE, "bn"),) and plan.forced
    assert detect.seen == []


def test_no_speech_gives_a_default_plan_without_detecting():
    detect = by_time([])

    plan = plan_language(5 * MINUTE, [], detect)

    assert plan.samples == 0 and detect.seen == []


def test_mixed_language_file_is_split_into_chunks_by_block():
    timeline = [(0, 20 * MINUTE, "hi"), (20 * MINUTE, 40 * MINUTE, "en"),
                (40 * MINUTE, 10 ** 9, "hi")]

    plan = plan_language(60 * MINUTE, hour_of_speech(), by_time(timeline))

    assert plan.mixed
    assert [c.language for c in plan.chunks] == ["hi", "en", "hi"]
    assert plan.chunks[0].start_ms == 0 and plan.chunks[-1].end_ms == 60 * MINUTE
    for a, b in zip(plan.chunks, plan.chunks[1:]):
        assert a.end_ms == b.start_ms                          # no gaps, no overlaps
    # the language is judged per ~3 minute block, so the switch is found to within a block
    assert abs(plan.chunks[1].start_ms - 20 * MINUTE) <= BLOCK_MS


def test_block_edges_move_into_the_longest_nearby_pause():
    spans = [(0, 170_000), (176_000, 400_000)]              # a 6 s pause near the 180 s mark

    edges = block_edges(spans, 400_000)

    assert edges[0] == 0 and edges[-1] == 400_000
    assert edges[1] == 173_000


def test_isolated_weak_flip_is_absorbed_but_a_sure_one_is_kept():
    assert smooth_blocks(["hi", "en", "hi"], [0.95, 0.6, 0.95], "hi") == ["hi", "hi", "hi"]
    assert smooth_blocks(["hi", "en", "hi"], [0.95, 0.99, 0.95], "hi") == ["hi", "en", "hi"]


def test_unsure_and_empty_blocks_take_the_neighbouring_language():
    assert smooth_blocks([None, "ko", "ko", None], [0, 0.9, 0.9, 0], "en") == ["ko", "ko", "ko", "ko"]
    assert smooth_blocks(["ko", "ja", "ko"], [0.9, 0.3, 0.9], "en") == ["ko", "ko", "ko"]


def test_short_file_is_never_split():
    calls = iter([{"hi": 0.9}, {"en": 0.9}, {"hi": 0.9}, {"en": 0.9}])
    spans = [(0, 100_000)]

    plan = plan_language(BLOCK_MS - 1000, spans, lambda s: next(calls), samples=3)

    assert len(plan.chunks) == 1


# --- progress -------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_progress(total_ms=100_000, start_ms=0, tty=False):
    out, clock = io.StringIO(), Clock()
    return Progress("Transcribing", total_ms, start_ms, stream=out, clock=clock, is_tty=tty), out, clock


def test_progress_shows_percent_elapsed_and_eta():
    progress, _, clock = make_progress()
    clock.now += 30
    progress.update(25_000)

    line = progress.line()

    assert " 25.0%" in line and "elapsed 0:00:30" in line and "ETA 0:01:30" in line


def test_eta_is_unknown_before_any_progress():
    progress, _, _ = make_progress()

    assert "ETA --:--:--" in progress.line()


def test_real_time_factor_is_processing_time_over_audio_time():
    progress, _, clock = make_progress(total_ms=60_000)
    clock.now += 90
    progress.update(60_000)

    assert progress.real_time_factor() == 1.5


def test_resumed_run_measures_speed_only_on_what_it_processed():
    progress, _, clock = make_progress(total_ms=100_000, start_ms=40_000)
    clock.now += 30
    progress.update(70_000)

    assert progress.processed_s == 30
    assert progress.real_time_factor() == 1.0
    assert progress.eta_s() == 30


def test_progress_never_goes_backwards_or_past_the_end():
    progress, _, _ = make_progress(total_ms=10_000)
    progress.update(8000)
    progress.update(3000)
    assert progress.position_ms == 8000
    progress.update(99_999)
    assert progress.fraction == 1.0


def test_non_terminal_output_logs_a_line_per_ten_percent():
    progress, out, _ = make_progress()
    for pos in range(0, 100_001, 1000):
        progress.update(pos)

    lines = out.getvalue().strip().splitlines()
    assert 9 <= len(lines) <= 11 and "\r" not in out.getvalue()


def test_format_duration():
    assert format_duration(0) == "0:00:00"
    assert format_duration(3725.9) == "1:02:05"
    assert format_duration(-5) == "0:00:00"
