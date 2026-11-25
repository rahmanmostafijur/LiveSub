"""Tests for the VAD segmentation state machine (with a scripted fake VAD)."""
import queue

import numpy as np
import pytest

from config import VadConfig
from queues import put_drop_oldest
from segmenter import FRAME_S, Segmenter, SegmenterWorker
from vad import FRAME_SAMPLES

CFG = VadConfig()


def fake_vad(frame: np.ndarray) -> float:
    """Each test frame is filled with its own speech probability."""
    return float(frame[0])


def build(*parts: tuple[float, float]) -> np.ndarray:
    """Audio from (seconds, speech_prob) parts, frame-aligned."""
    frames = []
    for seconds, prob in parts:
        frames.extend([np.full(FRAME_SAMPLES, prob, dtype=np.float32)]
                      * round(seconds / FRAME_S))
    return np.concatenate(frames)


def run(audio: np.ndarray, cfg: VadConfig = CFG, chunk: int = 1600, flush: bool = True):
    seg = Segmenter(cfg, fake_vad)
    out = []
    for i in range(0, len(audio), chunk):
        out.extend(seg.feed(audio[i:i + chunk]))
    if flush:
        tail = seg.flush()
        out.extend([tail] if tail else [])
    return out


def test_silence_produces_no_segments():
    assert run(build((5.0, 0.05))) == []


def test_single_utterance_ends_at_pause_with_preroll_and_pad():
    segs = run(build((1.0, 0.0), (1.0, 0.9), (1.0, 0.0)))
    assert len(segs) == 1
    s = segs[0]
    assert s.reason == "pause"
    assert s.overlap_s == 0
    assert s.start_s == pytest.approx(1.0 - CFG.pre_roll_s, abs=FRAME_S)
    assert s.duration_s == pytest.approx(CFG.pre_roll_s + 1.0 + CFG.pad_s, abs=2 * FRAME_S)
    assert len(s.audio) == round(s.duration_s / FRAME_S) * FRAME_SAMPLES


def test_short_blip_is_discarded():
    assert run(build((1.0, 0.0), (0.1, 0.9), (1.0, 0.0))) == []


def test_short_pause_splits_once_min_length_reached():
    segs = run(build((2.5, 0.9), (0.25, 0.1), (1.0, 0.9), (1.0, 0.0)))
    assert [s.reason for s in segs] == ["split", "pause"]
    assert all(s.overlap_s == 0 for s in segs)


def test_short_pause_before_min_length_does_not_split():
    segs = run(build((1.0, 0.9), (0.25, 0.1), (1.0, 0.9), (1.0, 0.0)))
    assert len(segs) == 1


def test_continuous_speech_is_cut_with_overlap_and_bounded_length():
    segs = run(build((10.0, 0.9), (1.0, 0.0)))
    assert len(segs) >= 3
    assert all(s.duration_s <= CFG.max_segment_s + 1e-9 for s in segs)
    assert segs[0].reason == "cut"
    assert segs[1].overlap_s == pytest.approx(CFG.overlap_s, abs=FRAME_S)
    # consecutive segments overlap by exactly the declared amount
    for prev, nxt in zip(segs, segs[1:]):
        assert prev.end_s - nxt.start_s == pytest.approx(nxt.overlap_s, abs=1e-9)


def test_cut_prefers_quietest_point_and_skips_overlap_at_real_pause():
    # a single silent frame at ~3.5s inside continuous speech
    segs = run(build((3.5, 0.9), (FRAME_S, 0.1), (3.0, 0.9), (1.0, 0.0)))
    first, second = segs[0], segs[1]
    assert first.reason == "cut"
    assert first.end_s == pytest.approx(3.5 + FRAME_S, abs=FRAME_S)
    assert second.overlap_s == 0


def test_cut_mid_speech_uses_overlap_at_dip():
    segs = run(build((3.5, 0.9), (FRAME_S, 0.4), (3.0, 0.9), (1.0, 0.0)))
    assert segs[0].end_s == pytest.approx(3.5 + FRAME_S, abs=FRAME_S)
    assert segs[1].overlap_s > 0


def test_overlap_only_remainder_is_not_emitted():
    # speech stops right after the cut: the next segment would hold only overlap
    segs = run(build((4.0, 0.9), (2.0, 0.0)))
    assert all(s.duration_s - s.overlap_s > CFG.min_speech_s for s in segs)


def test_flush_closes_open_segment():
    segs = run(build((1.5, 0.9)), flush=True)
    assert len(segs) == 1 and segs[0].reason == "flush"


def test_chunk_size_does_not_change_result():
    audio = build((0.5, 0.0), (6.0, 0.9), (0.3, 0.1), (2.0, 0.9), (1.0, 0.0))
    a = run(audio, chunk=1600)
    b = run(audio, chunk=FRAME_SAMPLES)
    c = run(audio, chunk=777)
    assert [(s.start_s, s.end_s) for s in a] == [(s.start_s, s.end_s) for s in b]
    assert [(s.start_s, s.end_s) for s in a] == [(s.start_s, s.end_s) for s in c]


def test_segment_audio_matches_source_samples():
    audio = build((0.5, 0.0), (1.5, 0.9), (1.0, 0.0))
    s = Segmenter(CFG, fake_vad).feed(audio)[0]
    start = round(s.start_s / FRAME_S) * FRAME_SAMPLES
    assert np.array_equal(s.audio, audio[start:start + len(s.audio)])


def test_config_validation_rejects_inverted_limits():
    with pytest.raises(ValueError):
        VadConfig(min_segment_s=4.0, max_segment_s=2.0)
    with pytest.raises(ValueError):
        VadConfig(threshold=0.3, neg_threshold=0.5)


def test_worker_requires_16khz():
    with pytest.raises(ValueError):
        SegmenterWorker(queue.Queue(), queue.Queue(), CFG, 48000, speech_prob=fake_vad)


def test_worker_flushes_when_audio_stops():
    in_q, out_q = queue.Queue(), queue.Queue()
    worker = SegmenterWorker(in_q, out_q, VadConfig(gap_flush_s=0.2), 16000,
                             speech_prob=fake_vad)
    worker.start()
    in_q.put(build((1.0, 0.9)))      # speech, then the stream simply stops
    segment = out_q.get(timeout=2.0)
    worker.stop()
    assert segment.reason == "flush"


def test_put_drop_oldest_keeps_newest():
    q = queue.Queue(maxsize=2)
    assert put_drop_oldest(q, 1) is False
    put_drop_oldest(q, 2)
    assert put_drop_oldest(q, 3) is True
    assert [q.get_nowait(), q.get_nowait()] == [2, 3]
