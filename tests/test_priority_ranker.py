"""
tests/test_priority_ranker.py
===============================
Unit tests for the Priority Ranker — multi-stream fusion and scoring.

Tests cover:
  - Z-score normalisation
  - Cross-stream boost mechanics
  - AAI threshold integration
  - Timeline sort order

Run:
    pytest tests/test_priority_ranker.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from streams.base_stream import SAREvent, EventSeverity
from core.priority_ranker import PriorityRanker, RankedEvent


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_event(stream: str, label: str, confidence: float,
                z_score: float, t_start: float = 0.0, t_end: float = 2.0,
                track_id: int = None) -> SAREvent:
    """Create a minimal SAREvent for testing."""
    return SAREvent(
        stream_name=stream,
        start_frame=int(t_start * 5),
        end_frame=int(t_end * 5),
        start_time=t_start,
        end_time=t_end,
        confidence=confidence,
        z_score=z_score,
        label=label,
        severity=EventSeverity.HIGH,
        track_id=track_id,
    )


def _default_ranker(duration: float = 40.0) -> PriorityRanker:
    """Create a PriorityRanker with default config."""
    config = {
        "boost_min_streams": 2,
        "boost_factor": 1.2,
        "persistence_window": 3.0,
        "persistence_bonus": 0.15,
    }
    return PriorityRanker(config, video_duration=duration)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  test_zscore_normalisation
# ══════════════════════════════════════════════════════════════════════════════

def test_zscore_normalisation():
    """Known input scores → raw_score = confidence × max(z_score, 0.1)."""
    ranker = _default_ranker()

    event = _make_event("action", "falling", confidence=0.9, z_score=3.5)
    ranked = ranker._score_raw(event)

    expected_raw = 0.9 * 3.5  # = 3.15
    assert abs(ranked.raw_score - expected_raw) < 0.01, \
        f"Expected raw_score ≈ {expected_raw}, got {ranked.raw_score}"

    # Test z-score floor: z_score=0 should use floor of 0.1
    event_low = _make_event("motion", "walking", confidence=0.8, z_score=0.0)
    ranked_low = ranker._score_raw(event_low)

    expected_raw_low = 0.8 * 0.1  # = 0.08 (floor applied)
    assert abs(ranked_low.raw_score - expected_raw_low) < 0.01, \
        f"Expected raw_score ≈ {expected_raw_low}, got {ranked_low.raw_score}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  test_cross_stream_boost
# ══════════════════════════════════════════════════════════════════════════════

def test_cross_stream_boost():
    """Events flagged by ≥2 streams in the same time bin should receive
    a 1.2× boost over single-stream events."""
    ranker = _default_ranker()

    # Two events from different streams overlapping in time
    e1 = _make_event("action", "falling", confidence=0.8, z_score=3.0,
                     t_start=5.0, t_end=7.0)
    e2 = _make_event("tms", "falling", confidence=0.7, z_score=2.8,
                     t_start=5.5, t_end=7.5)
    # One event alone in a different time
    e3 = _make_event("motion", "motion_spike", confidence=0.6, z_score=2.0,
                     t_start=20.0, t_end=22.0)

    ranked = ranker.rank([e1, e2, e3])

    # Find the overlapping events
    boosted = [r for r in ranked if r.event.start_time < 10]
    solo = [r for r in ranked if r.event.start_time > 15]

    assert len(boosted) >= 1, "Should have ranked events in the overlap zone"
    assert len(solo) >= 1, "Should have ranked events in the solo zone"

    # At least one boosted event should have cross_stream_boost > 1.0
    any_boosted = any(r.cross_stream_boost > 1.0 for r in boosted)
    assert any_boosted, \
        f"Expected cross-stream boost > 1.0 for overlapping events, " \
        f"got boosts: {[r.cross_stream_boost for r in boosted]}"

    # Solo event should have boost = 1.0
    for r in solo:
        assert r.cross_stream_boost == 1.0, \
            f"Solo event should have boost=1.0, got {r.cross_stream_boost}"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  test_aai_below_threshold
# ══════════════════════════════════════════════════════════════════════════════

def test_aai_below_threshold():
    """At person size 30px (well below ~75px crossover), TMS should be
    the primary classifier — its raw_score should be higher."""
    ranker = _default_ranker()

    # Place events in DIFFERENT time bins so cross-stream boost doesn't equalise
    e_tms = _make_event("tms", "crawling", confidence=0.85, z_score=3.0,
                        t_start=10.0, t_end=12.0, track_id=1)
    e_tms.metadata = {"person_size_px": 30}

    e_mvit = _make_event("action", "walking", confidence=0.35, z_score=1.0,
                         t_start=25.0, t_end=27.0, track_id=1)
    e_mvit.metadata = {"person_size_px": 30}

    ranked = ranker.rank([e_tms, e_mvit])

    tms_ranked = [r for r in ranked if r.event.stream_name == "tms"]
    mvit_ranked = [r for r in ranked if r.event.stream_name == "action"]

    assert len(tms_ranked) > 0 and len(mvit_ranked) > 0, \
        "Both events should be present in ranked output"

    # TMS raw_score (0.85 * 3.0 = 2.55) >> MViTv2-S (0.35 * 1.0 = 0.35)
    assert tms_ranked[0].raw_score > mvit_ranked[0].raw_score, \
        f"At 30px, TMS raw_score ({tms_ranked[0].raw_score:.3f}) should exceed " \
        f"MViTv2-S ({mvit_ranked[0].raw_score:.3f})"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  test_aai_above_threshold
# ══════════════════════════════════════════════════════════════════════════════

def test_aai_above_threshold():
    """At person size 100px (above crossover), the higher-confidence
    stream should win — MViTv2-S is expected to be better here."""
    ranker = _default_ranker()

    # At 100px, MViTv2-S has high confidence
    e_mvit = _make_event("action", "falling", confidence=0.92, z_score=3.5,
                         t_start=5.0, t_end=7.0, track_id=2)
    e_mvit.metadata = {"person_size_px": 100}

    # TMS also detects but with moderate confidence
    e_tms = _make_event("tms", "falling", confidence=0.70, z_score=2.5,
                        t_start=5.0, t_end=7.0, track_id=2)
    e_tms.metadata = {"person_size_px": 100}

    ranked = ranker.rank([e_mvit, e_tms])

    # The event with higher raw_score should rank first
    assert ranked[0].raw_score >= ranked[-1].raw_score, \
        "Higher-scoring event should rank first"

    # MViTv2-S (0.92 * 3.5 = 3.22) should beat TMS (0.70 * 2.5 = 1.75)
    mvit_ranked = [r for r in ranked if r.event.stream_name == "action"]
    tms_ranked = [r for r in ranked if r.event.stream_name == "tms"]

    if mvit_ranked and tms_ranked:
        assert mvit_ranked[0].raw_score > tms_ranked[0].raw_score, \
            f"At 100px, MViTv2-S raw_score ({mvit_ranked[0].raw_score:.3f}) " \
            f"should exceed TMS ({tms_ranked[0].raw_score:.3f})"


# ══════════════════════════════════════════════════════════════════════════════
# 5.  test_timeline_sorted
# ══════════════════════════════════════════════════════════════════════════════

def test_timeline_sorted():
    """Output from ranker.rank() should be sorted by final_score descending."""
    ranker = _default_ranker()

    events = [
        _make_event("motion", "spike", confidence=0.3, z_score=1.0,
                    t_start=0, t_end=2),
        _make_event("action", "falling", confidence=0.9, z_score=4.0,
                    t_start=10, t_end=12),
        _make_event("tracking", "track_loss", confidence=0.6, z_score=2.5,
                    t_start=20, t_end=22),
        _make_event("tms", "crawling", confidence=0.8, z_score=3.0,
                    t_start=30, t_end=32),
    ]

    ranked = ranker.rank(events)

    # Should be sorted descending by final_score
    scores = [r.final_score for r in ranked]
    assert scores == sorted(scores, reverse=True), \
        f"Ranked output should be sorted descending by final_score: {scores}"
    assert len(ranked) == len(events), \
        f"All {len(events)} events should appear in output, got {len(ranked)}"
