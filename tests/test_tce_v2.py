"""
tests/test_tce_v2.py
========================
Unit tests for Temporal Criticality Evolution v2 (tce_v2_pilot.py).

Tests cover:
  - Initial state assessment: lying geometry → CRITICAL_STATIC
  - Initial state assessment: upright geometry → MOVING_SLOW
  - Dwell escalation: score increases with time
  - Confidence modulation: high-confidence still → higher score
  - State transitions: MOVING_FAST → DECELERATING when speed drops
  - Failure case: stationary low AR must NOT get low priority

Run:
    pytest tests/test_tce_v2.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from tce_v2_pilot import assess_initial_state, tce_state_machine_v2


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_stationary_frames(n: int, cx: float = 500.0, cy: float = 400.0,
                            w: float = 60.0, h: float = 30.0,
                            jitter: float = 2.0) -> list:
    """Generate n frames of a stationary person.
    Each frame: [cx, cy, bbox_width, bbox_height]."""
    rng = np.random.default_rng(42)
    return [
        [cx + rng.normal(0, jitter), cy + rng.normal(0, jitter), w, h]
        for _ in range(n)
    ]


def _make_moving_frames(n: int, speed: float = 30.0,
                        w: float = 40.0, h: float = 80.0) -> list:
    """Generate n frames of a person moving rightward at `speed` px/frame."""
    rng = np.random.default_rng(42)
    return [
        [200 + i * speed + rng.normal(0, 1), 400 + rng.normal(0, 1), w, h]
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Initial state: lying geometry → CRITICAL_STATIC
# ══════════════════════════════════════════════════════════════════════════════

def test_initial_state_lying_geometry():
    """A stationary person with low AR (h/w < 0.6, i.e. wide bbox)
    should start as CRITICAL_STATIC."""
    # Wide bbox: w=60, h=25 → AR = h/w = 0.42
    frames = _make_stationary_frames(10, w=60.0, h=25.0, jitter=1.0)

    state, score = assess_initial_state(frames, speed_threshold=20.0)

    assert state == "CRITICAL_STATIC", \
        f"Lying geometry should → CRITICAL_STATIC, got {state}"
    assert score >= 0.7, \
        f"CRITICAL_STATIC score should be high (≥0.7), got {score}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Initial state: upright geometry → MOVING_SLOW
# ══════════════════════════════════════════════════════════════════════════════

def test_initial_state_upright_moving():
    """A person moving slowly with upright AR (h/w > 1.0) should start
    as MOVING_SLOW or MOVING_FAST."""
    # Upright bbox: w=30, h=80 → AR = 2.67
    # Speed must exceed threshold (20px/frame) to register as moving
    frames = [
        [200 + i * 25.0, 400 + np.random.normal(0, 1), 30.0, 80.0]
        for i in range(10)
    ]

    state, score = assess_initial_state(frames, speed_threshold=20.0)

    assert state in ("MOVING_SLOW", "MOVING_FAST"), \
        f"Upright moving person should → MOVING_SLOW/FAST, got {state}"
    assert score < 0.5, \
        f"Moving person should have moderate score (<0.5), got {score}"


def test_initial_state_upright_stationary():
    """A stationary person with upright AR should become SUSTAINED_STILL,
    NOT CRITICAL_STATIC (AR > 0.6)."""
    # Upright bbox: w=30, h=80 → AR = 2.67 > 0.6
    frames = _make_stationary_frames(10, w=30.0, h=80.0, jitter=1.0)

    state, score = assess_initial_state(frames, speed_threshold=20.0)

    # Should be SUSTAINED_STILL, not CRITICAL_STATIC (AR condition not met)
    assert state != "CRITICAL_STATIC", \
        f"Upright stationary should NOT be CRITICAL_STATIC, got {state}"
    assert state == "SUSTAINED_STILL", \
        f"Upright stationary should be SUSTAINED_STILL, got {state}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dwell escalation: score increases with time
# ══════════════════════════════════════════════════════════════════════════════

def test_dwell_escalation():
    """A person remaining stationary for longer should get a higher score."""
    # Short stationary period (20 frames)
    short_traj = _make_stationary_frames(20, w=60.0, h=25.0, jitter=1.0)
    state_short, score_short = tce_state_machine_v2(short_traj, speed_thresh=20.0)

    # Long stationary period (80 frames)
    long_traj = _make_stationary_frames(80, w=60.0, h=25.0, jitter=1.0)
    state_long, score_long = tce_state_machine_v2(long_traj, speed_thresh=20.0)

    assert score_long >= score_short, \
        f"Longer dwell should yield equal or higher score: " \
        f"long={score_long}, short={score_short}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Confidence modulation (via consistent stillness)
# ══════════════════════════════════════════════════════════════════════════════

def test_confidence_via_stillness():
    """A consistently still person (no jitter) should score higher than
    one with significant jitter but same overall displacement."""
    # Very still (jitter=0.1)
    still_traj = _make_stationary_frames(50, w=60.0, h=25.0, jitter=0.1)
    _, score_still = tce_state_machine_v2(still_traj, speed_thresh=20.0)

    # Jittery (jitter=15.0 — near the speed threshold)
    jittery_traj = _make_stationary_frames(50, w=60.0, h=25.0, jitter=15.0)
    _, score_jittery = tce_state_machine_v2(jittery_traj, speed_thresh=20.0)

    assert score_still >= score_jittery, \
        f"Steady stillness should score ≥ jittery: still={score_still}, jittery={score_jittery}"


# ══════════════════════════════════════════════════════════════════════════════
# 5. State transitions: MOVING_FAST → deceleration
# ══════════════════════════════════════════════════════════════════════════════

def test_state_transition_moving_to_stopped():
    """A person who starts moving fast then stops should transition to
    a stopped/collapsed state with elevated score."""
    # First 20 frames: moving fast
    moving = _make_moving_frames(20, speed=35.0, w=30, h=80)
    # Next 30 frames: stationary
    last_cx = moving[-1][0]
    last_cy = moving[-1][1]
    stopped = _make_stationary_frames(30, cx=last_cx, cy=last_cy,
                                      w=30, h=80, jitter=1.0)
    traj = moving + stopped

    state, score = tce_state_machine_v2(traj, speed_thresh=20.0)

    # Should not be in a "moving" state
    assert state not in ("MOVING_FAST", "MOVING_SLOW"), \
        f"Person who stopped should not be in {state}"
    # Should have elevated score from collapse/stopped pattern
    assert score > 0.15, \
        f"Stopped after moving should have elevated score, got {score}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Critical failure case: stationary + low AR must NOT get low priority
# ══════════════════════════════════════════════════════════════════════════════

def test_failure_case_stationary_low_ar_high_priority():
    """
    THE SPECIFIC FAILURE CASE THAT MOTIVATED TCE v2:
    A track starting stationary with low aspect ratio (person lying flat)
    must NOT get low priority. This was the bug in v1 where lying people
    got deprioritised because they never "transitioned" from movement.

    In v2, the initial state assessment should catch this immediately.
    """
    # 60 frames of a person lying still from the start
    traj = _make_stationary_frames(60, w=70.0, h=25.0, jitter=1.0)

    state, score = tce_state_machine_v2(traj, speed_thresh=20.0)

    # MUST NOT be low priority
    assert score >= 0.5, \
        f"CRITICAL FAILURE: Stationary person with low AR got score={score} (< 0.5). " \
        f"This is exactly the bug v2 was designed to fix!"

    # Should be in a critical/high-priority state
    assert state in ("CRITICAL_STATIC", "SUSTAINED_STILL", "COLLAPSED"), \
        f"Lying person should be in critical state, got {state}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ══════════════════════════════════════════════════════════════════════════════

def test_single_frame_trajectory():
    """Single-frame trajectory should return UNKNOWN gracefully."""
    traj = [[500, 400, 40, 80]]
    state, score = tce_state_machine_v2(traj)
    assert state == "UNKNOWN"
    assert score >= 0.0


def test_empty_trajectory():
    """Empty trajectory should return UNKNOWN gracefully."""
    state, score = tce_state_machine_v2([])
    assert state == "UNKNOWN"


def test_score_bounds():
    """Score should always be within [0.1, 0.99]."""
    for make_fn in [
        lambda: _make_stationary_frames(30, w=60, h=25, jitter=1),
        lambda: _make_moving_frames(30, speed=35),
        lambda: _make_stationary_frames(100, w=60, h=25, jitter=0.1),
    ]:
        traj = make_fn()
        state, score = tce_state_machine_v2(traj)
        assert 0.1 <= score <= 0.99, \
            f"Score {score} out of bounds for state {state}"
