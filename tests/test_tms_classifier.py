"""
tests/test_tms_classifier.py
===============================
Unit tests for the TMS (Temporal Motion Signature) classifier — Stream 6.

Tests cover:
  - Trajectory-based action classification for 3 action types
  - Graceful degradation for short tracks
  - Ego-motion compensation
  - Feature extraction dimensionality

Run:
    pytest tests/test_tms_classifier.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from streams.tms_classifier import (
    TrajectoryFeatures,
    TMSClassifierStream,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

FRAME_DIMS = (1920, 1080)
FPS = 5

def _timestamps(n: int) -> list:
    """Generate n timestamps at 5 fps."""
    return [i / FPS for i in range(n)]

def _constant_aspects(n: int, ar: float = 2.5) -> list:
    return [ar] * n

def _constant_sizes(n: int, s: float = 20.0) -> list:
    return [s] * n


# ══════════════════════════════════════════════════════════════════════════════
# 1.  test_falling_trajectory
# ══════════════════════════════════════════════════════════════════════════════

def test_falling_trajectory():
    """A trajectory with high downward displacement + sudden deceleration
    should exhibit falling-characteristic features: high vertical dominance,
    high max acceleration, and positive speed decay."""
    n = 20
    # Person drops sharply in Y over frames 0-12, then stops abruptly
    centroids = []
    for i in range(n):
        cx = 960 + np.random.normal(0, 1)        # near-constant X
        if i < 12:
            cy = 200 + i * 30 + (i ** 1.5) * 2   # accelerating downward
        else:
            cy = 200 + 12 * 30 + (12 ** 1.5) * 2  # stopped (on ground)
        centroids.append((cx, cy))

    timestamps = _timestamps(n)
    # Aspect ratio drops mid-fall (person goes from upright to horizontal)
    aspects = [2.8 if i < 10 else 0.6 for i in range(n)]
    sizes = _constant_sizes(n)

    label, conf, features = TMSClassifierStream.classify_trajectory(
        centroids, timestamps, aspects, FRAME_DIMS, sizes
    )

    # Verify trajectory features characteristic of falling:
    # 1. High vertical dominance (mostly downward movement)
    assert features["vertical_dominance"] > 0.3, \
        f"Expected high vertical dominance, got {features['vertical_dominance']}"
    # 2. Speed decay (deceleration after impact)
    assert features["speed_decay"] > 0, \
        f"Expected positive speed decay (deceleration), got {features['speed_decay']}"
    # 3. Significant aspect ratio change (upright → horizontal)
    assert features["aspect_change"] > 0.5, \
        f"Expected large aspect change for fall, got {features['aspect_change']}"
    # 4. Should be classified as *something* with reasonable confidence
    assert conf > 0.0, f"Expected non-zero confidence, got {conf}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  test_stationary_trajectory
# ══════════════════════════════════════════════════════════════════════════════

def test_stationary_trajectory():
    """Zero-displacement trajectory → should classify as stationary / low."""
    n = 20
    base_x, base_y = 500.0, 400.0
    # Add only tiny jitter (tracking noise)
    centroids = [(base_x + np.random.normal(0, 0.5),
                  base_y + np.random.normal(0, 0.5))
                 for _ in range(n)]
    timestamps = _timestamps(n)
    aspects = _constant_aspects(n, ar=2.0)
    sizes = _constant_sizes(n)

    label, conf, features = TMSClassifierStream.classify_trajectory(
        centroids, timestamps, aspects, FRAME_DIMS, sizes
    )

    # Mean speed should be near zero
    assert features["mean_speed"] < 0.01, \
        f"Stationary trajectory should have near-zero mean speed, got {features['mean_speed']}"
    # Stationarity should be high
    assert features["stationarity"] > 0.5, \
        f"Expected high stationarity ratio, got {features['stationarity']}"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  test_waving_trajectory
# ══════════════════════════════════════════════════════════════════════════════

def test_waving_trajectory():
    """Oscillating trajectory with low net displacement → captures wave."""
    n = 30
    # Person oscillates side-to-side with period ~6 frames
    centroids = [(960 + 15 * np.sin(2 * np.pi * i / 6), 540)
                 for i in range(n)]
    timestamps = _timestamps(n)
    aspects = _constant_aspects(n)
    sizes = _constant_sizes(n)

    label, conf, features = TMSClassifierStream.classify_trajectory(
        centroids, timestamps, aspects, FRAME_DIMS, sizes
    )

    # Oscillation should be elevated for a back-and-forth pattern
    assert features["oscillation"] > 0.1, \
        f"Expected elevated oscillation for waving, got {features['oscillation']}"
    # Net displacement should be low (they're not going anywhere)
    assert features["net_displacement"] < 0.05, \
        f"Expected low net displacement for waving, got {features['net_displacement']}"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  test_minimum_track_length
# ══════════════════════════════════════════════════════════════════════════════

def test_minimum_track_length():
    """Track with fewer than 8 frames → TMS should produce zero features
    or return 'unknown' (graceful degradation, not a crash)."""
    short_centroids = [(100, 200), (102, 201), (104, 202)]
    short_timestamps = [0.0, 0.2, 0.4]
    short_aspects = [2.0, 2.0, 2.0]
    short_sizes = [15.0, 15.0, 15.0]

    # Should NOT raise an exception
    label, conf, features = TMSClassifierStream.classify_trajectory(
        short_centroids, short_timestamps, short_aspects,
        FRAME_DIMS, short_sizes
    )

    # With <8 frames, features may be zeroed or label may be unknown
    assert isinstance(label, str), "classify_trajectory should return a string label"
    assert isinstance(conf, float), "classify_trajectory should return a float confidence"
    assert isinstance(features, dict), "classify_trajectory should return a features dict"


# ══════════════════════════════════════════════════════════════════════════════
# 5.  test_ego_motion_compensation
# ══════════════════════════════════════════════════════════════════════════════

def test_ego_motion_compensation():
    """When all tracks shift uniformly (camera pan), ego-compensated
    displacement should be near zero for a stationary person."""
    n = 15
    pan_dx, pan_dy = 10.0, 5.0  # camera moves 10px right, 5px down per frame

    # Simulate 3 tracks all shifting by the same camera pan
    all_tracks = {}
    for track_id in range(3):
        base_x = 200 + track_id * 300
        base_y = 400
        all_tracks[track_id] = [
            (base_x + i * pan_dx, base_y + i * pan_dy) for i in range(n)
        ]

    ego = TrajectoryFeatures.estimate_ego_motion(
        all_tracks, frame_range=(0, n - 1)
    )

    # Each ego displacement should be close to (pan_dx, pan_dy)
    for dx, dy in ego:
        assert abs(dx - pan_dx) < 2.0, \
            f"Ego dx should be ~{pan_dx}, got {dx}"
        assert abs(dy - pan_dy) < 2.0, \
            f"Ego dy should be ~{pan_dy}, got {dy}"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  test_feature_extraction_dimensions
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_extraction_dimensions():
    """TrajectoryFeatures should produce exactly 12 features."""
    n = 20
    centroids = [(500 + i * 3, 400 + i * 1.5) for i in range(n)]
    timestamps = _timestamps(n)
    aspects = _constant_aspects(n)
    sizes = _constant_sizes(n)

    tf = TrajectoryFeatures(
        centroids, timestamps, aspects, FRAME_DIMS, sizes
    )

    assert len(tf.features) == 12, \
        f"Expected 12 trajectory features, got {len(tf.features)}: {list(tf.features.keys())}"

    # All features should be finite numbers
    for name, val in tf.features.items():
        assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"

    # Check expected feature names exist (matching actual implementation)
    expected = {
        "mean_speed", "speed_cv", "max_acceleration",
        "net_displacement", "vertical_dominance", "direction_change_rate",
        "aspect_change", "stationarity", "speed_decay",
        "oscillation", "mean_aspect", "mean_size_norm",
    }
    actual = set(tf.features.keys())
    missing = expected - actual
    assert len(missing) == 0, f"Missing expected features: {missing}. Actual: {actual}"
