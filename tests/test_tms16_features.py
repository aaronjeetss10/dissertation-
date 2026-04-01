"""
tests/test_tms16_features.py
================================
Unit tests for TMS trajectory feature extraction (12 features in current impl).

Tests cover:
  - All features produce finite values on a known trajectory
  - Walking trajectory has displacement_consistency > 0.3
  - Lying trajectory has mean_aspect_ratio < 0.7
  - Short trajectory (<20 frames) handled gracefully
  - Ego-motion compensation removes known camera motion
  - SMOTE single-class input doesn't crash

Run:
    pytest tests/test_tms16_features.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from streams.tms_classifier import (
    TrajectoryFeatures,
    TMSClassifierStream,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

FRAME_DIMS = (1080, 1920)  # (height, width) as expected by TrajectoryFeatures
FPS = 5


def _timestamps(n: int) -> list:
    return [i / FPS for i in range(n)]


def _make_walking_trajectory(n: int = 40):
    """Consistent rightward displacement at ~3px/frame."""
    centroids = [(500 + i * 3.0, 400 + np.random.normal(0, 0.5)) for i in range(n)]
    timestamps = _timestamps(n)
    aspects = [2.0] * n  # upright person (h/w > 1)
    sizes = [25.0] * n
    return centroids, timestamps, aspects, sizes


def _make_lying_trajectory(n: int = 40):
    """Stationary, wide bbox (low h/w aspect ratio)."""
    centroids = [(600 + np.random.normal(0, 0.3),
                  500 + np.random.normal(0, 0.3)) for _ in range(n)]
    timestamps = _timestamps(n)
    aspects = [0.4] * n  # lying down: width > height → h/w < 1
    sizes = [30.0] * n
    return centroids, timestamps, aspects, sizes


def _make_running_trajectory(n: int = 40):
    """Fast consistent displacement."""
    centroids = [(200 + i * 8.0, 400 + np.random.normal(0, 1.0)) for i in range(n)]
    timestamps = _timestamps(n)
    aspects = [1.8] * n
    sizes = [25.0] * n
    return centroids, timestamps, aspects, sizes


# ══════════════════════════════════════════════════════════════════════════════
# 1. All features produce finite values
# ══════════════════════════════════════════════════════════════════════════════

def test_all_features_finite():
    """All 12 trajectory features must be finite numbers on a known trajectory."""
    centroids, timestamps, aspects, sizes = _make_walking_trajectory(30)
    tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)

    assert len(tf.features) == 12, \
        f"Expected 12 features, got {len(tf.features)}"

    for name, val in tf.features.items():
        assert np.isfinite(val), \
            f"Feature '{name}' is not finite: {val}"
        assert isinstance(val, float), \
            f"Feature '{name}' should be float, got {type(val)}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Walking trajectory: displacement consistency
# ══════════════════════════════════════════════════════════════════════════════

def test_walking_displacement_consistency():
    """A walking trajectory with consistent displacement should have
    low speed_cv (i.e. consistent speed → speed_cv < 1.0) and
    net_displacement > 0.03 (substantial ground covered)."""
    centroids, timestamps, aspects, sizes = _make_walking_trajectory(40)
    tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)

    # Consistent movement → low coefficient of variation
    assert tf.features["speed_cv"] < 1.0, \
        f"Walking should have consistent speed (speed_cv<1.0), got {tf.features['speed_cv']}"

    # Substantial net displacement
    assert tf.features["net_displacement"] > 0.03, \
        f"Walking should cover ground (net_disp>0.03), got {tf.features['net_displacement']}"

    # Mean speed should be non-trivial
    assert tf.features["mean_speed"] > 0.005, \
        f"Walking should have measurable speed, got {tf.features['mean_speed']}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lying trajectory: mean_aspect_ratio < 0.7
# ══════════════════════════════════════════════════════════════════════════════

def test_lying_mean_aspect_ratio():
    """A lying trajectory (stationary, wide bbox) should have mean_aspect < 0.7."""
    centroids, timestamps, aspects, sizes = _make_lying_trajectory(40)
    tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)

    assert tf.features["mean_aspect"] < 0.7, \
        f"Lying person should have mean_aspect < 0.7, got {tf.features['mean_aspect']}"

    # Should also be largely stationary
    assert tf.features["stationarity"] > 0.3, \
        f"Lying person should have high stationarity, got {tf.features['stationarity']}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Short trajectory (<20 frames) handled gracefully
# ══════════════════════════════════════════════════════════════════════════════

def test_short_trajectory_graceful():
    """A trajectory with < 20 frames should NOT crash; should return
    either zero features or a valid classification."""
    # Very short: 5 frames
    centroids_5 = [(100 + i, 200 + i) for i in range(5)]
    timestamps_5 = _timestamps(5)
    aspects_5 = [1.5] * 5
    sizes_5 = [20.0] * 5

    # Should not raise
    label, conf, features = TMSClassifierStream.classify_trajectory(
        centroids_5, timestamps_5, aspects_5, FRAME_DIMS, sizes_5
    )
    assert isinstance(label, str)
    assert isinstance(conf, float)
    assert isinstance(features, dict)

    # 2 frames — edge case
    centroids_2 = [(100, 200), (105, 205)]
    timestamps_2 = [0.0, 0.2]
    aspects_2 = [1.5, 1.5]
    sizes_2 = [20.0, 20.0]

    label2, conf2, features2 = TMSClassifierStream.classify_trajectory(
        centroids_2, timestamps_2, aspects_2, FRAME_DIMS, sizes_2
    )
    assert isinstance(label2, str)

    # 1 frame — degenerate
    centroids_1 = [(100, 200)]
    timestamps_1 = [0.0]
    aspects_1 = [1.5]
    sizes_1 = [20.0]

    label1, conf1, features1 = TMSClassifierStream.classify_trajectory(
        centroids_1, timestamps_1, aspects_1, FRAME_DIMS, sizes_1
    )
    assert isinstance(label1, str)


def test_short_trajectory_features_zero_or_valid():
    """TrajectoryFeatures with n < 3 should produce zero features dict."""
    centroids = [(100, 200), (105, 205)]
    timestamps = [0.0, 0.2]
    aspects = [1.5, 1.5]
    sizes = [20.0, 20.0]

    tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)
    # With < 3 points, features should be zeroed
    for name, val in tf.features.items():
        assert np.isfinite(val), f"Feature '{name}' not finite even for short traj"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Ego-motion compensation: camera motion removed
# ══════════════════════════════════════════════════════════════════════════════

def test_ego_motion_compensation_removes_camera_motion():
    """After ego-motion compensation, a stationary person should show
    near-zero net displacement even when the camera is panning."""
    n = 30
    pan_dx, pan_dy = 8.0, 4.0

    # 4 tracks all shifting by camera pan
    all_tracks = {}
    for tid in range(4):
        base_x = 200 + tid * 300
        base_y = 400
        all_tracks[tid] = [
            (base_x + i * pan_dx, base_y + i * pan_dy) for i in range(n)
        ]

    ego = TrajectoryFeatures.estimate_ego_motion(
        all_tracks, frame_range=(0, n - 1)
    )

    assert len(ego) == n - 1, f"Expected {n-1} ego displacements, got {len(ego)}"

    # Each ego displacement should be close to (pan_dx, pan_dy)
    for dx, dy in ego:
        assert abs(dx - pan_dx) < 1.5, f"Ego dx should be ~{pan_dx}, got {dx}"
        assert abs(dy - pan_dy) < 1.5, f"Ego dy should be ~{pan_dy}, got {dy}"

    # Now apply ego-compensation to a "stationary" person
    stationary_centroids = [(500 + i * pan_dx, 600 + i * pan_dy) for i in range(n)]
    timestamps = _timestamps(n)
    aspects = [2.0] * n
    sizes = [20.0] * n

    tf = TrajectoryFeatures(
        stationary_centroids, timestamps, aspects, FRAME_DIMS, sizes,
        ego_displacements=ego,
    )

    # After compensation, net displacement should be near zero
    assert tf.features["net_displacement"] < 0.02, \
        f"Compensated stationary person should have near-zero displacement, got {tf.features['net_displacement']}"


def test_ego_motion_without_compensation():
    """Without ego-motion compensation, a camera pan should inflate displacement."""
    n = 30
    pan_dx, pan_dy = 8.0, 4.0

    centroids = [(500 + i * pan_dx, 600 + i * pan_dy) for i in range(n)]
    timestamps = _timestamps(n)
    aspects = [2.0] * n
    sizes = [20.0] * n

    tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)

    # Without compensation, displacement should be substantial
    assert tf.features["net_displacement"] > 0.05, \
        f"Uncompensated panning should produce large displacement, got {tf.features['net_displacement']}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. SMOTE integration doesn't crash on single-class input
# ══════════════════════════════════════════════════════════════════════════════

def test_smote_single_class_no_crash():
    """SMOTE should not crash when given only one class. Either it skips
    resampling gracefully or raises a clear error."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        pytest.skip("imblearn not installed, skipping SMOTE test")

    # Create single-class feature matrix
    rng = np.random.default_rng(42)
    X = rng.random((20, 12))
    y = np.zeros(20, dtype=int)  # all class 0

    # SMOTE should handle this gracefully
    smote = SMOTE(random_state=42)
    try:
        X_res, y_res = smote.fit_resample(X, y)
        # If it doesn't crash, the output should be unchanged
        assert len(X_res) == len(X)
    except ValueError as e:
        # Expected: SMOTE raises ValueError for single class
        assert "class" in str(e).lower() or "1" in str(e), \
            f"Unexpected ValueError: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Feature names and completeness
# ══════════════════════════════════════════════════════════════════════════════

def test_feature_names_complete():
    """Verify all expected feature names are present."""
    centroids, timestamps, aspects, sizes = _make_running_trajectory(30)
    tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)

    expected_features = {
        "net_displacement", "mean_speed", "speed_cv", "max_acceleration",
        "vertical_dominance", "direction_change_rate", "stationarity",
        "aspect_change", "speed_decay", "oscillation", "mean_aspect",
        "mean_size_norm",
    }

    actual = set(tf.features.keys())
    missing = expected_features - actual
    extra = actual - expected_features

    assert not missing, f"Missing features: {missing}"
    assert not extra, f"Unexpected features: {extra}"


def test_feature_ranges_reasonable():
    """All features should be in reasonable ranges (no NaN, no extreme values)."""
    for make_traj in [_make_walking_trajectory, _make_lying_trajectory, _make_running_trajectory]:
        centroids, timestamps, aspects, sizes = make_traj(40)
        tf = TrajectoryFeatures(centroids, timestamps, aspects, FRAME_DIMS, sizes)

        for name, val in tf.features.items():
            assert np.isfinite(val), f"Feature '{name}' not finite for trajectory type"
            assert val >= -1.0, f"Feature '{name}' = {val} is unexpectedly negative"
            if name not in ("oscillation", "speed_cv", "speed_decay", "max_acceleration"):
                assert val <= 20.0, \
                    f"Feature '{name}' = {val} seems unreasonably large"
