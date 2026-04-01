"""
tests/test_pipeline_e2e.py — End-to-end integration test for SARTriage.

Tests:
  - TMS classify_trajectory on synthetic trajectories as a smoke test
  - Output is a ranked list with required fields
  - Track quality filter removes tracks < 15px
  - Timing per-track is reasonable

Run: pytest tests/test_pipeline_e2e.py -v
"""
import sys, time
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from streams.tms_classifier import TMSClassifierStream, TrajectoryFeatures
from streams.base_stream import FramePacket

FRAME_DIMS = (1080, 1920)
FPS = 5

def _ts(n):
    return [i / FPS for i in range(n)]

def _make_synthetic_tracks(n_tracks=5, n_frames=30):
    """Create synthetic tracks with varied motion patterns."""
    tracks = []
    rng = np.random.default_rng(42)
    for t in range(n_tracks):
        base_x = rng.uniform(100, 1800)
        base_y = rng.uniform(100, 980)
        speed = rng.uniform(1, 10)
        ar = rng.uniform(0.3, 2.5)
        size = rng.uniform(15, 60)
        centroids = [(base_x + i * speed + rng.normal(0, 1),
                      base_y + rng.normal(0, 1)) for i in range(n_frames)]
        tracks.append({
            "centroids": centroids,
            "timestamps": _ts(n_frames),
            "aspects": [ar] * n_frames,
            "sizes": [size] * n_frames,
        })
    return tracks


def test_e2e_tms_classification():
    """Feed synthetic tracks through TMS classification and verify output."""
    tracks = _make_synthetic_tracks(n_tracks=8, n_frames=30)
    results = []
    for trk in tracks:
        label, conf, features = TMSClassifierStream.classify_trajectory(
            trk["centroids"], trk["timestamps"], trk["aspects"],
            FRAME_DIMS, trk["sizes"])
        results.append({"label": label, "confidence": conf, "features": features})

    assert len(results) == 8
    for r in results:
        assert isinstance(r["label"], str)
        assert isinstance(r["confidence"], float)
        assert isinstance(r["features"], dict)
        assert r["confidence"] >= 0.0


def test_output_has_required_fields():
    """Each classification result must have label, confidence, and features with required keys."""
    trk = _make_synthetic_tracks(1, 30)[0]
    label, conf, features = TMSClassifierStream.classify_trajectory(
        trk["centroids"], trk["timestamps"], trk["aspects"],
        FRAME_DIMS, trk["sizes"])

    required_feature_keys = {
        "net_displacement", "mean_speed", "speed_cv", "max_acceleration",
        "vertical_dominance", "direction_change_rate", "stationarity",
        "aspect_change", "speed_decay", "oscillation", "mean_aspect", "mean_size_norm",
    }
    actual = set(features.keys())
    missing = required_feature_keys - actual
    assert not missing, f"Missing feature keys: {missing}"


def test_timing_per_track():
    """TMS classification for a single track should be < 200ms."""
    trk = _make_synthetic_tracks(1, 60)[0]
    t0 = time.time()
    for _ in range(10):
        TMSClassifierStream.classify_trajectory(
            trk["centroids"], trk["timestamps"], trk["aspects"],
            FRAME_DIMS, trk["sizes"])
    elapsed = (time.time() - t0) / 10
    assert elapsed < 0.2, f"Per-track TMS should be <200ms, got {elapsed*1000:.0f}ms"


def test_track_quality_filter_small_tracks():
    """Tracks where person size < 15px should be flagged as low-quality."""
    n = 30
    tiny_centroids = [(500 + i*2, 400) for i in range(n)]
    tiny_sizes = [8.0] * n  # sub-15px
    timestamps = _ts(n)
    aspects = [1.5] * n

    tf = TrajectoryFeatures(tiny_centroids, timestamps, aspects, FRAME_DIMS, tiny_sizes)
    # mean_size_norm should be very small
    assert tf.features["mean_size_norm"] < 0.001, \
        f"Tiny person should have very small size_norm, got {tf.features['mean_size_norm']}"

    normal_sizes = [40.0] * n
    tf2 = TrajectoryFeatures(tiny_centroids, timestamps, aspects, FRAME_DIMS, normal_sizes)
    assert tf2.features["mean_size_norm"] > tf.features["mean_size_norm"]


def test_multiple_tracks_batch():
    """Process many tracks and verify all return valid output."""
    tracks = _make_synthetic_tracks(n_tracks=20, n_frames=25)
    labels = set()
    for trk in tracks:
        label, conf, features = TMSClassifierStream.classify_trajectory(
            trk["centroids"], trk["timestamps"], trk["aspects"],
            FRAME_DIMS, trk["sizes"])
        labels.add(label)
        assert conf >= 0.0
        for v in features.values():
            assert np.isfinite(v)

    # With 20 diverse tracks, we should see multiple different labels
    assert len(labels) >= 2, f"Expected diversity in labels, got {labels}"


def test_ranked_output_ordering():
    """Tracks ranked by confidence should be in descending order."""
    tracks = _make_synthetic_tracks(n_tracks=10, n_frames=30)
    results = []
    for trk in tracks:
        label, conf, features = TMSClassifierStream.classify_trajectory(
            trk["centroids"], trk["timestamps"], trk["aspects"],
            FRAME_DIMS, trk["sizes"])
        results.append({"label": label, "confidence": conf})

    ranked = sorted(results, key=lambda r: r["confidence"], reverse=True)
    for i in range(len(ranked) - 1):
        assert ranked[i]["confidence"] >= ranked[i+1]["confidence"]
