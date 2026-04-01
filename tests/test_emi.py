"""
tests/test_emi.py — Unit tests for Ego-Motion Intelligence (EMI).

Run: pytest tests/test_emi.py -v
"""
import sys, math
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "sartriage"))

from core.emi import (
    EMIExtractor, EMIFeatures, EMIMultiplierConfig, FlightPhase,
    classify_flight_phase, decompose_homography, emi_to_multiplier,
    generate_synthetic_homographies, aggregate_emi_multiplier,
)

def test_transit_phase_high_speed():
    f = EMIFeatures(translational_speed=25.0, rotational_rate=0.005,
                    hover_index=0.007, circling_index=0.0, descent_rate=0.0, deceleration=0.0)
    assert classify_flight_phase(f) == FlightPhase.TRANSIT

def test_transit_from_synthetic():
    ext = EMIExtractor(sigma_hover=5.0, circling_window=20, fps=5.0)
    Hs = generate_synthetic_homographies(n_frames=40, behavior="cruise", fps=5.0, rng=np.random.default_rng(42))
    feats = ext.extract_sequence(Hs)
    assert len(feats) == 39
    assert np.mean([f.translational_speed for f in feats]) > 10.0

def test_hovering_phase():
    f = EMIFeatures(translational_speed=0.5, rotational_rate=0.002,
                    hover_index=0.9, circling_index=0.0, descent_rate=0.0, deceleration=0.0)
    assert classify_flight_phase(f) == FlightPhase.HOVERING

def test_hovering_from_synthetic():
    ext = EMIExtractor(sigma_hover=5.0, circling_window=20, fps=5.0)
    Hs = generate_synthetic_homographies(n_frames=40, behavior="hover", fps=5.0, rng=np.random.default_rng(42))
    feats = ext.extract_sequence(Hs)
    assert np.mean([f.hover_index for f in feats]) > 0.7

def test_degenerate_identity_homography():
    tx, ty, rot, scale, sh = decompose_homography(np.eye(3))
    assert abs(tx) < 0.01 and abs(ty) < 0.01 and abs(rot) < 0.01

def test_near_degenerate_homography():
    H = np.array([[1e-10,0,0],[0,1e-10,0],[0,0,1]], dtype=np.float64)
    tx, ty, rot, scale, sh = decompose_homography(H)
    assert np.isfinite(tx) and np.isfinite(ty) and np.isfinite(rot)

def test_emi_with_identity_sequence():
    ext = EMIExtractor(sigma_hover=5.0, circling_window=20, fps=5.0)
    feats = ext.extract_sequence([np.eye(3) for _ in range(10)])
    assert len(feats) == 10
    for f in feats:
        assert f.hover_index > 0.9

def test_multiplier_bounds():
    cfg = EMIMultiplierConfig()
    cases = [
        EMIFeatures(translational_speed=0, hover_index=1.0),
        EMIFeatures(translational_speed=30, hover_index=0.0),
        EMIFeatures(translational_speed=0, hover_index=1.0, descent_rate=0.5, circling_index=1.0),
    ]
    for feat in cases:
        m = emi_to_multiplier(feat, cfg)
        assert cfg.min_multiplier <= m <= cfg.max_multiplier

def test_hover_increases_multiplier():
    cfg = EMIMultiplierConfig()
    assert emi_to_multiplier(EMIFeatures(hover_index=1.0), cfg) > emi_to_multiplier(EMIFeatures(hover_index=0.0), cfg)

def test_speed_decreases_multiplier():
    cfg = EMIMultiplierConfig()
    m_slow = emi_to_multiplier(EMIFeatures(translational_speed=1.0, hover_index=0.5), cfg)
    m_fast = emi_to_multiplier(EMIFeatures(translational_speed=30.0, hover_index=0.5), cfg)
    assert m_slow >= m_fast

def test_all_flight_phases_reachable():
    mapping = {
        FlightPhase.TRANSIT: EMIFeatures(translational_speed=25, rotational_rate=0.005, hover_index=0.01),
        FlightPhase.HOVERING: EMIFeatures(translational_speed=0.3, rotational_rate=0.001, hover_index=0.95),
        FlightPhase.DESCENDING: EMIFeatures(translational_speed=3, rotational_rate=0.01, hover_index=0.5, descent_rate=0.15),
        FlightPhase.APPROACHING: EMIFeatures(translational_speed=5, rotational_rate=0.01, hover_index=0.3, descent_rate=0.1, deceleration=-3.0),
        FlightPhase.CIRCLING: EMIFeatures(translational_speed=5, rotational_rate=0.05, hover_index=0.3, circling_index=0.5),
        FlightPhase.SCANNING: EMIFeatures(translational_speed=5, rotational_rate=0.02, hover_index=0.3),
    }
    for expected, feat in mapping.items():
        assert classify_flight_phase(feat) == expected, f"Expected {expected}"

def test_aggregate_emi_multiplier():
    fl = [EMIFeatures(hover_index=0.9, translational_speed=0.5),
          EMIFeatures(hover_index=0.1, translational_speed=15),
          EMIFeatures(hover_index=0.8, translational_speed=1)]
    res = aggregate_emi_multiplier(fl, [0, 1, 0], 3)
    assert len(res) == 3 and res[0] > 1.0 and res[2] == 1.0

def test_features_to_dict():
    f = EMIFeatures(translational_speed=5.2, rotational_rate=0.012, hover_index=0.8,
                    circling_index=0.3, descent_rate=0.05, deceleration=-1.5, pattern_deviation=2.3)
    d = f.to_dict()
    assert isinstance(d, dict) and len(d) == 7
