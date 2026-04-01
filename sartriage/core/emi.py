"""
core/emi.py
============
Ego-Motion Intelligence (EMI) — extracts drone flight behaviour features
from frame-to-frame homographies and converts them into priority score
multipliers.

The key insight: a drone autonomously *circles* or *hovers* over a
region of interest far more often than over background.  When the
operator sees something suspicious they stop, circle, or descend to
inspect.  EMI captures these five behavioural signals:

    1. translational_speed  — how fast the drone is translating (px/frame)
    2. rotational_rate      — how much the view is rotating (rad/frame)
    3. hover_index          — how "still" the drone is (0=speeding, 1=perfect hover)
    4. circling_index       — how orbit-like the trajectory is
    5. descent_rate         — proxy for altitude change via scale change

Each feature is derived from the 3×3 homography matrix H estimated
between consecutive frames (via ORB / RANSAC or equivalent).

Multiplier logic
-----------------
EMI features produce a multiplicative factor ∈ [1.0, max_multiplier]
for the event priority score:

    multiplier = 1.0 + w₁·hover + w₂·circling + w₃·descent - w₄·speed

So if the drone is hovering (high hover_index) and descending (positive
descent_rate), the events in that temporal region get a substantial
priority boost.  If the drone is cruising at high speed, multiplier ≈ 1.0
(no boost — the operator isn't lingering on anything).

All thresholds are read from ``config.yaml`` under ``ranker.emi``.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════════════
# Flight Phase Classification (7 phases from spec)
# ════════════════════════════════════════════════════════════════════════

class FlightPhase(enum.Enum):
    """Operational flight phases with associated attention scores.

    Each phase indicates a different level of operator/system attention
    to the ground below.  Higher attention → events more likely to be
    true positives.
    """
    TRANSIT     = "transit"       # high speed, linear path
    SCANNING    = "scanning"      # moderate speed, sweeping rotation
    INTEREST    = "interest"      # decelerating, slight circling
    HOVERING    = "hovering"      # near-zero speed, sustained
    CIRCLING    = "circling"      # moderate speed, high rotation
    DESCENDING  = "descending"    # negative altitude change
    APPROACHING = "approaching"   # deceleration + descent


_FLIGHT_PHASE_ATTENTION = {
    FlightPhase.TRANSIT:     0.1,
    FlightPhase.SCANNING:    0.3,
    FlightPhase.INTEREST:    0.6,
    FlightPhase.HOVERING:    0.8,
    FlightPhase.CIRCLING:    0.7,
    FlightPhase.DESCENDING:  0.9,
    FlightPhase.APPROACHING: 1.0,
}


# ════════════════════════════════════════════════════════════════════════
# EMI Feature Vector
# ════════════════════════════════════════════════════════════════════════

@dataclass
class EMIFeatures:
    """Seven-dimensional ego-motion feature vector for a single frame pair.

    Attributes
    ----------
    translational_speed : float
        Magnitude of (tx, ty) from the homography, in pixels/frame.
    rotational_rate : float
        Absolute rotation angle extracted from H, in radians/frame.
    hover_index : float ∈ [0, 1]
        Soft measure of hovering.  1.0 = perfect hover.
    circling_index : float ∈ [0, 1]
        How "orbit-like" the recent trajectory is.
    descent_rate : float
        Proxy for altitude decrease from scale change.
    deceleration : float
        Speed change over the sliding window (negative = decelerating).
    pattern_deviation : float
        Distance from fitted linear flight path (0 = on pattern).
    """
    translational_speed: float = 0.0
    rotational_rate: float = 0.0
    hover_index: float = 0.0
    circling_index: float = 0.0
    descent_rate: float = 0.0
    deceleration: float = 0.0
    pattern_deviation: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "translational_speed": round(self.translational_speed, 4),
            "rotational_rate": round(self.rotational_rate, 6),
            "hover_index": round(self.hover_index, 4),
            "circling_index": round(self.circling_index, 4),
            "descent_rate": round(self.descent_rate, 4),
            "deceleration": round(self.deceleration, 4),
            "pattern_deviation": round(self.pattern_deviation, 4),
        }


# ════════════════════════════════════════════════════════════════════════
# Homography decomposition
# ════════════════════════════════════════════════════════════════════════

def decompose_homography(
    H: np.ndarray,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> Tuple[float, float, float, float, float]:
    """Extract interpretable motion parameters from a 3×3 homography.

    The homography H maps points from frame t to frame t+1:

        p_{t+1} = H · p_t

    We decompose it into translation, rotation, and scale via the
    affine approximation (valid when the perspective warp is small,
    which is typical for drone footage with distant ground planes).

    Parameters
    ----------
    H : (3, 3) ndarray
        Frame-to-frame homography, normalised so H[2,2] ≈ 1.
    frame_width, frame_height : int
        Frame dimensions (for normalising translation).

    Returns
    -------
    tx : float
        Horizontal translation (pixels).
    ty : float
        Vertical translation (pixels).
    rotation : float
        Rotation angle (radians).  Positive = counter-clockwise.
    scale : float
        Scale factor.  >1 = scene getting larger (drone descending).
    shear : float
        Shear component (usually negligible for drone footage).
    """
    # Normalise H so H[2,2] = 1
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]

    # Translation: H[0,2] and H[1,2]
    tx = H[0, 2]
    ty = H[1, 2]

    # The upper-left 2×2 block encodes rotation + scale + shear
    a, b = H[0, 0], H[0, 1]
    c, d = H[1, 0], H[1, 1]

    # Scale = sqrt(det(upper_left_2x2))
    det = a * d - b * c
    scale = math.sqrt(abs(det)) if abs(det) > 1e-10 else 1.0

    # Rotation = atan2(c, a)  (from the rotation component of the affine)
    rotation = math.atan2(c, a)

    # Shear (skew) — usually ≈ 0 for rigid drone motion
    shear = (a * b + c * d) / max(det, 1e-10)

    return tx, ty, rotation, scale, shear


# ════════════════════════════════════════════════════════════════════════
# EMI Extractor
# ════════════════════════════════════════════════════════════════════════

class EMIExtractor:
    """Extract Ego-Motion Intelligence features from homography sequences.

    Maintains a sliding window of recent translations and rotations to
    compute the circling_index (requires temporal context).

    Parameters
    ----------
    sigma_hover : float
        Softness of the hover index exponential.  Lower = stricter
        definition of hover.  Default 5.0 (pixels/frame).
    circling_window : int
        Number of recent frames used to estimate circling behaviour.
    fps : float
        Video frame rate (for converting rates to /second).
    frame_dims : tuple
        (width, height) of the video frame.
    """

    def __init__(
        self,
        sigma_hover: float = 5.0,
        circling_window: int = 20,
        fps: float = 5.0,
        frame_dims: Tuple[int, int] = (1920, 1080),
    ):
        self.sigma_hover = sigma_hover
        self.circling_window = circling_window
        self.fps = fps
        self.frame_width, self.frame_height = frame_dims

        # Sliding window buffers
        self._tx_history: deque = deque(maxlen=circling_window)
        self._ty_history: deque = deque(maxlen=circling_window)
        self._rot_history: deque = deque(maxlen=circling_window)
        self._speed_history: deque = deque(maxlen=circling_window)

    def reset(self) -> None:
        """Clear the sliding window (call between videos)."""
        self._tx_history.clear()
        self._ty_history.clear()
        self._rot_history.clear()
        self._speed_history.clear()

    def extract(self, H: np.ndarray) -> EMIFeatures:
        """Extract 5 EMI features from a single frame-to-frame homography.

        Parameters
        ----------
        H : (3, 3) ndarray
            Homography mapping frame[t] → frame[t+1].

        Returns
        -------
        EMIFeatures
            The 5-dimensional feature vector.
        """
        tx, ty, rotation, scale, _ = decompose_homography(
            H, self.frame_width, self.frame_height,
        )

        # 1. Translational speed (px/frame)
        speed = math.sqrt(tx ** 2 + ty ** 2)

        # 2. Rotational rate (rad/frame)
        rot_rate = abs(rotation)

        # 3. Hover index: exp(-speed / σ)
        hover = math.exp(-speed / self.sigma_hover)

        # 4. Circling index (computed from sliding window)
        self._tx_history.append(tx)
        self._ty_history.append(ty)
        self._rot_history.append(rotation)
        self._speed_history.append(speed)
        circling = self._compute_circling_index()

        # 5. Descent rate (scale change per second)
        descent = (scale - 1.0) * self.fps

        # 6. Deceleration (speed change over window)
        if len(self._speed_history) >= 2:
            deceleration = self._speed_history[-1] - self._speed_history[0]
        else:
            deceleration = 0.0

        # 7. Pattern deviation (distance from linear fit)
        pattern_deviation = self._compute_pattern_deviation()

        return EMIFeatures(
            translational_speed=speed,
            rotational_rate=rot_rate,
            hover_index=hover,
            circling_index=circling,
            descent_rate=descent,
            deceleration=deceleration,
            pattern_deviation=pattern_deviation,
        )

    def extract_sequence(
        self, homographies: Sequence[np.ndarray]
    ) -> List[EMIFeatures]:
        """Extract EMI features for a sequence of homographies.

        Parameters
        ----------
        homographies : list of (3,3) ndarrays
            N homographies from consecutive frame pairs.

        Returns
        -------
        list[EMIFeatures]
            N feature vectors.
        """
        self.reset()
        return [self.extract(H) for H in homographies]

    def _compute_circling_index(self) -> float:
        """Compute how orbit-like the recent trajectory is.

        A drone circling a point of interest will show:
        1. Consistent rotation in one direction
        2. Translation vectors that rotate (the heading changes
           while the drone moves sideways)
        3. Low net displacement (it comes back close to where it started)

        We combine three sub-signals:

        rotation_consistency
            = |mean(rotations)| / mean(|rotations|)
            High when all rotations are in the same direction.

        heading_variance
            Standard deviation of translation heading angles.
            High heading variance + consistent rotation = circling.

        return_ratio
            = 1 - (net_displacement / total_path_length)
            High when the drone returns near its start.

        The final circling_index is the geometric mean of these three.
        """
        if len(self._tx_history) < 3:
            return 0.0

        # ── Rotation consistency ──
        rots = np.array(self._rot_history)
        abs_rots = np.abs(rots)
        mean_abs_rot = abs_rots.mean()
        if mean_abs_rot > 1e-8:
            rot_consistency = abs(rots.mean()) / mean_abs_rot
        else:
            rot_consistency = 0.0

        # ── Heading variance ──
        txs = np.array(self._tx_history)
        tys = np.array(self._ty_history)
        headings = np.arctan2(tys, txs)  # heading angle per frame
        # Circular standard deviation
        heading_var = 1.0 - np.abs(np.mean(np.exp(1j * headings)))
        # Normalise: high variance → high circling
        heading_signal = min(heading_var * 2.0, 1.0)

        # ── Return ratio ──
        cum_tx = np.cumsum(txs)
        cum_ty = np.cumsum(tys)
        net_disp = math.sqrt(cum_tx[-1] ** 2 + cum_ty[-1] ** 2)
        total_path = np.sum(np.sqrt(txs ** 2 + tys ** 2))
        if total_path > 1e-8:
            return_ratio = 1.0 - min(net_disp / total_path, 1.0)
        else:
            return_ratio = 1.0  # stationary = trivially "returned"

        # ── Geometric mean ──
        product = rot_consistency * heading_signal * return_ratio
        if product <= 0:
            return 0.0
        circling = product ** (1.0 / 3.0)

        return float(np.clip(circling, 0.0, 1.0))

    def _compute_pattern_deviation(self) -> float:
        """Compute deviation from a fitted linear flight path.

        Fits a line to the cumulative translation path and measures
        the mean perpendicular distance from that line.  A drone
        following a linear search pattern will have deviation ≈ 0.
        """
        if len(self._tx_history) < 5:
            return 0.0

        txs = np.array(self._tx_history)
        tys = np.array(self._ty_history)

        # Cumulative path
        cum_x = np.cumsum(txs)
        cum_y = np.cumsum(tys)

        # Fit a line: y = mx + b
        n = len(cum_x)
        t = np.arange(n, dtype=float)
        if n < 3:
            return 0.0

        # Fit x(t) and y(t) independently
        cx = np.polyfit(t, cum_x, 1)
        cy = np.polyfit(t, cum_y, 1)

        # Predicted positions on the line
        x_fit = np.polyval(cx, t)
        y_fit = np.polyval(cy, t)

        # Mean deviation from fitted line
        deviations = np.sqrt((cum_x - x_fit) ** 2 + (cum_y - y_fit) ** 2)
        return float(np.mean(deviations))


def classify_flight_phase(features: EMIFeatures) -> FlightPhase:
    """Classify a single frame's EMI features into a flight phase.

    Rule-based classifier per spec:
        TRANSIT:     high speed, low rotation, linear path
        SCANNING:    moderate speed, sweeping rotation
        INTEREST:    decelerating, slight circling
        HOVERING:    near-zero speed, sustained
        CIRCLING:    moderate speed, high rotation
        DESCENDING:  positive descent rate
        APPROACHING: deceleration + descent

    Parameters
    ----------
    features : EMIFeatures

    Returns
    -------
    FlightPhase
    """
    speed = features.translational_speed
    rot = features.rotational_rate
    hover = features.hover_index
    circling = features.circling_index
    descent = features.descent_rate
    decel = features.deceleration

    # APPROACHING: decelerating + descending (highest priority check)
    if decel < -2.0 and descent > 0.05:
        return FlightPhase.APPROACHING

    # DESCENDING: significant positive descent rate
    if descent > 0.1:
        return FlightPhase.DESCENDING

    # HOVERING: near-zero speed sustained
    if hover > 0.7 and speed < 2.0:
        return FlightPhase.HOVERING

    # CIRCLING: moderate speed + high rotation + circling_index
    if circling > 0.3 and rot > 0.02:
        return FlightPhase.CIRCLING

    # INTEREST: decelerating with some circling
    if decel < -1.0 and circling > 0.1:
        return FlightPhase.INTEREST

    # SCANNING: moderate speed with rotation
    if 2.0 <= speed <= 10.0 and rot > 0.01:
        return FlightPhase.SCANNING

    # TRANSIT: default — fast, straight-line flight
    return FlightPhase.TRANSIT


def get_attention_score(phase: FlightPhase) -> float:
    """Get the attention score for a flight phase (0.1 → 1.0)."""
    return _FLIGHT_PHASE_ATTENTION.get(phase, 0.1)


def attention_based_multiplier(
    features: EMIFeatures, beta: float = 0.3,
) -> float:
    """Compute priority multiplier from attention score.

    Formula (from spec):
        multiplier = 1 + β · attention_score

    Parameters
    ----------
    features : EMIFeatures
    beta : float
        Tunable weight (default 0.3).

    Returns
    -------
    float ∈ [1.0, 1.3]
    """
    phase = classify_flight_phase(features)
    attention = get_attention_score(phase)
    return 1.0 + beta * attention


# ════════════════════════════════════════════════════════════════════════
# EMI → Priority Multiplier
# ════════════════════════════════════════════════════════════════════════

@dataclass
class EMIMultiplierConfig:
    """Weights and bounds for converting EMI features into a multiplier.

    The multiplier is computed as:

        raw = 1.0
            + w_hover   · hover_index
            + w_circle  · circling_index
            + w_descent · max(descent_rate, 0)  / descent_norm
            - w_speed   · min(speed / speed_norm, 1.0)

        multiplier = clip(raw, 1.0, max_multiplier)

    Default weights are calibrated so that:
    - Perfect hover  → ×1.4 boost
    - Circling orbit → ×1.3 boost
    - Descending     → ×1.2 boost per norm (stacks)
    - High speed     → boost ↓ toward ×1.0
    """
    w_hover: float = 0.40
    w_circle: float = 0.30
    w_descent: float = 0.20
    w_speed: float = 0.25

    descent_norm: float = 0.5   # descent_rate / norm = contribution
    speed_norm: float = 20.0    # speed above this → full speed penalty

    max_multiplier: float = 1.3
    min_multiplier: float = 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EMIMultiplierConfig":
        """Build from the ``ranker.emi`` config dict."""
        return cls(
            w_hover=config.get("w_hover", 0.40),
            w_circle=config.get("w_circle", 0.30),
            w_descent=config.get("w_descent", 0.20),
            w_speed=config.get("w_speed", 0.25),
            descent_norm=config.get("descent_norm", 0.5),
            speed_norm=config.get("speed_norm", 20.0),
            max_multiplier=config.get("max_multiplier", 1.3),
            min_multiplier=config.get("min_multiplier", 1.0),
        )


def emi_to_multiplier(
    features: EMIFeatures,
    cfg: Optional[EMIMultiplierConfig] = None,
) -> float:
    """Convert EMI features into a priority score multiplier.

    Parameters
    ----------
    features : EMIFeatures
        The 5-dimensional feature vector.
    cfg : EMIMultiplierConfig, optional
        Weight configuration.  Uses defaults if not provided.

    Returns
    -------
    float ∈ [cfg.min_multiplier, cfg.max_multiplier]
        Multiplicative factor for the event's priority score.

    Examples
    --------
    >>> f = EMIFeatures(translational_speed=0, hover_index=1.0)
    >>> emi_to_multiplier(f)   # perfect hover → ~1.4
    1.4

    >>> f = EMIFeatures(translational_speed=30, hover_index=0.0)
    >>> emi_to_multiplier(f)   # cruising → ~1.0
    1.0
    """
    if cfg is None:
        cfg = EMIMultiplierConfig()

    # Normalised components
    hover_contrib = cfg.w_hover * features.hover_index
    circle_contrib = cfg.w_circle * features.circling_index

    # Descent: only positive descent (going down) contributes
    descent_normed = max(features.descent_rate, 0.0) / max(cfg.descent_norm, 1e-8)
    descent_contrib = cfg.w_descent * min(descent_normed, 1.0)

    # Speed penalty: high speed reduces the multiplier
    speed_normed = min(features.translational_speed / max(cfg.speed_norm, 1e-8), 1.0)
    speed_penalty = cfg.w_speed * speed_normed

    raw = 1.0 + hover_contrib + circle_contrib + descent_contrib - speed_penalty

    return float(np.clip(raw, cfg.min_multiplier, cfg.max_multiplier))


def aggregate_emi_multiplier(
    features_list: Sequence[EMIFeatures],
    bins: Sequence[int],
    n_bins: int,
    bin_resolution: float = 0.5,
    cfg: Optional[EMIMultiplierConfig] = None,
) -> List[float]:
    """Compute per-bin EMI multipliers from a sequence of frame-level features.

    Each temporal bin gets the mean EMI multiplier of all features that
    fall into that bin.

    Parameters
    ----------
    features_list : sequence of EMIFeatures
        One feature per frame-pair (N-1 for N frames).
    bins : sequence of int
        Bin index for each feature (parallel to features_list).
    n_bins : int
        Total number of temporal bins.
    bin_resolution : float
        Seconds per bin.
    cfg : EMIMultiplierConfig, optional

    Returns
    -------
    list[float]
        Per-bin multipliers (1.0 for bins with no data).
    """
    if cfg is None:
        cfg = EMIMultiplierConfig()

    bin_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins

    for feat, b in zip(features_list, bins):
        if 0 <= b < n_bins:
            mult = emi_to_multiplier(feat, cfg)
            bin_sums[b] += mult
            bin_counts[b] += 1

    result = []
    for s, c in zip(bin_sums, bin_counts):
        if c > 0:
            result.append(s / c)
        else:
            result.append(1.0)  # no data → neutral multiplier

    return result


# ════════════════════════════════════════════════════════════════════════
# Synthetic Homography Generator (for testing / evaluation)
# ════════════════════════════════════════════════════════════════════════

def generate_synthetic_homographies(
    n_frames: int = 100,
    behavior: str = "hover",
    fps: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate synthetic homography sequences for testing.

    Parameters
    ----------
    n_frames : int
        Number of frames (produces n_frames-1 homographies).
    behavior : str
        One of: "hover", "cruise", "circle", "descend", "search".
    fps : float
    rng : numpy Generator, optional

    Returns
    -------
    list of (3,3) ndarrays
        Synthetic homographies.
    """
    rng = rng or np.random.default_rng(42)

    homographies = []
    for i in range(n_frames - 1):
        t = i / fps

        if behavior == "hover":
            # Nearly stationary with slight jitter
            tx = rng.normal(0, 0.3)
            ty = rng.normal(0, 0.3)
            rot = rng.normal(0, 0.001)
            scale = 1.0 + rng.normal(0, 0.0005)

        elif behavior == "cruise":
            # Fast, consistent translation
            tx = 15.0 + rng.normal(0, 1.0)
            ty = 3.0 + rng.normal(0, 0.5)
            rot = rng.normal(0, 0.005)
            scale = 1.0 + rng.normal(0, 0.001)

        elif behavior == "circle":
            # Orbit around a point: rotating heading + consistent rotation
            phase = 2 * math.pi * t / 8.0  # 8-second orbit
            radius = 5.0
            tx = radius * math.cos(phase) + rng.normal(0, 0.5)
            ty = radius * math.sin(phase) + rng.normal(0, 0.5)
            rot = 0.04 + rng.normal(0, 0.005)  # ~2.3°/frame
            scale = 1.0 + rng.normal(0, 0.001)

        elif behavior == "descend":
            # Hovering + descending (scale increasing)
            tx = rng.normal(0, 0.5)
            ty = rng.normal(0, 0.5)
            rot = rng.normal(0, 0.002)
            scale = 1.0 + 0.02 + rng.normal(0, 0.002)  # +2% per frame

        elif behavior == "search":
            # Slow, sweeping search pattern
            tx = 3.0 * math.sin(2 * math.pi * t / 15.0) + rng.normal(0, 0.5)
            ty = 2.0 * math.cos(2 * math.pi * t / 10.0) + rng.normal(0, 0.5)
            rot = 0.01 * math.sin(t) + rng.normal(0, 0.003)
            scale = 1.0 + rng.normal(0, 0.001)

        else:
            tx, ty, rot, scale = 0, 0, 0, 1.0

        # Build homography from affine components
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        H = np.array([
            [scale * cos_r, -scale * sin_r, tx],
            [scale * sin_r,  scale * cos_r, ty],
            [0,              0,              1],
        ], dtype=np.float64)

        # Add small perspective distortion (realistic)
        H[2, 0] += rng.normal(0, 1e-6)
        H[2, 1] += rng.normal(0, 1e-6)

        homographies.append(H)

    return homographies


# ════════════════════════════════════════════════════════════════════════
# CLI / Demo
# ════════════════════════════════════════════════════════════════════════

def demo():
    """Demonstrate EMI feature extraction and multiplier computation."""
    print(f"\n{'═' * 65}")
    print(f"  EMI — Ego-Motion Intelligence Demo")
    print(f"{'═' * 65}")

    behaviors = ["hover", "cruise", "circle", "descend", "search"]
    extractor = EMIExtractor(sigma_hover=5.0, circling_window=20, fps=5.0)
    cfg = EMIMultiplierConfig()

    for behavior in behaviors:
        extractor.reset()
        homographies = generate_synthetic_homographies(
            n_frames=60, behavior=behavior, fps=5.0,
        )

        features = extractor.extract_sequence(homographies)

        # Aggregate over the sequence
        avg_speed = np.mean([f.translational_speed for f in features])
        avg_rot = np.mean([f.rotational_rate for f in features])
        avg_hover = np.mean([f.hover_index for f in features])
        avg_circle = np.mean([f.circling_index for f in features])
        avg_descent = np.mean([f.descent_rate for f in features])

        # Compute multiplier from the average feature
        avg_feat = EMIFeatures(
            translational_speed=avg_speed,
            rotational_rate=avg_rot,
            hover_index=avg_hover,
            circling_index=avg_circle,
            descent_rate=avg_descent,
        )
        mult = emi_to_multiplier(avg_feat, cfg)

        print(f"\n  Behavior: {behavior.upper()}")
        print(f"    speed={avg_speed:6.2f}  rot={avg_rot:.4f}  "
              f"hover={avg_hover:.3f}  circle={avg_circle:.3f}  "
              f"descent={avg_descent:+.4f}")
        print(f"    → multiplier = {mult:.3f}×")

    # Show multiplier sensitivity
    print(f"\n  {'─' * 55}")
    print(f"  Multiplier sensitivity (hover_index sweep):")
    for h in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        f = EMIFeatures(hover_index=h)
        m = emi_to_multiplier(f)
        bar = "█" * int((m - 1.0) * 30)
        print(f"    hover={h:.1f}  → {m:.3f}×  {bar}")

    print(f"\n  Multiplier sensitivity (descent_rate sweep):")
    for d in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        f = EMIFeatures(descent_rate=d, hover_index=0.5)
        m = emi_to_multiplier(f)
        bar = "█" * int((m - 1.0) * 30)
        print(f"    descent={d:.1f}  → {m:.3f}×  {bar}")

    print(f"\n  ✅ EMI demo complete\n")


if __name__ == "__main__":
    demo()
