"""
evaluation/generate_tms_dataset.py
====================================
Standalone synthetic trajectory generator for TMS-12 feature extraction.

Produces 5,000 (configurable) synthetic centroid trajectories across 8
SAR-critical action classes, each annotated with the 12 TMS features
defined in ``streams/tms_classifier.py::TrajectoryFeatures``.

Key capabilities
-----------------
* **8 SAR classes**: falling, crawling, lying_down, running, waving,
  stumbling, collapsed, walking
* **Configurable Gaussian noise**: per-frame (cx, cy) perturbation scaled
  by ``noise_sigma`` (default 1.5 px, tuneable per-class)
* **Sinusoidal ego-motion drift**: simulates slow UAV camera pan via
  ``A·sin(2πf·t + φ)`` on both axes, configurable amplitude & frequency.
* **Falling model**: sudden V_y → 0 transition (impact frame) combined
  with aspect-ratio shift from tall (≈1.4) to wide (≈0.35), modelling
  freefall → ground-impact physics.

Output
------
``evaluation/results/tms_synthetic_dataset.csv``  — one row per trajectory
    Columns: label, trial, {12 TMS features}, n_frames, noise_sigma,
             ego_amplitude, ego_frequency

``evaluation/results/tms_synthetic_meta.json``    — generation metadata

Usage
-----
    python -m sartriage.evaluation.generate_tms_dataset
    python -m sartriage.evaluation.generate_tms_dataset --n_per_class 1000 --noise_sigma 2.5

Author: SARTriage dissertation tooling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Resolve project root so the script works standalone ─────────────────
_PROJECT = Path(__file__).resolve().parent.parent          # sartriage/
sys.path.insert(0, str(_PROJECT))

from streams.tms_classifier import TrajectoryFeatures  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ════════════════════════════════════════════════════════════════════════

class GeneratorConfig:
    """All tuneable knobs for the synthetic dataset."""

    def __init__(
        self,
        n_per_class: int = 625,
        n_frames: int = 16,
        fps: int = 5,
        frame_dims: Tuple[int, int] = (1080, 1920),
        noise_sigma: float = 1.5,
        ego_amplitude: float = 3.0,
        ego_frequency: float = 0.15,
        person_size_px: float = 20.0,
        seed: Optional[int] = 42,
    ):
        self.n_per_class = n_per_class
        self.n_frames = n_frames
        self.fps = fps
        self.frame_dims = frame_dims
        self.noise_sigma = noise_sigma
        self.ego_amplitude = ego_amplitude
        self.ego_frequency = ego_frequency
        self.person_size_px = person_size_px
        self.seed = seed

    def to_dict(self) -> dict:
        return {
            "n_per_class": self.n_per_class,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "frame_dims": list(self.frame_dims),
            "noise_sigma": self.noise_sigma,
            "ego_amplitude": self.ego_amplitude,
            "ego_frequency": self.ego_frequency,
            "person_size_px": self.person_size_px,
            "seed": self.seed,
        }


# ════════════════════════════════════════════════════════════════════════
# Ego-motion model: sinusoidal UAV camera drift
# ════════════════════════════════════════════════════════════════════════

def _ego_drift(
    t: float,
    amplitude: float,
    frequency: float,
    phase_x: float,
    phase_y: float,
) -> Tuple[float, float]:
    """Return (dx, dy) ego-motion offset for timestamp *t*.

    Models a slow sinusoidal camera pan — the dominant artefact in
    real UAV footage when the drone is in survey/loiter mode.

        ego_x(t) = A · sin(2πf·t + φ_x)
        ego_y(t) = A · sin(2πf·t + φ_y)
    """
    ego_x = amplitude * math.sin(2 * math.pi * frequency * t + phase_x)
    ego_y = amplitude * math.sin(2 * math.pi * frequency * t + phase_y)
    return ego_x, ego_y


# ════════════════════════════════════════════════════════════════════════
# Per-class trajectory physics models
# ════════════════════════════════════════════════════════════════════════

def _synth_falling(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Falling trajectory with sudden V_y → 0 impact + aspect shift.

    Physics model
    -------------
    Phase 1 (freefall, frames 0 .. impact_frame):
        V_y grows quadratically  (gravitational acceleration g ≈ 9.8 m/s²
        mapped into pixel-scale).  Lateral drift is small & random.
        Aspect starts tall (≈ 1.4) — upright body silhouette.

    Phase 2 (impact, single frame):
        V_y snaps to ≈ 0  (sudden deceleration).
        Aspect drops sharply toward wide (≈ 0.35).

    Phase 3 (post-impact, remaining frames):
        Near-stationary with minor settling.
        Aspect remains wide (prone on ground).
    """
    impact_frac = rng.uniform(0.50, 0.70)              # impact at 50-70% of clip
    impact_frame = int(n * impact_frac)

    if i < impact_frame:
        # ── Freefall phase ──
        # Quadratic vertical displacement (d = ½·g·t²)
        progress = i / max(impact_frame, 1)
        dy = 3.0 * (progress ** 1.8) * impact_frame     # accelerating downward
        dx = rng.normal(0, 1.5)
        # Aspect: tall → transitioning
        aspect = max(0.45, 1.4 - 0.7 * (progress ** 1.2))
    elif i == impact_frame:
        # ── Impact frame: sudden V_y → 0 ──
        dy = rng.normal(0, 0.4)                         # near-zero vertical
        dx = rng.normal(0, 0.3)
        aspect = 0.40 + rng.normal(0, 0.03)             # now wide (prone)
    else:
        # ── Post-impact: stationary, prone ──
        dy = rng.normal(0, 0.25)
        dx = rng.normal(0, 0.25)
        aspect = 0.35 + rng.normal(0, 0.02)

    return dx, dy, max(0.2, aspect)


def _synth_crawling(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Slow, mostly horizontal, prone posture."""
    dx = 1.5 + rng.normal(0, 0.3)
    dy = rng.normal(0, 0.4)
    aspect = 0.5 + rng.normal(0, 0.04)
    return dx, dy, max(0.25, aspect)


def _synth_lying_down(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Nearly stationary, wide bbox (prone)."""
    dx = rng.normal(0, 0.25)
    dy = rng.normal(0, 0.25)
    aspect = 0.4 + rng.normal(0, 0.03)
    return dx, dy, max(0.2, aspect)


def _synth_running(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Fast, consistent, horizontal, upright posture."""
    direction = rng.choice([-1, 1])
    dx = direction * (8.0 + rng.normal(0, 1.0))
    dy = rng.normal(0, 0.8)
    aspect = 1.5 + rng.normal(0, 0.08)
    return dx, dy, max(0.8, aspect)


def _synth_walking(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Moderate horizontal speed, upright posture, consistent heading."""
    direction = rng.choice([-1, 1])
    dx = direction * (3.5 + rng.normal(0, 0.6))
    dy = rng.normal(0, 0.5)
    aspect = 1.3 + rng.normal(0, 0.06)
    return dx, dy, max(0.7, aspect)


def _synth_waving(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Oscillating position (arm/body sway), stationary overall."""
    dx = 5.0 * math.sin(i * 0.8) + rng.normal(0, 0.5)
    dy = rng.normal(0, 0.4)
    aspect = 1.2 + rng.normal(0, 0.05)
    return dx, dy, max(0.7, aspect)


def _synth_stumbling(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Erratic direction changes, decelerating, aspect transitioning."""
    speed = max(0.5, 5.0 - i * 0.3)
    angle = rng.uniform(-0.5, 0.5)
    dx = speed * math.cos(angle * i) + rng.normal(0, 2)
    dy = speed * math.sin(angle * i) + rng.normal(0, 2)
    aspect = max(0.5, 1.3 - i * 0.04)
    return dx, dy, aspect


def _synth_collapsed(
    i: int,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Moving → sudden stop + posture change (upright → prone)."""
    transition = n // 3
    if i < transition:
        # Active locomotion
        dx = 4.0 + rng.normal(0, 0.8)
        dy = rng.normal(0, 0.8)
        aspect = 1.3 + rng.normal(0, 0.05)
    else:
        # Post-collapse: stationary, prone
        decay = max(0.0, 1.0 - (i - transition) / max(n - transition, 1))
        dx = decay * rng.normal(0, 0.5)
        dy = decay * rng.normal(0, 0.5)
        aspect = 0.4 + rng.normal(0, 0.03)
    return dx, dy, max(0.2, aspect)


# Registry mapping class names to physics generators
_ACTION_GENERATORS = {
    "falling":    _synth_falling,
    "crawling":   _synth_crawling,
    "lying_down": _synth_lying_down,
    "running":    _synth_running,
    "walking":    _synth_walking,
    "waving":     _synth_waving,
    "stumbling":  _synth_stumbling,
    "collapsed":  _synth_collapsed,
}

SAR_CLASSES = list(_ACTION_GENERATORS.keys())


# ════════════════════════════════════════════════════════════════════════
# Core trajectory synthesiser
# ════════════════════════════════════════════════════════════════════════

def synthesise_trajectory(
    action: str,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[float, float]], List[float], List[float], List[float]]:
    """Generate a single synthetic trajectory for *action*.

    Returns
    -------
    centroids : list of (cx, cy) in pixel coordinates
    timestamps : list of float (seconds)
    aspects : list of float (bbox height / width)
    sizes : list of float (bbox max dimension in pixels)
    """
    h, w = cfg.frame_dims
    dt = 1.0 / cfg.fps
    gen_fn = _ACTION_GENERATORS[action]

    # Random starting position (avoid edges)
    cx = rng.uniform(150, w - 150)
    cy = rng.uniform(150, h - 150)

    # Random ego-motion phase offsets (per trajectory)
    phase_x = rng.uniform(0, 2 * math.pi)
    phase_y = rng.uniform(0, 2 * math.pi)

    centroids: List[Tuple[float, float]] = []
    timestamps: List[float] = []
    aspects: List[float] = []
    sizes: List[float] = []

    for i in range(cfg.n_frames):
        t = i * dt

        # 1. Action-specific displacement & aspect
        dx, dy, aspect = gen_fn(i, cfg.n_frames, dt, rng)

        # 2. Gaussian observation noise
        dx += rng.normal(0, cfg.noise_sigma)
        dy += rng.normal(0, cfg.noise_sigma)

        # 3. Sinusoidal ego-motion drift
        ego_x, ego_y = _ego_drift(
            t, cfg.ego_amplitude, cfg.ego_frequency, phase_x, phase_y
        )
        dx += ego_x
        dy += ego_y

        # 4. Update position (clip to frame bounds)
        cx = float(np.clip(cx + dx, 10, w - 10))
        cy = float(np.clip(cy + dy, 10, h - 10))

        centroids.append((cx, cy))
        timestamps.append(t)
        aspects.append(float(aspect))
        sizes.append(cfg.person_size_px)

    return centroids, timestamps, aspects, sizes


# ════════════════════════════════════════════════════════════════════════
# Dataset generation
# ════════════════════════════════════════════════════════════════════════

def generate_dataset(cfg: GeneratorConfig) -> List[Dict[str, Any]]:
    """Generate the full synthetic TMS-12 dataset.

    Returns a list of dicts, one per trajectory, containing:
        label, trial, {12 TMS features}, n_frames, noise_sigma, …
    """
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, Any]] = []

    total = cfg.n_per_class * len(SAR_CLASSES)
    generated = 0

    for action in SAR_CLASSES:
        action_ok = 0
        for trial in range(cfg.n_per_class):
            centroids, timestamps, aspects, sizes = synthesise_trajectory(
                action, cfg, rng
            )

            # Extract TMS-12 features via the production code path
            tf = TrajectoryFeatures(
                centroids, timestamps, aspects,
                cfg.frame_dims, sizes,
            )

            row: Dict[str, Any] = {
                "label": action,
                "trial": trial,
            }
            row.update(tf.features)                          # 12 TMS features
            row["n_frames"] = cfg.n_frames
            row["noise_sigma"] = cfg.noise_sigma
            row["ego_amplitude"] = cfg.ego_amplitude
            row["ego_frequency"] = cfg.ego_frequency
            rows.append(row)

            action_ok += 1
            generated += 1

        print(f"  ✓ {action:12s}  {action_ok:>5d} trajectories")

    print(f"\n  Total: {generated} trajectories across {len(SAR_CLASSES)} classes")
    return rows


# ════════════════════════════════════════════════════════════════════════
# I/O — CSV + JSON metadata
# ════════════════════════════════════════════════════════════════════════

def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write dataset to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  📄 Saved CSV  → {path}  ({len(rows)} rows)")


def save_metadata(cfg: GeneratorConfig, n_rows: int, elapsed: float, path: Path) -> None:
    """Write generation metadata to JSON."""
    meta = {
        "generator": "generate_tms_dataset.py",
        "classes": SAR_CLASSES,
        "n_classes": len(SAR_CLASSES),
        "n_per_class": cfg.n_per_class,
        "total_trajectories": n_rows,
        "feature_count": 12,
        "feature_names": [
            "net_displacement", "mean_speed", "speed_cv",
            "max_acceleration", "vertical_dominance",
            "direction_change_rate", "stationarity",
            "aspect_change", "speed_decay", "oscillation",
            "mean_aspect", "mean_size_norm",
        ],
        "noise_model": "Gaussian iid per-frame, sigma = {:.2f} px".format(
            cfg.noise_sigma
        ),
        "ego_motion_model": (
            "Sinusoidal drift: A·sin(2πf·t + φ), "
            f"A = {cfg.ego_amplitude:.1f} px, f = {cfg.ego_frequency:.2f} Hz, "
            "phase randomised per trajectory"
        ),
        "falling_model": (
            "Quadratic freefall (d ∝ t^1.8) until impact at 50-70% of clip, "
            "then sudden V_y → 0 with aspect shift from ~1.4 (tall/upright) "
            "to ~0.35 (wide/prone)"
        ),
        "config": cfg.to_dict(),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  📋 Saved meta → {path}")


# ════════════════════════════════════════════════════════════════════════
# Quick sanity report
# ════════════════════════════════════════════════════════════════════════

def print_report(rows: List[Dict[str, Any]]) -> None:
    """Print per-class feature statistics for a quick sanity check."""
    from collections import defaultdict

    feat_names = [
        "net_displacement", "mean_speed", "speed_cv",
        "max_acceleration", "vertical_dominance",
        "direction_change_rate", "stationarity",
        "aspect_change", "speed_decay", "oscillation",
        "mean_aspect", "mean_size_norm",
    ]

    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)

    print("\n" + "=" * 88)
    print("  Per-class feature means (sanity check)")
    print("=" * 88)

    header = f"  {'class':12s}"
    for fn in feat_names[:6]:
        header += f" {fn[:10]:>10s}"
    print(header)
    print("  " + "-" * 72)

    for cls in SAR_CLASSES:
        line = f"  {cls:12s}"
        for fn in feat_names[:6]:
            vals = [r[fn] for r in by_class[cls]]
            line += f" {np.mean(vals):10.4f}"
        print(line)

    print()
    header2 = f"  {'class':12s}"
    for fn in feat_names[6:]:
        header2 += f" {fn[:10]:>10s}"
    print(header2)
    print("  " + "-" * 72)

    for cls in SAR_CLASSES:
        line = f"  {cls:12s}"
        for fn in feat_names[6:]:
            vals = [r[fn] for r in by_class[cls]]
            line += f" {np.mean(vals):10.4f}"
        print(line)

    print("=" * 88)


# ════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TMS-12 trajectory dataset for SAR action classes."
    )
    parser.add_argument(
        "--n_per_class", type=int, default=625,
        help="Trajectories per class (default 625 → 5,000 total for 8 classes)",
    )
    parser.add_argument(
        "--n_frames", type=int, default=16,
        help="Frames per trajectory (default 16 = 3.2s at 5 fps)",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Target frame rate (default 5)",
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=1.5,
        help="Gaussian noise σ in pixels (default 1.5)",
    )
    parser.add_argument(
        "--ego_amplitude", type=float, default=3.0,
        help="Sinusoidal ego-motion amplitude in pixels (default 3.0)",
    )
    parser.add_argument(
        "--ego_frequency", type=float, default=0.15,
        help="Sinusoidal ego-motion frequency in Hz (default 0.15)",
    )
    parser.add_argument(
        "--person_size", type=float, default=20.0,
        help="Simulated person bbox size in pixels (default 20.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: evaluation/results/tms_synthetic_dataset.csv)",
    )
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_per_class=args.n_per_class,
        n_frames=args.n_frames,
        fps=args.fps,
        noise_sigma=args.noise_sigma,
        ego_amplitude=args.ego_amplitude,
        ego_frequency=args.ego_frequency,
        person_size_px=args.person_size,
        seed=args.seed,
    )

    total = cfg.n_per_class * len(SAR_CLASSES)
    print(f"\n{'─' * 60}")
    print(f"  TMS-12 Synthetic Dataset Generator")
    print(f"{'─' * 60}")
    print(f"  Classes:        {len(SAR_CLASSES)} — {', '.join(SAR_CLASSES)}")
    print(f"  Per class:      {cfg.n_per_class}")
    print(f"  Total:          {total}")
    print(f"  Frames/traj:    {cfg.n_frames}  ({cfg.n_frames / cfg.fps:.1f}s at {cfg.fps} fps)")
    print(f"  Noise σ:        {cfg.noise_sigma} px")
    print(f"  Ego-motion:     A={cfg.ego_amplitude} px, f={cfg.ego_frequency} Hz (sinusoidal)")
    print(f"  Person size:    {cfg.person_size_px} px")
    print(f"  Seed:           {cfg.seed}")
    print(f"{'─' * 60}\n")

    t0 = time.time()
    rows = generate_dataset(cfg)
    elapsed = time.time() - t0

    # Save outputs
    csv_path = Path(args.output) if args.output else RESULTS_DIR / "tms_synthetic_dataset.csv"
    meta_path = csv_path.with_name("tms_synthetic_meta.json")

    save_csv(rows, csv_path)
    save_metadata(cfg, len(rows), elapsed, meta_path)

    # Sanity report
    print_report(rows)

    print(f"\n  ⏱  Generated in {elapsed:.2f}s")
    print(f"  ✅ Done — {len(rows)} trajectories × 12 features\n")


if __name__ == "__main__":
    main()
