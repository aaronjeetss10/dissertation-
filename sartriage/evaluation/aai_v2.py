"""
evaluation/aai_v2.py
=======================
Adaptive Accuracy Integration v2 (AAI-v2)

Replaces the v1 hard sigmoid crossover with a small MLP meta-classifier
that outputs a **continuous weighting vector** [w_pixel, w_traj] deciding
how much to trust the Pixel stream (MViTv2-S) vs the Trajectory stream
(TrajMAE/TMS).

Architecture
------------
    Input:  (person_pixel_size, detection_confidence, motion_magnitude)
               │
    ┌──────────▼──────────┐
    │  BatchNorm1d(3)     │  Normalise inputs (no manual scaling needed)
    └──────────┬──────────┘
    ┌──────────▼──────────┐
    │  Linear(3 → 32)     │  Hidden layer 1
    │  GELU + Dropout(0.1)│
    └──────────┬──────────┘
    ┌──────────▼──────────┐
    │  Linear(32 → 16)    │  Hidden layer 2
    │  GELU + Dropout(0.1)│
    └──────────┬──────────┘
    ┌──────────▼──────────┐
    │  Linear(16 → 2)     │  Output logits
    │  Softmax            │  → [w_pixel, w_traj]  ∈ [0,1], sum=1
    └─────────────────────┘

Total parameters: 3×32 + 32 + 32×16 + 16 + 16×2 + 2 = 706

Training
--------
The MLP is trained on synthetic data generated from the existing AAI-v1
empirical anchors (MViTv2-S confidence and TMS accuracy curves) plus
realistic noise.  The training signal is the *soft oracle weight*:

    w*_pixel = mvit_acc(size) / (mvit_acc(size) + tms_acc(size))

This means the MLP learns to do proportional allocation, not binary
switching.  At 20px where MViTv2 is terrible, w_pixel ≈ 0.30.
At 100px where MViTv2 dominates, w_pixel ≈ 0.57.

Run:
    python evaluation/aai_v2.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ════════════════════════════════════════════════════════════════════════
# 1.  Empirical data anchors (from AAI-v1)
# ════════════════════════════════════════════════════════════════════════

# Real MViTv2-S mean softmax confidence per size bucket
_MVIT_ANCHORS = {
    10: 0.32,  15: 0.36,  20: 0.40,  30: 0.47,
    40: 0.51,  50: 0.55,  60: 0.59,  80: 0.65,
    100: 0.81, 150: 0.88,
}

_TMS_MOVEMENT_ACC = 0.92
_TMS_STATIONARY_ACC = 0.06
_SAR_MOVEMENT_RATIO = 0.65
_SAR_STATIONARY_RATIO = 0.35


def _conf_to_accuracy(conf: float) -> float:
    """Map softmax confidence to expected top-1 accuracy."""
    return float(np.clip(1.10 * conf - 0.08, 0.05, 0.95))


def _interpolate_mvit_acc(px: int) -> float:
    """Log-linear interpolation of MViTv2-S accuracy from anchor points."""
    anchors = sorted(_MVIT_ANCHORS.items())
    log_px = math.log(max(px, 1))
    for i in range(len(anchors) - 1):
        s0, c0 = anchors[i]
        s1, c1 = anchors[i + 1]
        if s0 <= px <= s1:
            t = (log_px - math.log(s0)) / (math.log(s1) - math.log(s0))
            conf = c0 + t * (c1 - c0)
            return _conf_to_accuracy(conf)
    if px < anchors[0][0]:
        return _conf_to_accuracy(anchors[0][1] * 0.85)
    return _conf_to_accuracy(anchors[-1][1])


def _tms_overall_accuracy() -> float:
    return _TMS_MOVEMENT_ACC * _SAR_MOVEMENT_RATIO + _TMS_STATIONARY_ACC * _SAR_STATIONARY_RATIO


# ════════════════════════════════════════════════════════════════════════
# 2.  AAI-v2 MLP Meta-Classifier
# ════════════════════════════════════════════════════════════════════════

class AAIv2MetaClassifier(nn.Module):
    """Small MLP that outputs stream weighting from detection context.

    Input features (3-dim):
        - person_pixel_size : float   (bbox diagonal in pixels)
        - detection_confidence : float (YOLO / detector confidence)
        - motion_magnitude : float    (optical flow magnitude or proxy)

    Output (2-dim, softmax):
        - w_pixel : float    (weight for MViTv2-S / pixel stream)
        - w_traj  : float    (weight for TrajMAE / trajectory stream)
        - w_pixel + w_traj = 1.0

    Parameters
    ----------
    hidden_dims : tuple
        Hidden layer sizes.  Default (32, 16) → 706 params total.
    dropout : float
        Dropout probability between hidden layers.
    temperature : float
        Softmax temperature.  Lower → sharper weighting.
        Higher → more blended.  Default 1.0.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        # Input normalisation (learned during training)
        self.input_bn = nn.BatchNorm1d(3)

        # Build MLP layers
        layers = []
        in_dim = 3
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output: 2 logits → softmax → [w_pixel, w_traj]
        layers.append(nn.Linear(in_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3) tensor
            [person_pixel_size, detection_confidence, motion_magnitude]

        Returns
        -------
        (B, 2) tensor
            [w_pixel, w_traj] summing to 1.0 per sample.
        """
        x = self.input_bn(x)
        logits = self.mlp(x)
        return F.softmax(logits / self.temperature, dim=-1)

    def get_weights(
        self,
        person_size: float,
        det_conf: float,
        motion_mag: float,
    ) -> Tuple[float, float]:
        """Convenience: single-sample inference → (w_pixel, w_traj).

        Parameters
        ----------
        person_size : float
            Bounding-box diagonal in pixels.
        det_conf : float
            Detection confidence ∈ [0, 1].
        motion_mag : float
            Motion magnitude (optical flow or displacement).

        Returns
        -------
        (w_pixel, w_traj) : tuple of floats
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(
                [[person_size, det_conf, motion_mag]],
                dtype=torch.float32,
            ).to(next(self.parameters()).device)
            w = self(x)
            return float(w[0, 0].cpu()), float(w[0, 1].cpu())

    def fuse_scores(
        self,
        person_size: float,
        det_conf: float,
        motion_mag: float,
        pixel_score: float,
        traj_score: float,
    ) -> Tuple[float, float, float, str]:
        """Fuse pixel and trajectory stream scores using learned weights.

        Returns
        -------
        (fused_score, w_pixel, w_traj, dominant_stream) : tuple
        """
        w_pixel, w_traj = self.get_weights(person_size, det_conf, motion_mag)
        fused = w_pixel * pixel_score + w_traj * traj_score
        dominant = "pixel" if w_pixel > w_traj else "trajectory"
        return fused, w_pixel, w_traj, dominant

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════════════════
# 3.  Training data generation
# ════════════════════════════════════════════════════════════════════════

def generate_training_data(
    n_samples: int = 10_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data from AAI-v1 empirical curves.

    For each sample:
    1. Draw a random person_pixel_size ∈ [5, 200]
    2. Draw detection_confidence correlated with size (larger → higher conf)
    3. Draw motion_magnitude from a realistic distribution
    4. Compute oracle soft labels: w*_pixel = mvit_acc / (mvit_acc + tms_acc)

    The oracle weights are *soft* — the MLP learns proportional allocation
    rather than binary switching.

    Parameters
    ----------
    n_samples : int
    seed : int

    Returns
    -------
    X : (N, 3) ndarray
        [person_pixel_size, detection_confidence, motion_magnitude]
    y : (N, 2) ndarray
        [w*_pixel, w*_traj] soft labels (sum to 1)
    """
    rng = np.random.default_rng(seed)

    # ── Person pixel sizes: log-uniform for good coverage ──
    # SAR footage is heavily skewed small, so we over-sample small sizes
    sizes = np.exp(rng.uniform(np.log(5), np.log(200), n_samples))

    # ── Detection confidence: correlated with size + noise ──
    # Larger objects → higher YOLO confidence
    base_conf = 0.3 + 0.5 * (1.0 - np.exp(-sizes / 60.0))
    det_confs = base_conf + rng.normal(0, 0.08, n_samples)
    det_confs = np.clip(det_confs, 0.10, 0.99)

    # ── Motion magnitude: mixture of still and moving ──
    # ~35% near-zero (stationary), ~65% moderate-to-high (moving in SAR)
    is_moving = rng.random(n_samples) < _SAR_MOVEMENT_RATIO
    motion_mag = np.where(
        is_moving,
        rng.lognormal(mean=2.0, sigma=0.8, size=n_samples),  # moving
        rng.exponential(scale=0.5, size=n_samples),            # stationary
    )
    motion_mag = np.clip(motion_mag, 0.0, 100.0)

    # ── Compute oracle soft weights ──
    tms_acc_base = _tms_overall_accuracy()

    w_pixel = np.zeros(n_samples)
    w_traj = np.zeros(n_samples)

    for i in range(n_samples):
        px = int(np.clip(sizes[i], 5, 200))
        mvit_acc = _interpolate_mvit_acc(px)

        # Modulate TMS accuracy by motion:
        # TMS is great at movement, terrible at stationary
        if is_moving[i]:
            tms_acc_i = _TMS_MOVEMENT_ACC
        else:
            tms_acc_i = _TMS_STATIONARY_ACC

        # Modulate MViTv2 accuracy by detection confidence:
        # Low det_conf → even worse pixel classification
        conf_factor = np.clip(det_confs[i] / 0.7, 0.3, 1.2)
        mvit_acc_adjusted = mvit_acc * conf_factor

        # Soft oracle: proportional to relative accuracy
        total = mvit_acc_adjusted + tms_acc_i + 1e-8
        w_pixel[i] = mvit_acc_adjusted / total
        w_traj[i] = tms_acc_i / total

    # Add small noise to labels to prevent overconfident training
    label_noise = rng.normal(0, 0.03, (n_samples, 2))
    y = np.column_stack([w_pixel, w_traj]) + label_noise
    y = np.clip(y, 0.01, 0.99)
    # Re-normalise to sum to 1
    y = y / y.sum(axis=1, keepdims=True)

    X = np.column_stack([sizes, det_confs, motion_mag])
    return X.astype(np.float32), y.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# 4.  Training loop
# ════════════════════════════════════════════════════════════════════════

def train_aai_v2(
    n_samples: int = 10_000,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 1.0,
    seed: int = 42,
) -> Tuple[AAIv2MetaClassifier, Dict]:
    """Train the AAI-v2 meta-classifier.

    Parameters
    ----------
    n_samples : int
        Number of synthetic training samples.
    epochs : int
        Training epochs.
    batch_size : int
    lr : float
        Learning rate.
    temperature : float
        Softmax temperature for the output layer.
    seed : int

    Returns
    -------
    model : AAIv2MetaClassifier
    metrics : dict
        Training history and evaluation metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n  [1/4] Generating training data ({n_samples:,} samples)...")
    X, y = generate_training_data(n_samples, seed)

    # Train/val split (80/20)
    n_val = int(0.2 * n_samples)
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    print(f"    Train: {len(X_train):,}  Val: {len(X_val):,}")

    # ── Build model ──
    print(f"\n  [2/4] Building AAI-v2 MLP (τ={temperature})...")
    model = AAIv2MetaClassifier(
        hidden_dims=(32, 16), dropout=0.1, temperature=temperature,
    ).to(DEVICE)
    print(f"    Parameters: {model.param_count:,}")

    # ── Loss: KL divergence (soft labels → soft predictions) ──
    # KL(target || predicted) is the natural loss for distribution matching
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training ──
    print(f"\n  [3/4] Training ({epochs} epochs, lr={lr}, bs={batch_size})...")
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            # KL divergence loss: D_KL(y_true || y_pred)
            loss = F.kl_div(
                pred.log().clamp(min=-20),  # log predictions
                yb,                          # target distribution
                reduction="batchmean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(X_train)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = F.kl_div(
                    pred.log().clamp(min=-20), yb,
                    reduction="batchmean",
                )
                val_loss += loss.item() * xb.size(0)
                # MAE on w_pixel (interpretable metric)
                val_mae_sum += (pred[:, 0] - yb[:, 0]).abs().sum().item()

        val_loss /= len(X_val)
        val_mae = val_mae_sum / len(X_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"train_kl={train_loss:.4f}  val_kl={val_loss:.4f}  "
                  f"val_mae(w_pixel)={val_mae:.4f}")

    return model, history


# ════════════════════════════════════════════════════════════════════════
# 5.  Evaluation
# ════════════════════════════════════════════════════════════════════════

def evaluate_aai_v2(model: AAIv2MetaClassifier) -> Dict:
    """Evaluate AAI-v2 against v1 across the standard size buckets.

    Returns
    -------
    dict with per-bucket results + summary metrics.
    """
    print(f"\n  [4/4] Evaluating AAI-v2...")

    size_buckets = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150]
    tms_acc = _tms_overall_accuracy()

    # ── v1 crossover (from saved results) ──
    v1_crossover = 75.0  # from aai_results.json
    v1_steepness = 0.12

    print(f"\n  {'Size':>6}  {'w_pixel':>8}  {'w_traj':>7}  "
          f"{'v2 fused':>9}  {'v1 fused':>9}  {'Δ':>6}  {'MViTv2':>7}  {'TMS':>6}")
    print("  " + "─" * 65)

    results = []
    for px in size_buckets:
        mvit_acc = _interpolate_mvit_acc(px)

        # Typical detection context for this size
        det_conf = min(0.3 + 0.5 * (1 - math.exp(-px / 60)), 0.95)
        motion_mag = 5.0  # moderate motion

        # ── v2: MLP weighting ──
        w_pixel, w_traj = model.get_weights(px, det_conf, motion_mag)
        v2_fused = w_pixel * mvit_acc + w_traj * tms_acc

        # ── v1: sigmoid crossover ──
        v1_w_pixel = 1.0 / (1.0 + math.exp(-v1_steepness * (px - v1_crossover)))
        v1_fused = v1_w_pixel * mvit_acc + (1 - v1_w_pixel) * tms_acc

        delta = v2_fused - v1_fused

        results.append({
            "size_px": px,
            "w_pixel": round(w_pixel, 4),
            "w_traj": round(w_traj, 4),
            "mvit_acc": round(mvit_acc, 4),
            "tms_acc": round(tms_acc, 4),
            "v2_fused_acc": round(v2_fused, 4),
            "v1_fused_acc": round(v1_fused, 4),
            "delta": round(delta, 4),
        })

        print(f"  {px:4d}px  {w_pixel:7.3f}  {w_traj:6.3f}  "
              f"{v2_fused:8.1%}  {v1_fused:8.1%}  {delta:+5.1%}  "
              f"{mvit_acc:6.1%}  {tms_acc:5.1%}")

    # ── Motion sensitivity: stationary vs moving ──
    print(f"\n  Motion sensitivity (size=30px):")
    for motion, label in [(0.5, "stationary"), (5.0, "walking"), (15.0, "running")]:
        wp, wt = model.get_weights(30.0, 0.5, motion)
        print(f"    {label:<12} → w_pixel={wp:.3f}  w_traj={wt:.3f}")

    # ── Detection confidence sensitivity ──
    print(f"\n  Detection confidence sensitivity (size=40px, motion=5.0):")
    for conf in [0.15, 0.30, 0.50, 0.70, 0.90]:
        wp, wt = model.get_weights(40.0, conf, 5.0)
        print(f"    conf={conf:.2f} → w_pixel={wp:.3f}  w_traj={wt:.3f}")

    return {"per_bucket": results}


# ════════════════════════════════════════════════════════════════════════
# 6.  Figures
# ════════════════════════════════════════════════════════════════════════

def plot_aai_v2(model: AAIv2MetaClassifier, history: Dict):
    """Generate AAI-v2 publication figures."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel 1: Weight surface (size × motion) ──
    ax = axes[0, 0]
    sizes = np.linspace(5, 160, 100)
    motions = np.linspace(0, 30, 100)
    S, M = np.meshgrid(sizes, motions)
    W = np.zeros_like(S)

    model.eval()
    with torch.no_grad():
        for i in range(100):
            x_batch = torch.tensor(
                [[S[i, j], 0.6, M[i, j]] for j in range(100)],
                dtype=torch.float32,
            ).to(DEVICE)
            w_batch = model(x_batch)[:, 0].cpu().numpy()
            W[i, :] = w_batch

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "traj_pixel", ["#c0392b", "#f5f5f5", "#2c3e50"])
    im = ax.contourf(S, M, W, levels=np.linspace(0, 1, 50), cmap=cmap)
    ax.contour(S, M, W, levels=[0.5], colors=["#e67e22"], linewidths=2.5,
               linestyles=["--"])
    plt.colorbar(im, ax=ax, label="w_pixel (MViTv2 weight)", shrink=0.85)
    ax.set_xlabel("Person Size (px)")
    ax.set_ylabel("Motion Magnitude")
    ax.set_title("AAI-v2: Learned Weight Surface\n"
                 "(orange = 50% boundary)", fontweight="bold")

    # ── Panel 2: Weight curves by size ──
    ax = axes[0, 1]
    sizes_dense = np.linspace(5, 160, 200)
    for motion_val, color, label in [
        (0.5, "#c0392b", "Stationary (0.5)"),
        (5.0, "#e67e22", "Walking (5.0)"),
        (15.0, "#2c3e50", "Running (15.0)"),
    ]:
        w_pixels = []
        model.eval()
        with torch.no_grad():
            for s in sizes_dense:
                x = torch.tensor([[s, 0.6, motion_val]], dtype=torch.float32).to(DEVICE)
                w = model(x)[0, 0].cpu().item()
                w_pixels.append(w)
        ax.plot(sizes_dense, w_pixels, color=color, linewidth=2, label=label)

    ax.axhline(0.5, color="gray", linewidth=1, linestyle=":", alpha=0.5)
    ax.set_xlabel("Person Size (px)")
    ax.set_ylabel("w_pixel (MViTv2 weight)")
    ax.set_title("AAI-v2: Weight vs Size by Motion Level", fontweight="bold")
    ax.legend(fontsize=9, title="Motion Magnitude", title_fontsize=9)
    ax.set_ylim(0, 1)

    # ── Panel 3: Fused accuracy comparison ──
    ax = axes[1, 0]
    size_buckets = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150]
    tms_acc = _tms_overall_accuracy()
    v1_crossover, v1_steep = 75.0, 0.12

    mvit_accs, v1_fused, v2_fused = [], [], []
    for px in size_buckets:
        mvit_a = _interpolate_mvit_acc(px)
        mvit_accs.append(mvit_a * 100)

        # v1
        w1 = 1.0 / (1.0 + math.exp(-v1_steep * (px - v1_crossover)))
        v1_fused.append((w1 * mvit_a + (1 - w1) * tms_acc) * 100)

        # v2
        det_conf = min(0.3 + 0.5 * (1 - math.exp(-px / 60)), 0.95)
        wp, wt = model.get_weights(px, det_conf, 5.0)
        v2_fused.append((wp * mvit_a + wt * tms_acc) * 100)

    ax.plot(size_buckets, mvit_accs, "o:", color="#2c3e50", linewidth=1.5,
            markersize=5, alpha=0.5, label="MViTv2-S only")
    ax.axhline(tms_acc * 100, color="#c0392b", linewidth=1.5, linestyle="--",
               alpha=0.5, label=f"TMS only ({tms_acc:.0%})")
    ax.plot(size_buckets, v1_fused, "s-", color="#95a5a6", linewidth=2,
            markersize=5, label="AAI-v1 (sigmoid)")
    ax.plot(size_buckets, v2_fused, "D-", color="#27ae60", linewidth=2.5,
            markersize=6, label="AAI-v2 (MLP)", zorder=5)

    ax.set_xlabel("Person Size (px)")
    ax.set_ylabel("Fused Accuracy (%)")
    ax.set_title("Fused Accuracy: v1 vs v2", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 100)

    # ── Panel 4: Training loss ──
    ax = axes[1, 1]
    ax.plot(history["train_loss"], color="#2c3e50", linewidth=1.5,
            label="Train KL", alpha=0.7)
    ax.plot(history["val_loss"], color="#e67e22", linewidth=2,
            label="Val KL")
    ax2 = ax.twinx()
    ax2.plot(history["val_mae"], color="#27ae60", linewidth=2,
             linestyle="--", label="Val MAE(w_pixel)")
    ax2.set_ylabel("MAE(w_pixel)", color="#27ae60")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Training History", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.suptitle("Adaptive Accuracy Integration v2 — MLP Meta-Classifier",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "aai_v2.png", dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  ✓ aai_v2.png")


# ════════════════════════════════════════════════════════════════════════
# 7.  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print("  Adaptive Accuracy Integration v2 (AAI-v2)")
    print("═" * 70)
    print(f"  Device: {DEVICE}")

    # ── Train ──
    model, history = train_aai_v2(
        n_samples=10_000,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        temperature=1.0,
    )

    # ── Evaluate ──
    eval_results = evaluate_aai_v2(model)

    # ── Compare v1 vs v2 ──
    print(f"\n  {'─' * 55}")
    print(f"  AAI-v2 vs v1 summary:")
    v1_acc = 0.8815  # from aai_results.json
    final_mae = history["val_mae"][-1]
    print(f"    v1 meta-classifier (LogReg): {v1_acc:.1%} binary accuracy")
    print(f"    v2 meta-classifier (MLP):    MAE(w_pixel) = {final_mae:.4f}")
    print(f"    v2 params:                   {model.param_count:,}")
    print(f"    v2 temperature:              {model.temperature}")

    # ── Save model + results ──
    model_path = RESULTS_DIR / "aai_v2_weights.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "temperature": model.temperature,
        "hidden_dims": (32, 16),
        "param_count": model.param_count,
    }, model_path)
    print(f"\n  ✓ Model → {model_path}")

    # Save results JSON
    results = {
        "model": {
            "type": "AAIv2MetaClassifier",
            "params": model.param_count,
            "hidden_dims": [32, 16],
            "temperature": model.temperature,
            "inputs": ["person_pixel_size", "detection_confidence", "motion_magnitude"],
            "outputs": ["w_pixel", "w_traj"],
        },
        "training": {
            "n_samples": 10_000,
            "epochs": 100,
            "final_train_kl": round(history["train_loss"][-1], 6),
            "final_val_kl": round(history["val_loss"][-1], 6),
            "final_val_mae_w_pixel": round(final_mae, 6),
        },
        "evaluation": eval_results,
        "v1_comparison": {
            "v1_binary_accuracy": v1_acc,
            "v2_val_mae": round(final_mae, 6),
            "improvement": "v2 outputs continuous weights (soft allocation) "
                           "vs v1 binary switching",
        },
    }
    results_path = RESULTS_DIR / "aai_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results → {results_path}")

    # ── Plot ──
    print(f"\n  Generating figures...")
    plot_aai_v2(model, history)

    # ── Final demo ──
    print(f"\n  {'─' * 55}")
    print(f"  Live demo — fusion at different scenarios:")
    demos = [
        (15, 0.3, 0.5, "Tiny, low-conf, still"),
        (15, 0.3, 10.0, "Tiny, low-conf, moving"),
        (50, 0.6, 5.0, "Medium, good conf"),
        (100, 0.9, 3.0, "Large, high conf"),
        (100, 0.9, 0.2, "Large, high conf, still"),
    ]
    for px, conf, motion, desc in demos:
        wp, wt = model.get_weights(px, conf, motion)
        print(f"    {desc:<30} → w_pixel={wp:.3f}  w_traj={wt:.3f}")

    print(f"\n  ✅ AAI-v2 complete")


if __name__ == "__main__":
    main()
