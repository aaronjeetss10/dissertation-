"""
evaluation/ego_motion_signal.py
=================================
Proof-of-concept: Can UAV flight trajectory patterns serve as an implicit
signal for event detection?

Hypothesis: Drone pilots unconsciously change flight behaviour (slow down,
circle, hover) when they spot something interesting.  The ego-motion pattern
that we normally *subtract* might itself carry information.

This experiment:
  1. Simulates realistic ego-motion + event co-occurrence patterns
  2. Extracts 5 ego-motion features per 2-second window
  3. Correlates features with event presence
  4. Trains logistic regression to predict event windows from ego alone
  5. Generates time-series, correlation heatmap, and ROC curve

Run:
    python evaluation/ego_motion_signal.py
"""

from __future__ import annotations

import json, sys, math, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Simulate Ego-Motion + Event Data
# ══════════════════════════════════════════════════════════════════════════════

def simulate_flight_data(n_windows: int = 120, fps: float = 5.0,
                          window_sec: float = 2.0, seed: int = 42):
    """Simulate realistic drone flight ego-motion and event occurrence.

    Models two flight regimes:
      - Survey mode (80%): steady forward flight, low events
      - Investigation mode (20%): deceleration, circling, hovering
        with higher event probability

    Returns per-window features and event labels.
    """
    rng = np.random.RandomState(seed)
    frames_per_window = int(fps * window_sec)

    # Generate continuous flight path
    # Base flight: forward at ~5 px/frame with gentle turns
    base_speed = 5.0  # px/frame global motion
    heading = 0.0     # radians

    windows = []

    for w in range(n_windows):
        # Decide flight mode
        # Simulate operator behaviour: 20% of windows are "investigation"
        # where pilot slows down and changes direction
        is_investigation = rng.random() < 0.20

        # Events are more likely during investigation (but not guaranteed)
        if is_investigation:
            event_prob = 0.55   # 55% chance of event during investigation
            speed_factor = rng.uniform(0.15, 0.5)   # slow down significantly
            heading_change = rng.uniform(-0.8, 0.8)  # change direction
            stability = rng.uniform(0.3, 1.5)        # moderate jitter (hovering)
        else:
            event_prob = 0.08   # 8% chance during normal survey
            speed_factor = rng.uniform(0.8, 1.2)     # steady speed
            heading_change = rng.uniform(-0.1, 0.1)  # gentle turns
            stability = rng.uniform(0.5, 2.5)        # more variance from wind

        has_event = rng.random() < event_prob
        event_criticality = rng.uniform(1.5, 3.5) if has_event else 0.0

        # Generate per-frame ego-motion vectors for this window
        ego_vectors = []
        window_speed = base_speed * speed_factor
        heading += heading_change

        for f in range(frames_per_window):
            # Add within-window dynamics
            frame_speed = window_speed + rng.normal(0, stability)
            frame_heading = heading + rng.normal(0, 0.05)
            dx = frame_speed * math.cos(frame_heading) + rng.normal(0, 0.3)
            dy = frame_speed * math.sin(frame_heading) + rng.normal(0, 0.3)
            ego_vectors.append((dx, dy))

        ego_vectors = np.array(ego_vectors)

        # Compute ego-motion features
        speeds = np.sqrt(ego_vectors[:, 0]**2 + ego_vectors[:, 1]**2)
        angles = np.arctan2(ego_vectors[:, 1], ego_vectors[:, 0])

        mean_ego_speed = float(np.mean(speeds))
        ego_speed_change = float(speeds[-1] - speeds[0])  # deceleration < 0
        ego_direction_change = float(np.mean(np.abs(np.diff(angles))))
        ego_stability = float(np.var(speeds))  # low = hovering
        # Altitude proxy: detection size (smaller = higher altitude)
        # Simulate: investigation mode → pilot descends → larger detections
        if is_investigation:
            ego_altitude_proxy = rng.uniform(25, 50)  # larger = lower
        else:
            ego_altitude_proxy = rng.uniform(12, 25)  # smaller = higher

        windows.append({
            "window_idx": w,
            "timestamp": w * window_sec,
            "mean_ego_speed": round(mean_ego_speed, 3),
            "ego_speed_change": round(ego_speed_change, 3),
            "ego_direction_change": round(ego_direction_change, 4),
            "ego_stability": round(ego_stability, 4),
            "ego_altitude_proxy": round(ego_altitude_proxy, 2),
            "has_event": int(has_event),
            "event_criticality": round(event_criticality, 3),
            "is_investigation": int(is_investigation),
        })

    return windows


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "mean_ego_speed",
    "ego_speed_change",
    "ego_direction_change",
    "ego_stability",
    "ego_altitude_proxy",
]

FEATURE_LABELS = [
    "Mean Ego Speed\n(px/frame)",
    "Speed Change\n(deceleration)",
    "Direction Change\n(angular vel.)",
    "Speed Variance\n(stability)",
    "Altitude Proxy\n(det. size)",
]


def analyse_correlations(windows):
    """Compute point-biserial correlations between ego features and events."""
    from scipy import stats

    X = np.array([[w[f] for f in FEATURE_NAMES] for w in windows])
    y = np.array([w["has_event"] for w in windows])

    correlations = {}
    for i, feat_name in enumerate(FEATURE_NAMES):
        r, p = stats.pointbiserialr(y, X[:, i])
        correlations[feat_name] = {
            "r": round(float(r), 4),
            "p": round(float(p), 6),
            "significant": bool(p < 0.05),
        }
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {feat_name:<25} r = {r:+.3f}  p = {p:.4f}  {sig}")

    return correlations, X, y


def train_ego_classifier(X, y):
    """Train logistic regression to predict event windows from ego features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve, classification_report

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, random_state=42, max_iter=500)

    # 5-fold cross-validated predictions for unbiased ROC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = cross_val_predict(clf, X_s, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, y_probs)
    fpr, tpr, thresholds = roc_curve(y, y_probs)

    # Fit final model for coefficients
    clf.fit(X_s, y)
    coefs = dict(zip(FEATURE_NAMES, [round(float(c), 4) for c in clf.coef_[0]]))

    print(f"\n    Logistic Regression AUC: {auc:.3f}")
    print(f"    Coefficients: {coefs}")

    return auc, fpr, tpr, y_probs, coefs


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(windows, correlations, auc, fpr, tpr, y_probs, X, y):
    """Three-panel publication figure."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    timestamps = [w["timestamp"] for w in windows]
    speeds = [w["mean_ego_speed"] for w in windows]
    events = [w["has_event"] for w in windows]
    event_times = [t for t, e in zip(timestamps, events) if e]
    event_crits = [w["event_criticality"] for w in windows if w["has_event"]]

    # ── Panel 1: Time series — ego speed with event markers ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(timestamps, speeds, color="#2c3e50", linewidth=1.2, alpha=0.8,
             label="Ego Speed")
    # Smooth
    kernel = np.ones(5) / 5
    smoothed = np.convolve(speeds, kernel, mode="same")
    ax1.plot(timestamps, smoothed, color="#3498db", linewidth=2,
             label="Smoothed (5-pt)")

    for et, ec in zip(event_times, event_crits):
        color = "#e74c3c" if ec > 2.5 else "#f39c12" if ec > 1.5 else "#95a5a6"
        ax1.axvline(x=et, color=color, alpha=0.4, linewidth=1.5)
    # Dummy entries for legend
    ax1.axvline(x=-10, color="#e74c3c", alpha=0.6, linewidth=2,
                label="Critical event")
    ax1.axvline(x=-10, color="#f39c12", alpha=0.6, linewidth=2,
                label="High event")

    ax1.set_xlabel("Time (seconds)", fontsize=11)
    ax1.set_ylabel("Ego Motion Speed (px/frame)", fontsize=11)
    ax1.set_title("Ego-Motion Speed vs Detected Events\n"
                 "Vertical lines = detected events (red = critical)",
                 fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(0, max(timestamps))

    # ── Panel 2: Correlation heatmap ──
    ax2 = fig.add_subplot(gs[0, 2])
    feat_labels_short = ["Speed", "Δ Speed", "Δ Direction", "Stability", "Alt. Proxy"]
    r_values = [correlations[f]["r"] for f in FEATURE_NAMES]
    p_values = [correlations[f]["p"] for f in FEATURE_NAMES]

    colors = ["#27ae60" if abs(r) > 0.15 else "#f39c12" if abs(r) > 0.08
              else "#e74c3c" for r in r_values]
    bars = ax2.barh(feat_labels_short, r_values, color=colors, alpha=0.85)
    for bar, r, p in zip(bars, r_values, p_values):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax2.text(bar.get_width() + 0.01 * np.sign(bar.get_width()),
                bar.get_y() + bar.get_height() / 2,
                f"r={r:+.3f}{sig}", va="center", fontsize=9, fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Point-Biserial Correlation with Event Presence", fontsize=10)
    ax2.set_title("Feature–Event Correlations", fontsize=12, fontweight="bold")

    # ── Panel 3: ROC curve ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(fpr, tpr, color="#2c3e50", linewidth=2.5,
             label=f"Ego-Motion LR (AUC = {auc:.3f})")
    ax3.plot([0, 1], [0, 1], "--", color="#bdc3c7", linewidth=1,
             label="Random (AUC = 0.500)")
    ax3.fill_between(fpr, tpr, alpha=0.1, color="#2c3e50")
    ax3.set_xlabel("False Positive Rate", fontsize=11)
    ax3.set_ylabel("True Positive Rate", fontsize=11)
    ax3.set_title("ROC: Can Ego-Motion Predict Events?", fontsize=12,
                 fontweight="bold")
    ax3.legend(fontsize=10, loc="lower right")

    # ── Panel 4: Speed distribution: event vs no-event ──
    ax4 = fig.add_subplot(gs[1, 1])
    speeds_event = [w["mean_ego_speed"] for w in windows if w["has_event"]]
    speeds_no = [w["mean_ego_speed"] for w in windows if not w["has_event"]]
    bins = np.linspace(0, 8, 25)
    ax4.hist(speeds_no, bins=bins, alpha=0.6, color="#3498db", density=True,
             label=f"No Event (n={len(speeds_no)})")
    ax4.hist(speeds_event, bins=bins, alpha=0.6, color="#e74c3c", density=True,
             label=f"Event (n={len(speeds_event)})")
    ax4.set_xlabel("Mean Ego Speed (px/frame)", fontsize=11)
    ax4.set_ylabel("Density", fontsize=11)
    ax4.set_title("Speed Distribution: Event vs Non-Event Windows",
                 fontsize=12, fontweight="bold")
    ax4.legend(fontsize=10)

    # ── Panel 5: Direction change distribution ──
    ax5 = fig.add_subplot(gs[1, 2])
    dir_event = [w["ego_direction_change"] for w in windows if w["has_event"]]
    dir_no = [w["ego_direction_change"] for w in windows if not w["has_event"]]
    bins2 = np.linspace(0, 0.5, 25)
    ax5.hist(dir_no, bins=bins2, alpha=0.6, color="#3498db", density=True,
             label=f"No Event (n={len(dir_no)})")
    ax5.hist(dir_event, bins=bins2, alpha=0.6, color="#e74c3c", density=True,
             label=f"Event (n={len(dir_event)})")
    ax5.set_xlabel("Direction Change Rate (rad/frame)", fontsize=11)
    ax5.set_ylabel("Density", fontsize=11)
    ax5.set_title("Direction Change: Event vs Non-Event",
                 fontsize=12, fontweight="bold")
    ax5.legend(fontsize=10)

    plt.suptitle("Ego-Motion as Implicit Event Signal — Proof of Concept",
                fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(FIGURES_DIR / "ego_motion_signal.png", dpi=250,
               bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ ego_motion_signal.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print("  Ego-Motion as Implicit Event Signal — Proof of Concept")
    print("═" * 70)

    # Generate data
    print("\n  [1/4] Simulating flight data (120 windows × 2s)...")
    windows = simulate_flight_data(n_windows=120, seed=42)
    n_events = sum(w["has_event"] for w in windows)
    n_invest = sum(w["is_investigation"] for w in windows)
    print(f"    {len(windows)} windows, {n_events} with events, "
          f"{n_invest} investigation mode")

    # Correlations
    print("\n  [2/4] Computing feature–event correlations...")
    correlations, X, y = analyse_correlations(windows)

    # Classifier
    print("\n  [3/4] Training ego-motion event predictor...")
    auc, fpr, tpr, y_probs, coefs = train_ego_classifier(X, y)

    # Figure
    print("\n  [4/4] Generating figures...")
    plot_results(windows, correlations, auc, fpr, tpr, y_probs, X, y)

    # Key finding
    strongest_feat = max(correlations.items(),
                        key=lambda x: abs(x[1]["r"]))
    feat_name, feat_stats = strongest_feat

    print("\n" + "═" * 70)
    print("  KEY FINDING")
    print("═" * 70)

    if auc > 0.65:
        print(f"\n  Ego-motion features achieve {auc:.1%} AUC for predicting")
        print(f"  event-containing windows, with {feat_name} showing the")
        print(f"  strongest correlation (r={feat_stats['r']:+.3f}, "
              f"p={feat_stats['p']:.4f}).")
        print(f"\n  This suggests UAV flight patterns carry implicit information")
        print(f"  about operator attention that could supplement computer")
        print(f"  vision-based event detection.")
    else:
        print(f"\n  Ego-motion features achieve {auc:.1%} AUC — a {'moderate' if auc > 0.55 else 'weak'}")
        print(f"  signal.  {feat_name} shows the strongest correlation")
        print(f"  (r={feat_stats['r']:+.3f}, p={feat_stats['p']:.4f}).")
        print(f"\n  The signal is {'statistically significant but modest' if feat_stats['significant'] else 'not statistically significant'}.")

    # Honest context
    print(f"\n  ⚠ Important context:")
    print(f"    This is simulated data modelling expected pilot behaviour.")
    print(f"    VisDrone captures systematic survey flights without operator")
    print(f"    reaction; the signal is expected to be stronger on real")
    print(f"    mission footage where pilots actively investigate.")
    print(f"    Validation on real SAR flight logs is needed.")

    # Save
    results = {
        "n_windows": len(windows),
        "n_events": n_events,
        "n_investigation_windows": n_invest,
        "event_rate": round(n_events / len(windows), 4),
        "auc": round(auc, 4),
        "correlations": correlations,
        "classifier_coefficients": coefs,
        "strongest_feature": feat_name,
        "strongest_r": feat_stats["r"],
        "strongest_p": feat_stats["p"],
        "note": ("Simulated data modelling expected pilot behaviour. "
                 "VisDrone captures systematic survey flights without "
                 "operator reaction; signal expected stronger on real "
                 "mission footage."),
    }
    with open(RESULTS_DIR / "ego_motion_signal.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to ego_motion_signal.json")
    print("  ✓ Done!")


if __name__ == "__main__":
    main()
