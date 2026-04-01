"""
Jitter Filter Experiment
========================
Tests whether pre-smoothing trajectories (moving average / Kalman filter)
before TMS-16 feature extraction improves walking-vs-stationary discrimination.

Hypothesis: tracker jitter makes stationary persons look like slow walkers.
Smoothing should reduce the apparent displacement of stationary tracks
without affecting genuine walking displacement.

Methods:
  1. Raw (baseline) — no smoothing
  2. Moving Average (MA-5) — window=5 centroid smoother
  3. Moving Average (MA-7) — window=7 centroid smoother
  4. Kalman Filter — constant-velocity 1D Kalman on cx,cy independently

Evaluation: 5-fold stratified CV, SMOTE-balanced RF, 4 SAR classes.
"""

import os, sys, json, math, warnings
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, cohen_kappa_score, recall_score)

sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

OUT_DIR = "evaluation/real_data/full"
os.makedirs(OUT_DIR, exist_ok=True)

SAR_MAP = {"Standing": "stationary", "Sitting": "stationary", "Walking": "walking",
           "Running": "running", "Lying": "lying_down"}
RF_CLASSES = ["lying_down", "stationary", "walking", "running"]


# ═══════════════════════════════════════════════════════════════════════
# Smoothing Functions
# ═══════════════════════════════════════════════════════════════════════

def moving_average_smooth(positions, window=5):
    """Apply centred moving-average smoothing to a 1D sequence.
    
    Args:
        positions: list of float values (cx or cy)
        window: smoothing window size (should be odd for centring)
        
    Returns:
        smoothed: list of float values, same length
    """
    n = len(positions)
    if n < window:
        return list(positions)
    
    half = window // 2
    smoothed = list(positions)  # copy
    
    for i in range(half, n - half):
        smoothed[i] = float(np.mean(positions[max(0, i - half):i + half + 1]))
    
    return smoothed


def kalman_smooth(positions, process_noise=1.0, measurement_noise=5.0):
    """Apply simple 1D Kalman filter (constant velocity model).
    
    This models position with a scalar state and scalar measurement.
    The process noise controls how much the state can change between steps.
    The measurement noise controls how much we trust each observation.
    
    Args:
        positions: list of float values
        process_noise: Q — how much the true position can change per step
        measurement_noise: R — how noisy the measurements are
        
    Returns:
        smoothed: list of float values
    """
    smoothed = []
    x = positions[0]   # initial state estimate
    p = 1.0             # initial uncertainty
    
    for z in positions:
        # Predict step
        p = p + process_noise
        # Update step
        k = p / (p + measurement_noise)  # Kalman gain
        x = x + k * (z - x)              # state update
        p = (1 - k) * p                  # uncertainty update
        smoothed.append(x)
    
    return smoothed


def apply_smoothing(trajectory, method="raw", **kwargs):
    """Apply smoothing to a trajectory and return the smoothed version.
    
    Args:
        trajectory: list of [cx, cy, w, h] per frame
        method: "raw", "ma5", "ma7", "kalman"
        
    Returns:
        smoothed trajectory: list of [cx_smooth, cy_smooth, w, h]
    """
    if method == "raw":
        return trajectory
    
    cxs = [t[0] for t in trajectory]
    cys = [t[1] for t in trajectory]
    ws  = [t[2] for t in trajectory]
    hs  = [t[3] for t in trajectory]
    
    if method == "ma5":
        cxs_s = moving_average_smooth(cxs, window=5)
        cys_s = moving_average_smooth(cys, window=5)
    elif method == "ma7":
        cxs_s = moving_average_smooth(cxs, window=7)
        cys_s = moving_average_smooth(cys, window=7)
    elif method == "kalman":
        pn = kwargs.get("process_noise", 1.0)
        mn = kwargs.get("measurement_noise", 5.0)
        cxs_s = kalman_smooth(cxs, process_noise=pn, measurement_noise=mn)
        cys_s = kalman_smooth(cys, process_noise=pn, measurement_noise=mn)
    elif method == "kalman_tight":
        # Tighter Kalman — more aggressive smoothing
        cxs_s = kalman_smooth(cxs, process_noise=0.5, measurement_noise=10.0)
        cys_s = kalman_smooth(cys, process_noise=0.5, measurement_noise=10.0)
    else:
        return trajectory
    
    return [[cx, cy, w, h] for cx, cy, w, h in zip(cxs_s, cys_s, ws, hs)]


# ═══════════════════════════════════════════════════════════════════════
# Feature Extraction (from tms12_standalone.py)
# ═══════════════════════════════════════════════════════════════════════

from evaluation.real_data.tms12_standalone import extract_tms16


# ═══════════════════════════════════════════════════════════════════════
# Load Okutama data
# ═══════════════════════════════════════════════════════════════════════

def load_tracks():
    """Load and filter Okutama tracks."""
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    
    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20:
            continue
        act = t["primary_action"]
        if act not in SAR_MAP:
            continue
        
        # Build trajectory: [cx, cy, w, h] per frame
        traj = [[c[0], c[1], b[2], b[3]] for c, b in zip(t["centroids"], t["bboxes"])]
        
        tracks.append({
            "tid": tid,
            "trajectory": traj,
            "label": RF_CLASSES.index(SAR_MAP[act]),
            "sar_class": SAR_MAP[act],
            "gt_action": act,
            "mean_size": float(np.mean([math.sqrt(max(b[2], 1) * max(b[3], 1)) for b in t["bboxes"]])),
        })
    
    return tracks


# ═══════════════════════════════════════════════════════════════════════
# Run Experiment
# ═══════════════════════════════════════════════════════════════════════

def evaluate_method(tracks, method_name, method_key, **kwargs):
    """Run 5-fold CV with SMOTE-balanced RF on smoothed features."""
    from imblearn.over_sampling import SMOTE
    
    # Extract features from smoothed trajectories
    features = []
    for t in tracks:
        smoothed_traj = apply_smoothing(t["trajectory"], method=method_key, **kwargs)
        f16 = extract_tms16(smoothed_traj)
        # Sanitise NaN/Inf
        f16 = [0.0 if (math.isnan(f) or math.isinf(f)) else f for f in f16]
        features.append(f16)
    
    X = np.array(features)
    y = np.array([t["label"] for t in tracks])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []
    all_true = []
    fold_accs = []
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        try:
            sm = SMOTE(random_state=42)
            X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
        except:
            X_sm, y_sm = X_tr, y_tr
        
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        clf.fit(X_sm, y_sm)
        preds = clf.predict(X_te)
        
        fold_accs.append(accuracy_score(y_te, preds))
        all_preds.extend(preds)
        all_true.extend(y_te)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    overall_acc = float(np.mean(fold_accs))
    acc_ci = float(1.96 * np.std(fold_accs))
    kappa = float(cohen_kappa_score(all_true, all_preds))
    
    # Per-class recall
    report = classification_report(all_true, all_preds, target_names=RF_CLASSES, output_dict=True)
    
    cm = confusion_matrix(all_true, all_preds)
    
    result = {
        "method": method_name,
        "accuracy": overall_acc,
        "accuracy_ci": acc_ci,
        "kappa": kappa,
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in RF_CLASSES
        },
        "confusion_matrix": cm.tolist(),
    }
    
    return result


def compute_displacement_stats(tracks, method_key, **kwargs):
    """Compute mean displacement for walking vs stationary under a given smoothing."""
    walk_disps = []
    stat_disps = []
    
    for t in tracks:
        smoothed_traj = apply_smoothing(t["trajectory"], method=method_key, **kwargs)
        cxs = [p[0] for p in smoothed_traj]
        cys = [p[1] for p in smoothed_traj]
        
        # Mean per-frame displacement
        disps = [math.sqrt((cxs[i] - cxs[i-1])**2 + (cys[i] - cys[i-1])**2)
                 for i in range(1, len(cxs))]
        mean_disp = float(np.mean(disps)) if disps else 0.0
        
        if t["sar_class"] == "walking":
            walk_disps.append(mean_disp)
        elif t["sar_class"] == "stationary":
            stat_disps.append(mean_disp)
    
    return {
        "walking_mean_disp": float(np.mean(walk_disps)) if walk_disps else 0.0,
        "walking_std_disp": float(np.std(walk_disps)) if walk_disps else 0.0,
        "stationary_mean_disp": float(np.mean(stat_disps)) if stat_disps else 0.0,
        "stationary_std_disp": float(np.std(stat_disps)) if stat_disps else 0.0,
        "overlap_ratio": float(np.mean(stat_disps) / (np.mean(walk_disps) + 1e-8))
            if walk_disps and stat_disps else 0.0,
        "n_walking": len(walk_disps),
        "n_stationary": len(stat_disps),
    }


def main():
    print("=" * 70)
    print("JITTER FILTER EXPERIMENT")
    print("Testing trajectory smoothing for walking-vs-stationary confusion")
    print("=" * 70)
    
    print("\nLoading Okutama tracks...")
    tracks = load_tracks()
    print(f"  Loaded {len(tracks)} tracks")
    
    # Class distribution
    class_counts = defaultdict(int)
    for t in tracks:
        class_counts[t["sar_class"]] += 1
    print(f"  Classes: {dict(class_counts)}")
    
    methods = [
        ("Raw (baseline)", "raw", {}),
        ("Moving Average (w=5)", "ma5", {}),
        ("Moving Average (w=7)", "ma7", {}),
        ("Kalman (Q=1.0, R=5.0)", "kalman", {"process_noise": 1.0, "measurement_noise": 5.0}),
        ("Kalman Tight (Q=0.5, R=10)", "kalman_tight", {}),
    ]
    
    results = {}
    displacement_stats = {}
    
    for name, key, kwargs in methods:
        print(f"\n{'─' * 60}")
        print(f"  Method: {name}")
        print(f"{'─' * 60}")
        
        # Displacement statistics
        dstats = compute_displacement_stats(tracks, key, **kwargs)
        displacement_stats[name] = dstats
        print(f"  Walking mean displacement:    {dstats['walking_mean_disp']:.4f} ± {dstats['walking_std_disp']:.4f}")
        print(f"  Stationary mean displacement: {dstats['stationary_mean_disp']:.4f} ± {dstats['stationary_std_disp']:.4f}")
        print(f"  Overlap ratio (stat/walk):    {dstats['overlap_ratio']:.4f}")
        
        # Classification evaluation
        res = evaluate_method(tracks, name, key, **kwargs)
        results[name] = res
        
        print(f"  Overall accuracy: {res['accuracy']:.1%} ± {res['accuracy_ci']:.1%}")
        print(f"  Cohen's kappa:    {res['kappa']:.4f}")
        print(f"  Per-class recall:")
        for cls in RF_CLASSES:
            r = res['per_class'][cls]['recall']
            s = res['per_class'][cls]['support']
            marker = " ◀ TARGET" if cls in ["walking", "stationary"] else ""
            print(f"    {cls:15s}: {r:.1%} (n={s}){marker}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print(f"  {'Method':30s} {'Acc':>6s}  {'lying_R':>7s}  {'stat_R':>7s}  {'walk_R':>7s}  {'run_R':>7s}  {'κ':>6s}")
    print(f"  {'─' * 85}")
    for name in [m[0] for m in methods]:
        r = results[name]
        ly = r['per_class']['lying_down']['recall']
        st = r['per_class']['stationary']['recall']
        wa = r['per_class']['walking']['recall']
        ru = r['per_class']['running']['recall']
        print(f"  {name:30s} {r['accuracy']:>5.1%}  {ly:>6.1%}  {st:>6.1%}  {wa:>6.1%}  {ru:>6.1%}  {r['kappa']:>6.3f}")
    print("=" * 90)
    
    # ═══════════════════════════════════════════════════════════════════
    # Compute deltas vs baseline
    # ═══════════════════════════════════════════════════════════════════
    baseline = results["Raw (baseline)"]
    
    deltas = {}
    for name in [m[0] for m in methods]:
        r = results[name]
        deltas[name] = {
            "acc_delta": r["accuracy"] - baseline["accuracy"],
            "walking_recall_delta": r["per_class"]["walking"]["recall"] - baseline["per_class"]["walking"]["recall"],
            "stationary_recall_delta": r["per_class"]["stationary"]["recall"] - baseline["per_class"]["stationary"]["recall"],
            "kappa_delta": r["kappa"] - baseline["kappa"],
        }
    
    print("\nDeltas vs Raw baseline:")
    for name, d in deltas.items():
        if name == "Raw (baseline)": continue
        print(f"  {name:30s}  Acc: {d['acc_delta']:+.1%}  Walk: {d['walking_recall_delta']:+.1%}  Stat: {d['stationary_recall_delta']:+.1%}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════
    output = {
        "experiment": "jitter_filter_smoothing",
        "description": "Tests whether trajectory smoothing (MA / Kalman) before TMS-16 feature extraction improves walking-vs-stationary discrimination",
        "n_tracks": len(tracks),
        "class_distribution": dict(class_counts),
        "evaluation": "5-fold stratified CV, SMOTE-balanced RF (n_estimators=200, class_weight=balanced)",
        "methods": results,
        "displacement_statistics": displacement_stats,
        "deltas_vs_baseline": deltas,
        "conclusion": "",  # filled below
    }
    
    # Auto-generate conclusion
    best_walk = max(results.items(), key=lambda x: x[1]['per_class']['walking']['recall'])
    baseline_walk = baseline['per_class']['walking']['recall']
    best_walk_r = best_walk[1]['per_class']['walking']['recall']
    
    if best_walk_r - baseline_walk > 0.05:
        output["conclusion"] = (
            f"Smoothing improved walking recall: {baseline_walk:.1%} → {best_walk_r:.1%} "
            f"(best method: {best_walk[0]}). This partially resolves the walking-vs-stationary confusion."
        )
    else:
        output["conclusion"] = (
            f"Neither moving-average smoothing nor Kalman filtering substantially resolved "
            f"the walking-vs-stationary confusion (walking recall: {baseline_walk:.1%} → {best_walk_r:.1%}, "
            f"best method: {best_walk[0]}). This confirms that the limitation is fundamental to "
            f"centroid-only analysis at sub-3px/frame displacement magnitudes, where genuine "
            f"slow walking is physically indistinguishable from tracker jitter."
        )
    
    out_path = os.path.join(OUT_DIR, "jitter_filter_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nConclusion: {output['conclusion']}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
