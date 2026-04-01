"""
error_decomposition.py — Systematic Error Decomposition for SARTriage v3
=========================================================================
Analyses v3 pipeline results on 3 Okutama test sequences.

Parts:
  A. Error decomposition per GT track (5 categories)
  B. Lying_down failure case study
  C. False positive analysis (top-10)
  D. Summary statistics (detection/tracking/classification/ranking)

Output:
  evaluation/real_data/full/error_decomposition.json
  evaluation/real_data/full/error_decomposition.png
  evaluation/real_data/full/lying_failure_cases.json
  evaluation/real_data/full/false_positive_analysis.json
"""

import json, os, math, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ────────────────────────────────────────────────────────────────────

V3_DIR = Path(__file__).parent / "full" / "end_to_end_v3"
OUT_DIR = Path(__file__).parent / "full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_DIR = Path("/Users/aaronsandhu/Downloads/TestSetFrames/Labels/MultiActionLabels/3840x2160")
SCALE_X = 1280.0 / 3840.0
SCALE_Y = 720.0 / 2160.0

SAR_MAP = {"Standing": "stationary", "Sitting": "stationary", "Walking": "walking",
           "Running": "running", "Lying": "lying_down", "Reading": "stationary",
           "Carrying": "walking", "Drinking": "stationary", "Pushing/Pulling": "walking"}

CATEGORY_LABELS = [
    "Full Success",
    "Ranking Failure",
    "Classification Failure",
    "Tracking Failure",
    "Detection Failure",
]
CATEGORY_COLORS = ["#27ae60", "#f39c12", "#e67e22", "#9b59b6", "#e74c3c"]

SEQUENCES = ["1.1.8", "1.2.3", "2.2.3"]
MIN_SIZE_PX = 15

# ═══════════════════════════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_gt_tracks(seq):
    """Load ground truth tracks from Okutama label files."""
    gt = defaultdict(lambda: {"frames": [], "bboxes": [], "action": None})
    label_path = LABELS_DIR / f"{seq}.txt"
    if not label_path.exists():
        print(f"  ⚠ Label file not found: {label_path}")
        return {}
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 11:
                continue
            tid = int(p[0])
            xmin = float(p[1]) * SCALE_X
            ymin = float(p[2]) * SCALE_Y
            xmax = float(p[3]) * SCALE_X
            ymax = float(p[4]) * SCALE_Y
            fi = int(p[5])
            act = p[10].strip('"')
            gt[tid]["frames"].append(fi)
            gt[tid]["bboxes"].append([xmin, ymin, xmax - xmin, ymax - ymin])
            gt[tid]["action"] = act
    return dict(gt)


def load_v3_results(seq):
    """Load v3 pipeline results for a sequence."""
    fn = f"v3_{seq.replace('.', '_')}.json"
    with open(V3_DIR / fn) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Error Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_errors(seq):
    """Classify each GT track into one of 5 failure categories."""
    gt_tracks = load_gt_tracks(seq)
    v3 = load_v3_results(seq)
    ranked = v3["ranked_output"]

    # Build lookup: gt_id → list of pipeline tracks that matched
    gt_to_pipeline = defaultdict(list)
    for t in ranked:
        if t["gt_match_id"] is not None:
            gt_to_pipeline[t["gt_match_id"]].append(t)

    # Also need to know which GT tracks were in filtered-out set
    n_before = v3["n_before_filter"]
    n_after = v3["n_tracks"]

    results = []
    for gt_id, gt_data in gt_tracks.items():
        gt_action = gt_data["action"]
        gt_sar = SAR_MAP.get(gt_action, "other")
        gt_n_frames = len(gt_data["frames"])

        # Compute average GT person size
        sizes = [math.sqrt(max(b[2], 1) * max(b[3], 1)) for b in gt_data["bboxes"]]
        mean_size = np.mean(sizes) if sizes else 0

        # Size bin
        if mean_size < 15:
            size_bin = "<15px"
        elif mean_size < 25:
            size_bin = "15-25px"
        elif mean_size < 40:
            size_bin = "25-40px"
        else:
            size_bin = "40+px"

        pipeline_tracks = gt_to_pipeline.get(gt_id, [])

        if not pipeline_tracks:
            # No pipeline track matched this GT person
            # Could be detection failure OR tracking failure
            # Heuristic: if person size < 15px or very few frames → likely detection
            # We classify as DETECTION FAILURE (YOLO didn't find them or they were filtered)
            category = "Detection Failure"
            cat_idx = 4
            detail = {
                "reason": "No pipeline track matched this GT person",
                "gt_n_frames": gt_n_frames,
                "gt_mean_size": round(mean_size, 1),
                "possible_cause": "Too small for YOLO" if mean_size < 20 else
                                  "Occluded or edge-of-frame" if mean_size < 30 else
                                  "Unknown — large enough to detect",
            }
        else:
            # GT person WAS tracked. Now check classification and ranking.
            best_track = min(pipeline_tracks, key=lambda t: t["rank"])
            ensemble_class = best_track["ensemble_class"]
            rank = best_track["rank"]
            final_priority = best_track["final_priority"]

            # Is classification correct?
            # Map ensemble class to SAR: stationary→stationary, walking→walking, running→running, lying_down→lying_down
            class_correct = (ensemble_class == gt_sar) or \
                            (ensemble_class == "stationary" and gt_sar in ("stationary",)) or \
                            (ensemble_class == "walking" and gt_sar in ("walking",))

            # Special case: "lying_down" in GT but classified differently
            if gt_sar == "lying_down" and ensemble_class != "lying_down":
                class_correct = False

            if not class_correct:
                category = "Classification Failure"
                cat_idx = 2
                detail = {
                    "gt_sar": gt_sar,
                    "predicted": ensemble_class,
                    "rank": rank,
                    "tms16": best_track["tms16_class"],
                    "scte": best_track["scte_class"],
                    "trajmae": best_track["trajmae_class"],
                    "mvit": best_track["mvit_class"],
                    "final_priority": final_priority,
                }
            elif rank <= 3:
                category = "Full Success"
                cat_idx = 0
                detail = {
                    "rank": rank,
                    "final_priority": final_priority,
                    "ensemble_class": ensemble_class,
                }
            else:
                # Classified correctly but ranked too low
                category = "Ranking Failure"
                cat_idx = 1
                detail = {
                    "rank": rank,
                    "final_priority": final_priority,
                    "ensemble_class": ensemble_class,
                    "tce_state": best_track["tce_state"],
                    "tce_score": best_track["tce_score"],
                    "fused_score": best_track["fused_score"],
                }

        results.append({
            "gt_id": gt_id,
            "gt_action": gt_action,
            "gt_sar": gt_sar,
            "gt_n_frames": gt_n_frames,
            "gt_mean_size": round(mean_size, 1),
            "size_bin": size_bin,
            "category": category,
            "category_idx": cat_idx,
            "detail": detail,
            "sequence": seq,
        })

    return results


def run_part_a():
    """Run error decomposition for all sequences."""
    print("\n📊 PART A: Error Decomposition")
    print("=" * 60)

    all_results = []
    for seq in SEQUENCES:
        results = decompose_errors(seq)
        all_results.extend(results)
        counts = Counter(r["category"] for r in results)
        print(f"\n  {seq}: {len(results)} GT persons")
        for cat in CATEGORY_LABELS:
            n = counts.get(cat, 0)
            pct = n / len(results) * 100 if results else 0
            print(f"    {cat:<25} {n:3d}  ({pct:5.1f}%)")

    # Overall
    total = len(all_results)
    overall = Counter(r["category"] for r in all_results)
    print(f"\n  OVERALL: {total} GT persons across 3 sequences")
    for cat in CATEGORY_LABELS:
        n = overall.get(cat, 0)
        pct = n / total * 100 if total else 0
        print(f"    {cat:<25} {n:3d}  ({pct:5.1f}%)")

    # Breakdown by action class
    action_breakdown = {}
    for r in all_results:
        sar = r["gt_sar"]
        if sar not in action_breakdown:
            action_breakdown[sar] = Counter()
        action_breakdown[sar][r["category"]] += 1

    print("\n  By action class:")
    for action in sorted(action_breakdown.keys()):
        counts = action_breakdown[action]
        total_a = sum(counts.values())
        print(f"    {action} ({total_a} tracks):")
        for cat in CATEGORY_LABELS:
            n = counts.get(cat, 0)
            if n > 0:
                print(f"      {cat:<25} {n:3d}  ({n/total_a*100:5.1f}%)")

    # Breakdown by size bin
    size_breakdown = {}
    for r in all_results:
        sb = r["size_bin"]
        if sb not in size_breakdown:
            size_breakdown[sb] = Counter()
        size_breakdown[sb][r["category"]] += 1

    print("\n  By person size:")
    for sb in ["<15px", "15-25px", "25-40px", "40+px"]:
        if sb not in size_breakdown:
            continue
        counts = size_breakdown[sb]
        total_s = sum(counts.values())
        print(f"    {sb} ({total_s} tracks):")
        for cat in CATEGORY_LABELS:
            n = counts.get(cat, 0)
            if n > 0:
                print(f"      {cat:<25} {n:3d}  ({n/total_s*100:5.1f}%)")

    return all_results, action_breakdown, size_breakdown


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Lying_down Failure Case Study
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_b(all_results):
    """Analyse lying_down GT tracks that failed to rank in top 3."""
    print("\n\n📋 PART B: Lying_down Failure Case Study")
    print("=" * 60)

    lying = [r for r in all_results if r["gt_sar"] == "lying_down"]
    if not lying:
        print("  No lying_down GT tracks found in the dataset.")
        print("  NOTE: Okutama 1.1.8 has 2 'Lying' GT persons, 1.2.3 and 2.2.3 have none.")
        # Still check if any lying were in the ranked output from other GT sources
        lying_cases = []
        for seq in SEQUENCES:
            v3 = load_v3_results(seq)
            for t in v3["ranked_output"]:
                if t.get("gt_action") == "Lying":
                    lying_cases.append({
                        "sequence": seq,
                        "track": t,
                        "was_top3": t["rank"] <= 3,
                    })
        if lying_cases:
            print(f"\n  Found {len(lying_cases)} pipeline tracks matching 'Lying' GT:")
            for lc in lying_cases:
                t = lc["track"]
                print(f"    Seq {lc['sequence']}: rank={t['rank']}, ensemble={t['ensemble_class']}, "
                      f"priority={t['final_priority']:.4f}, size={t['person_size_px']}px")
        return lying_cases

    lying_failures = [r for r in lying if r["category"] != "Full Success"]
    lying_successes = [r for r in lying if r["category"] == "Full Success"]

    print(f"  {len(lying)} total lying_down GT tracks")
    print(f"  {len(lying_successes)} successes (top-3), {len(lying_failures)} failures")

    failure_details = []
    for r in lying_failures:
        print(f"\n  GT ID {r['gt_id']} (seq {r['sequence']}):")
        print(f"    Category: {r['category']}")
        print(f"    Mean size: {r['gt_mean_size']}px")
        print(f"    N frames: {r['gt_n_frames']}")
        for k, v in r["detail"].items():
            print(f"    {k}: {v}")
        failure_details.append(r)

    # Pattern analysis
    if lying_failures:
        failure_modes = Counter(r["category"] for r in lying_failures)
        print(f"\n  Failure mode pattern: {dict(failure_modes)}")
        dominant = failure_modes.most_common(1)[0]
        print(f"  Dominant failure mode: {dominant[0]} ({dominant[1]}/{len(lying_failures)} = "
              f"{dominant[1]/len(lying_failures)*100:.0f}%)")

    return failure_details


# ═══════════════════════════════════════════════════════════════════════════════
# PART C: False Positive Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_c():
    """Analyse top-10 highest-priority tracks in each sequence."""
    print("\n\n🔍 PART C: False Positive Analysis (Top-10)")
    print("=" * 60)

    fp_analysis = {}
    all_tp = 0
    all_fp = 0

    for seq in SEQUENCES:
        v3 = load_v3_results(seq)
        top10 = v3["ranked_output"][:10]

        tp_tracks = [t for t in top10 if t.get("gt_match_id") is not None]
        fp_tracks = [t for t in top10 if t.get("gt_match_id") is None]

        all_tp += len(tp_tracks)
        all_fp += len(fp_tracks)

        print(f"\n  {seq}: {len(tp_tracks)} TP, {len(fp_tracks)} FP in top-10")

        fp_details = []
        for t in fp_tracks:
            print(f"    FP: rank={t['rank']}, track_id={t['track_id']}, "
                  f"ensemble={t['ensemble_class']}, size={t['person_size_px']}px, "
                  f"n_frames={t['n_frames']}, priority={t['final_priority']:.4f}")
            print(f"         TMS16={t['tms16_class']}, TrajMAE={t['trajmae_class']}, "
                  f"SCTE={t['scte_class']}, MViT={t['mvit_class']}")
            print(f"         TCE={t['tce_state']}({t['tce_score']:.3f}), "
                  f"AAI w_traj={t['aai_w_traj']:.3f}, fused={t['fused_score']:.4f}")
            fp_details.append({
                "rank": t["rank"],
                "track_id": t["track_id"],
                "ensemble_class": t["ensemble_class"],
                "person_size_px": t["person_size_px"],
                "n_frames": t["n_frames"],
                "final_priority": t["final_priority"],
                "tms16_class": t["tms16_class"],
                "trajmae_class": t["trajmae_class"],
                "scte_class": t["scte_class"],
                "mvit_class": t["mvit_class"],
                "tce_state": t["tce_state"],
                "tce_score": t["tce_score"],
                "aai_w_traj": t["aai_w_traj"],
                "aai_w_pixel": t["aai_w_pixel"],
                "fused_score": t["fused_score"],
                "likely_cause": (
                    "Short transient track (clutter)" if t["n_frames"] < 8 else
                    "Small person, likely background structure" if t["person_size_px"] < 18 else
                    "Misclassified background object"
                ),
            })

        fp_analysis[seq] = {
            "n_tp": len(tp_tracks),
            "n_fp": len(fp_tracks),
            "precision_at_10": len(tp_tracks) / 10,
            "false_positives": fp_details,
        }

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    print(f"\n  Overall top-10: {all_tp} TP, {all_fp} FP → Precision@10 = {precision:.1%}")

    fp_analysis["overall"] = {
        "total_tp": all_tp,
        "total_fp": all_fp,
        "precision_at_10": precision,
    }

    return fp_analysis


# ═══════════════════════════════════════════════════════════════════════════════
# PART D: Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def run_part_d(all_results):
    """Compute detection recall, tracking recall, classification accuracy, ranking P@3."""
    print("\n\n📈 PART D: Summary Statistics")
    print("=" * 60)

    summary = {}

    for seq in SEQUENCES:
        gt_tracks = load_gt_tracks(seq)
        v3 = load_v3_results(seq)
        ranked = v3["ranked_output"]
        seq_results = [r for r in all_results if r["sequence"] == seq]

        n_gt = len(gt_tracks)
        n_detected = sum(1 for r in seq_results if r["category"] != "Detection Failure")
        n_tracked = n_detected  # if detected, they were tracked (tracked=matched to pipeline track)
        n_classified_correct = sum(1 for r in seq_results
                                   if r["category"] in ("Full Success", "Ranking Failure"))

        # Ranking P@3: of top-3 ranked, how many are GT-relevant?
        top3 = ranked[:3]
        top3_has_gt = sum(1 for t in top3 if t.get("gt_match_id") is not None)
        # More specifically: how many top-3 are "important" (lying/sitting/stationary)?
        top3_relevant = sum(1 for t in top3
                            if t.get("gt_sar") in ("stationary", "lying_down", "running"))

        det_recall = n_detected / n_gt if n_gt > 0 else 0
        track_recall = n_tracked / n_detected if n_detected > 0 else 0  # trivially 1.0
        class_acc = n_classified_correct / n_tracked if n_tracked > 0 else 0
        rank_p3 = top3_has_gt / 3

        summary[seq] = {
            "n_gt_persons": n_gt,
            "detection_recall": round(det_recall, 3),
            "tracking_recall": round(track_recall, 3),
            "classification_accuracy": round(class_acc, 3),
            "ranking_precision_at_3": round(rank_p3, 3),
            "n_detected": n_detected,
            "n_tracked": n_tracked,
            "n_classified_correct": n_classified_correct,
            "top3_gt_matches": top3_has_gt,
            "ndcg3": v3["ndcg3"],
            "ndcg5": v3["ndcg5"],
        }

        print(f"\n  {seq}:")
        print(f"    GT persons:              {n_gt}")
        print(f"    Detection recall:        {det_recall:.1%}  ({n_detected}/{n_gt})")
        print(f"    Tracking recall:         {track_recall:.1%}  ({n_tracked}/{n_detected})")
        print(f"    Classification accuracy: {class_acc:.1%}  ({n_classified_correct}/{n_tracked})")
        print(f"    Ranking P@3:             {rank_p3:.1%}  ({top3_has_gt}/3)")
        print(f"    NDCG@3:                  {v3['ndcg3']:.4f}")

    # Overall
    totals = {k: sum(summary[s][k] for s in SEQUENCES)
              for k in ["n_gt_persons", "n_detected", "n_tracked", "n_classified_correct", "top3_gt_matches"]}
    overall = {
        "detection_recall": round(totals["n_detected"] / max(totals["n_gt_persons"], 1), 3),
        "tracking_recall": round(totals["n_tracked"] / max(totals["n_detected"], 1), 3),
        "classification_accuracy": round(totals["n_classified_correct"] / max(totals["n_tracked"], 1), 3),
        "ranking_precision_at_3": round(totals["top3_gt_matches"] / 9, 3),
        "mean_ndcg3": round(np.mean([summary[s]["ndcg3"] for s in SEQUENCES]), 4),
        **totals,
    }
    summary["overall"] = overall

    print(f"\n  OVERALL ({totals['n_gt_persons']} GT persons):")
    print(f"    Detection recall:        {overall['detection_recall']:.1%}")
    print(f"    Tracking recall:         {overall['tracking_recall']:.1%}")
    print(f"    Classification accuracy: {overall['classification_accuracy']:.1%}")
    print(f"    Ranking P@3:             {overall['ranking_precision_at_3']:.1%}")
    print(f"    Mean NDCG@3:             {overall['mean_ndcg3']:.4f}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_decomposition(all_results, action_breakdown, summary):
    """Create publication-quality error decomposition figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # ── Panel 1: Stacked bar by action class ──
    ax1 = fig.add_subplot(gs[0, :2])
    action_order = ["stationary", "walking", "running", "lying_down", "other"]
    actions_present = [a for a in action_order if a in action_breakdown]

    n_actions = len(actions_present)
    x = np.arange(n_actions)
    bottoms = np.zeros(n_actions)

    for cat_idx, cat in enumerate(CATEGORY_LABELS):
        vals = [action_breakdown[a].get(cat, 0) for a in actions_present]
        bars = ax1.bar(x, vals, bottom=bottoms, color=CATEGORY_COLORS[cat_idx],
                       label=cat, alpha=0.9, edgecolor="white", linewidth=0.5)
        # Add count labels on segments > 0
        for i, v in enumerate(vals):
            if v > 0:
                ax1.text(x[i], bottoms[i] + v / 2, str(v),
                         ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        bottoms += np.array(vals, dtype=float)

    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace("_", "\n") for a in actions_present], fontsize=11)
    ax1.set_ylabel("Number of GT Tracks", fontsize=12)
    ax1.set_title("Error Decomposition by Action Class", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: Overall pie chart ──
    ax2 = fig.add_subplot(gs[0, 2])
    overall_counts = Counter(r["category"] for r in all_results)
    sizes = [overall_counts.get(cat, 0) for cat in CATEGORY_LABELS]
    non_zero = [(s, c, l) for s, c, l in zip(sizes, CATEGORY_COLORS, CATEGORY_LABELS) if s > 0]
    if non_zero:
        s_nz, c_nz, l_nz = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            s_nz, labels=l_nz, colors=c_nz, autopct="%1.0f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        for t in texts:
            t.set_fontsize(8)
    ax2.set_title(f"Overall Error Distribution\n({len(all_results)} GT tracks)", fontsize=12, fontweight="bold")

    # ── Panel 3: Stacked bar by size bin ──
    ax3 = fig.add_subplot(gs[1, 0])
    size_order = ["<15px", "15-25px", "25-40px", "40+px"]
    sizes_present = [s for s in size_order
                     if any(r["size_bin"] == s for r in all_results)]
    n_sizes = len(sizes_present)
    x3 = np.arange(n_sizes)
    bottoms3 = np.zeros(n_sizes)

    size_bd = {}
    for r in all_results:
        sb = r["size_bin"]
        if sb not in size_bd:
            size_bd[sb] = Counter()
        size_bd[sb][r["category"]] += 1

    for cat_idx, cat in enumerate(CATEGORY_LABELS):
        vals = [size_bd.get(s, Counter()).get(cat, 0) for s in sizes_present]
        ax3.bar(x3, vals, bottom=bottoms3, color=CATEGORY_COLORS[cat_idx],
                alpha=0.9, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 1:
                ax3.text(x3[i], bottoms3[i] + v / 2, str(v),
                         ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        bottoms3 += np.array(vals, dtype=float)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(sizes_present, fontsize=10)
    ax3.set_ylabel("GT Tracks", fontsize=11)
    ax3.set_title("Error Decomposition by Person Size", fontsize=12, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Component recall waterfall ──
    ax4 = fig.add_subplot(gs[1, 1])
    overall = summary.get("overall", {})
    stages = ["Detection\nRecall", "Tracking\nRecall", "Classification\nAccuracy", "Ranking\nP@3"]
    values = [
        overall.get("detection_recall", 0),
        overall.get("tracking_recall", 0),
        overall.get("classification_accuracy", 0),
        overall.get("ranking_precision_at_3", 0),
    ]
    stage_colors = ["#e74c3c", "#9b59b6", "#e67e22", "#f39c12"]

    bars4 = ax4.bar(range(len(stages)), values, color=stage_colors, alpha=0.85,
                    edgecolor="white", linewidth=1)
    for bar, val in zip(bars4, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")
    ax4.set_xticks(range(len(stages)))
    ax4.set_xticklabels(stages, fontsize=9)
    ax4.set_ylim(0, 1.15)
    ax4.set_ylabel("Rate", fontsize=11)
    ax4.set_title("Pipeline Component Performance", fontsize=12, fontweight="bold")
    ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax4.grid(True, axis="y", alpha=0.3)

    # ── Panel 5: Per-sequence NDCG ──
    ax5 = fig.add_subplot(gs[1, 2])
    seq_names = SEQUENCES
    ndcg3_vals = [summary[s]["ndcg3"] for s in seq_names]
    ndcg5_vals = [summary[s]["ndcg5"] for s in seq_names]
    x5 = np.arange(len(seq_names))
    width5 = 0.3
    ax5.bar(x5 - width5 / 2, ndcg3_vals, width5, label="NDCG@3", color="#2c3e50", alpha=0.85)
    ax5.bar(x5 + width5 / 2, ndcg5_vals, width5, label="NDCG@5", color="#3498db", alpha=0.85)
    for i in range(len(seq_names)):
        ax5.text(x5[i] - width5 / 2, ndcg3_vals[i] + 0.02, f"{ndcg3_vals[i]:.2f}",
                 ha="center", fontsize=9, fontweight="bold")
        ax5.text(x5[i] + width5 / 2, ndcg5_vals[i] + 0.02, f"{ndcg5_vals[i]:.2f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax5.set_xticks(x5)
    ax5.set_xticklabels(seq_names, fontsize=10)
    ax5.set_ylim(0, 1.15)
    ax5.set_ylabel("NDCG Score", fontsize=11)
    ax5.set_title("Ranking Quality Per Sequence", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(True, axis="y", alpha=0.3)

    plt.suptitle("SARTriage v3 — Systematic Error Decomposition\n"
                 "Where does the pipeline lose each GT person?",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(OUT_DIR / "error_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved error_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  SARTriage v3 — Systematic Error Decomposition")
    print("=" * 70)

    # Part A
    all_results, action_breakdown, size_breakdown = run_part_a()

    # Part B
    lying_details = run_part_b(all_results)

    # Part C
    fp_analysis = run_part_c()

    # Part D
    summary = run_part_d(all_results)

    # Plot
    print("\n\n🎨 Generating figures...")
    plot_error_decomposition(all_results, action_breakdown, summary)

    # Save JSONs
    decomp_data = {
        "version": "v3_error_decomposition",
        "total_gt_tracks": len(all_results),
        "category_counts": dict(Counter(r["category"] for r in all_results)),
        "per_action_class": {k: dict(v) for k, v in action_breakdown.items()},
        "per_size_bin": {k: dict(v) for k, v in size_breakdown.items()},
        "tracks": all_results,
        "summary_stats": summary,
    }
    with open(OUT_DIR / "error_decomposition.json", "w") as f:
        json.dump(decomp_data, f, indent=2, default=str)
    print(f"  ✓ Saved error_decomposition.json")

    with open(OUT_DIR / "lying_failure_cases.json", "w") as f:
        json.dump({"lying_down_failures": lying_details or [],
                    "note": "Okutama test sequences have very few/no lying_down GT tracks"}, f, indent=2, default=str)
    print(f"  ✓ Saved lying_failure_cases.json")

    with open(OUT_DIR / "false_positive_analysis.json", "w") as f:
        json.dump(fp_analysis, f, indent=2, default=str)
    print(f"  ✓ Saved false_positive_analysis.json")

    print("\n" + "=" * 70)
    print("  ✓ Error decomposition complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
