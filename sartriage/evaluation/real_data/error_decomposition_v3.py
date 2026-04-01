"""
Error Decomposition Analysis — SARTriage v3
============================================
Decomposes pipeline errors into stages for 3 Okutama test sequences.

Two perspectives:
A) GT-track perspective: what happened to each of the 110 GT persons?
B) Pipeline-track perspective: of the 74 GT-matched tracks, where did errors occur?

Error categories (for matched tracks):
  1. FULL SUCCESS        — classified correctly AND ranked in top half
  2. CLASSIFICATION ERR  — tracked but wrong ensemble class
  3. RANKING FAILURE     — correct class but ranked in bottom half

For unmatched GT tracks:
  4. DETECTION/TRACKING FAILURE — GT person was never matched by the pipeline
     (pipeline may not have detected them, or created too-short/fragmented tracks
      that couldn't be matched via IoU to the GT annotation)
"""

import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

OUT_DIR = "evaluation/real_data/full"
os.makedirs(OUT_DIR, exist_ok=True)

SAR_MAP = {
    "Standing": "stationary", "Sitting": "stationary",
    "Walking": "walking", "Running": "running",
    "Lying": "lying_down",
    "Reading": "stationary", "Drinking": "stationary",
    "Carrying": "walking", "Pushing": "walking", "Pulling": "walking",
    "Hand Shaking": "stationary", "Hugging": "stationary",
    "Calling": "stationary",
}

CATEGORIES = [
    "Full Success",
    "Detection/Tracking Gap",
    "Classification Error",
    "Ranking Failure",
]

CAT_COLORS = {
    "Full Success":            "#22c55e",
    "Detection/Tracking Gap":  "#ef4444",
    "Classification Error":    "#eab308",
    "Ranking Failure":         "#a855f7",
}

SAR_CLASSES = ["lying_down", "stationary", "walking", "running"]
TEST_SEQUENCES = ["1.1.8", "1.2.3", "2.2.3"]


def load_gt_tracks():
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    
    gt_tracks = []
    for tid, t in data["tracks"].items():
        seq = t.get("sequence", "")
        if seq not in TEST_SEQUENCES:
            continue
        act = t["primary_action"]
        sar_class = SAR_MAP.get(act)
        if sar_class is None:
            continue
        
        try:
            numeric_id = int(tid.split("_track")[1])
        except (IndexError, ValueError):
            numeric_id = None
        
        gt_tracks.append({
            "gt_tid": tid,
            "numeric_id": numeric_id,
            "sequence": seq,
            "gt_action": act,
            "sar_class": sar_class,
            "n_frames": t["track_length_frames"],
            "mean_size_px": t.get("mean_size_px", 0),
        })
    
    return gt_tracks


def load_v3_results():
    v3_tracks = []
    v3_meta = {}
    
    for seq in TEST_SEQUENCES:
        fn = f"evaluation/real_data/full/end_to_end_v3/v3_{seq.replace('.', '_')}.json"
        with open(fn) as f:
            data = json.load(f)
        
        v3_meta[seq] = {
            "n_tracks": data["n_tracks"],
            "n_before_filter": data["n_before_filter"],
            "n_yolo_dets": data["n_yolo_dets"],
        }
        
        for r in data["ranked_output"]:
            r["sequence"] = seq
            v3_tracks.append(r)
    
    return v3_tracks, v3_meta


def decompose_errors(gt_tracks, v3_tracks, v3_meta):
    """Decompose from GT track perspective."""
    
    # Build lookup: (seq, gt_match_id) → v3 track
    matched_gt_ids = {}
    for v in v3_tracks:
        if v.get("gt_match_id") is not None:
            key = (v["sequence"], v["gt_match_id"])
            matched_gt_ids[key] = v
    
    # Median priority per sequence for "top half" definition
    priorities_per_seq = defaultdict(list)
    for v in v3_tracks:
        priorities_per_seq[v["sequence"]].append(v["final_priority"])
    median_priority = {
        seq: float(np.median(prios)) for seq, prios in priorities_per_seq.items()
    }
    
    results = []
    
    for gt in gt_tracks:
        seq = gt["sequence"]
        key = (seq, gt["numeric_id"])
        v3_match = matched_gt_ids.get(key)
        
        if v3_match is None:
            category = "Detection/Tracking Gap"
            v3_info = None
        else:
            ensemble_class = v3_match["ensemble_class"]
            gt_sar = gt["sar_class"]
            priority = v3_match["final_priority"]
            
            if ensemble_class != gt_sar:
                category = "Classification Error"
            elif priority < median_priority[seq]:
                category = "Ranking Failure"
            else:
                category = "Full Success"
            
            v3_info = {
                "track_id": v3_match["track_id"],
                "ensemble_class": ensemble_class,
                "tms16_class": v3_match["tms16_class"],
                "priority": priority,
                "n_frames": v3_match["n_frames"],
                "person_size_px": v3_match["person_size_px"],
            }
        
        results.append({
            "gt_tid": gt["gt_tid"],
            "sequence": seq,
            "gt_action": gt["gt_action"],
            "sar_class": gt["sar_class"],
            "mean_size_px": gt["mean_size_px"],
            "gt_n_frames": gt["n_frames"],
            "category": category,
            "v3_match": v3_info,
        })
    
    return results, median_priority


def make_stacked_bar(results, out_path):
    """Stacked bar chart — error decomposition by action class."""
    
    counts = {cls: {cat: 0 for cat in CATEGORIES} for cls in SAR_CLASSES}
    for r in results:
        cls = r["sar_class"]
        cat = r["category"]
        if cls in counts and cat in counts[cls]:
            counts[cls][cat] += 1
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(SAR_CLASSES))
    bar_width = 0.55
    bottom = np.zeros(len(SAR_CLASSES))
    
    for cat in CATEGORIES:
        values = [counts[cls][cat] for cls in SAR_CLASSES]
        bars = ax.bar(x, values, bar_width, bottom=bottom,
                      label=cat, color=CAT_COLORS[cat],
                      edgecolor='white', linewidth=1.0)
        
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i, bot + val/2, str(val), ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if cat != "Ranking Failure" else 'white')
        
        bottom += np.array(values)
    
    # Total labels on top
    for i, total in enumerate(bottom):
        ax.text(i, total + 0.8, f'n={int(total)}', ha='center', va='bottom',
                fontsize=10, color='#333', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in SAR_CLASSES], fontsize=13)
    ax.set_ylabel('Number of GT Tracks', fontsize=13)
    ax.set_title('SARTriage v3 — Error Decomposition by Action Class\n'
                 '(3 Okutama-Action Test Sequences, n=110 GT tracks)',
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95,
              fancybox=True, shadow=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(bottom) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {out_path}")


def make_pipeline_waterfall(results, v3_meta, out_path):
    """Waterfall/funnel chart showing track counts at each pipeline stage."""
    
    total_gt = len(results)
    total_yolo = sum(m["n_yolo_dets"] for m in v3_meta.values())
    total_raw = sum(m["n_before_filter"] for m in v3_meta.values())
    total_filtered = sum(m["n_tracks"] for m in v3_meta.values())
    
    n_matched = sum(1 for r in results if r["category"] != "Detection/Tracking Gap")
    n_correct_class = sum(1 for r in results if r["category"] in ["Full Success", "Ranking Failure"])
    n_correct_rank = sum(1 for r in results if r["category"] == "Full Success")
    
    stages = [
        ("GT Persons\n(ground truth)", total_gt),
        ("YOLO\nDetections", total_yolo),
        ("Raw\nTracks", total_raw),
        ("Filtered\nTracks", total_filtered),
        ("GT-Matched\nTracks", n_matched),
        ("Correctly\nClassified", n_correct_class),
        ("Correctly\nRanked", n_correct_rank),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    
    # Use different colors for different pipeline sections
    colors = ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#22c55e', '#4ade80', '#86efac']
    
    bars = ax.bar(range(len(stages)), values, color=colors, 
                  edgecolor='white', linewidth=1.5, width=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(i, val + max(values)*0.02, str(val),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Show dropout between stages
        if i > 0 and i >= 4:  # Only for GT-perspective stages
            prev_val = values[i-1] if i > 0 else val
            if prev_val > val and prev_val > 0:
                dropout = prev_val - val
                pct = dropout / prev_val * 100
                ax.annotate(f'−{dropout}\n({pct:.0f}%)',
                           xy=(i-0.5, (prev_val + val)/2),
                           fontsize=8, color='#dc2626', ha='center',
                           fontweight='bold')
    
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Pipeline Funnel: GT Persons → Correct Rankings\n'
                 '(showing where tracks are lost at each stage)',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    
    # Add separator line between pipeline stages and GT matching
    ax.axvline(3.5, color='#666', linestyle=':', alpha=0.5)
    ax.text(1.5, max(values)*0.95, 'Pipeline\nProcessing', ha='center',
            fontsize=9, color='#666', fontstyle='italic')
    ax.text(5, max(values)*0.95, 'GT Matching\n& Evaluation', ha='center',
            fontsize=9, color='#666', fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Funnel chart saved: {out_path}")


def main():
    print("=" * 70)
    print("ERROR DECOMPOSITION ANALYSIS — SARTriage v3")
    print("=" * 70)
    
    print("\nLoading data...")
    gt_tracks = load_gt_tracks()
    v3_tracks, v3_meta = load_v3_results()
    print(f"  {len(gt_tracks)} GT tracks, {len(v3_tracks)} V3 pipeline tracks")
    
    print("\nDecomposing errors...")
    results, median_priority = decompose_errors(gt_tracks, v3_tracks, v3_meta)
    
    # ═══════════════════════════════════════════════════════════════════
    # Summary statistics
    # ═══════════════════════════════════════════════════════════════════
    cat_counts = defaultdict(int)
    for r in results:
        cat_counts[r["category"]] += 1
    
    n_total = len(results)
    n_matched = n_total - cat_counts["Detection/Tracking Gap"]
    n_failures = n_total - cat_counts["Full Success"]
    
    print(f"\n{'='*65}")
    print(f"  OVERALL ERROR DECOMPOSITION (n={n_total} GT tracks)")
    print(f"{'='*65}")
    for cat in CATEGORIES:
        count = cat_counts[cat]
        pct = count / n_total * 100
        pct_fail = count / max(n_failures, 1) * 100 if cat != "Full Success" else 0
        extra = f"  [{pct_fail:.0f}% of failures]" if cat != "Full Success" else ""
        print(f"  {cat:25s}: {count:3d}  ({pct:5.1f}%){extra}")
    
    print(f"\n  GT Match Rate: {n_matched}/{n_total} = {n_matched/n_total:.1%}")
    print(f"  Success Rate (of matched): {cat_counts['Full Success']}/{n_matched} = {cat_counts['Full Success']/max(n_matched,1):.1%}")
    print(f"  Overall Success Rate: {cat_counts['Full Success']}/{n_total} = {cat_counts['Full Success']/n_total:.1%}")
    
    # ── By action class ──
    print(f"\n  {'':25s} {'lying':>6s} {'stat':>6s} {'walk':>6s} {'run':>6s} {'TOTAL':>6s}")
    print(f"  {'─'*58}")
    for cat in CATEGORIES:
        vals = [sum(1 for r in results if r["category"] == cat and r["sar_class"] == cls
                   ) for cls in SAR_CLASSES]
        vals.append(sum(vals))
        print(f"  {cat:25s} {vals[0]:6d} {vals[1]:6d} {vals[2]:6d} {vals[3]:6d} {vals[4]:6d}")
    
    totals = [sum(1 for r in results if r["sar_class"] == cls) for cls in SAR_CLASSES]
    totals.append(sum(totals))
    print(f"  {'─'*58}")
    print(f"  {'TOTAL':25s} {totals[0]:6d} {totals[1]:6d} {totals[2]:6d} {totals[3]:6d} {totals[4]:6d}")
    
    # ── Of matched tracks only ──
    matched_results = [r for r in results if r["category"] != "Detection/Tracking Gap"]
    print(f"\n  MATCHED TRACKS ONLY (n={len(matched_results)}):")
    for cat in ["Full Success", "Classification Error", "Ranking Failure"]:
        count = sum(1 for r in matched_results if r["category"] == cat)
        print(f"    {cat:25s}: {count:3d}  ({count/max(len(matched_results),1):.1%})")
    
    # Classification confusion for matched tracks
    print(f"\n  Classification confusion (matched tracks):")
    for r in matched_results:
        if r["category"] == "Classification Error":
            gt = r["sar_class"]
            pred = r["v3_match"]["ensemble_class"]
            print(f"    GT={gt:12s} → Pred={pred:12s}  (Track {r['v3_match']['track_id']}, {r['v3_match']['person_size_px']:.0f}px)")
    
    # ── Error attribution (% contribution) ──
    print(f"\n  ERROR ATTRIBUTION (% of total failures):")
    for cat in ["Detection/Tracking Gap", "Classification Error", "Ranking Failure"]:
        count = cat_counts[cat]
        pct = count / max(n_failures, 1) * 100
        print(f"    {cat:25s}: {count:3d} ({pct:.1f}%)")
    
    # ═══════════════════════════════════════════════════════════════════
    # Charts
    # ═══════════════════════════════════════════════════════════════════
    print("\nGenerating charts...")
    make_stacked_bar(results, os.path.join(OUT_DIR, "error_decomposition.png"))
    make_pipeline_waterfall(results, v3_meta, os.path.join(OUT_DIR, "error_decomposition_funnel.png"))
    
    # ═══════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════
    
    class_breakdown = {}
    for cls in SAR_CLASSES:
        cls_results = [r for r in results if r["sar_class"] == cls]
        cls_matched = [r for r in cls_results if r["category"] != "Detection/Tracking Gap"]
        class_breakdown[cls] = {
            "total_gt": len(cls_results),
            "matched": len(cls_matched),
            "match_rate": round(len(cls_matched) / max(len(cls_results), 1), 4),
            **{cat: sum(1 for r in cls_results if r["category"] == cat) for cat in CATEGORIES},
        }
    
    seq_breakdown = {}
    for seq in TEST_SEQUENCES:
        seq_results = [r for r in results if r["sequence"] == seq]
        seq_breakdown[seq] = {
            "total_gt": len(seq_results),
            "v3_tracks": v3_meta[seq]["n_tracks"],
            "v3_before_filter": v3_meta[seq]["n_before_filter"],
            "yolo_dets": v3_meta[seq]["n_yolo_dets"],
            "median_priority": float(median_priority[seq]),
            **{cat: sum(1 for r in seq_results if r["category"] == cat) for cat in CATEGORIES},
        }
    
    error_attribution = {}
    for cat in ["Detection/Tracking Gap", "Classification Error", "Ranking Failure"]:
        count = cat_counts[cat]
        error_attribution[cat] = {
            "count": count,
            "pct_of_total": round(count / n_total * 100, 1),
            "pct_of_failures": round(count / max(n_failures, 1) * 100, 1),
        }
    
    # Summary finding
    det_pct = error_attribution["Detection/Tracking Gap"]["pct_of_failures"]
    cls_pct = error_attribution["Classification Error"]["pct_of_failures"]
    rank_pct = error_attribution["Ranking Failure"]["pct_of_failures"]
    
    finding = (
        f"Of {n_total} GT persons, {n_matched} ({n_matched/n_total:.0%}) were successfully matched "
        f"by the pipeline. Of {n_failures} total failures: detection/tracking gaps account for "
        f"{det_pct:.0f}%, classification errors for {cls_pct:.0f}%, and ranking failures for "
        f"{rank_pct:.0f}%. The primary bottleneck is the detection-to-tracking pipeline "
        f"(IoU matching at drone resolution), not the TMS-16 classifier itself."
    )
    
    output = {
        "experiment": "error_decomposition_v3",
        "description": "Stage-by-stage error decomposition of SARTriage v3 on 3 Okutama test sequences",
        "n_gt_tracks": n_total,
        "n_matched": n_matched,
        "n_successes": cat_counts["Full Success"],
        "n_failures": n_failures,
        "match_rate": round(n_matched / n_total, 4),
        "success_rate_overall": round(cat_counts["Full Success"] / n_total, 4),
        "success_rate_matched": round(cat_counts["Full Success"] / max(n_matched, 1), 4),
        "overall_counts": {cat: cat_counts[cat] for cat in CATEGORIES},
        "error_attribution": error_attribution,
        "by_action_class": class_breakdown,
        "by_sequence": seq_breakdown,
        "per_track_results": results,
        "finding": finding,
    }
    
    out_path = os.path.join(OUT_DIR, "error_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved: {out_path}")
    print(f"\n{'='*70}")
    print(f"  FINDING: {finding}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
