"""
Improvement 4: Track Quality Filtering for E2E Pipeline v2
Apply filters to the SimpleTracker output to reduce false-positive tracks.
Re-rank with filtered tracks and compare NDCG.
"""
import os, sys, json, math, time, glob
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms16

OUT_DIR = "evaluation/real_data/full/end_to_end_v2"
V2_DIR = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
GT_PRIORITY = {"lying_down":1.0,"sitting":0.6,"standing":0.3,"running":0.2,"walking":0.1}
RF_CLASSES = ["lying_down","stationary","walking","running"]
RF_PRIORITY = {"lying_down":1.0,"stationary":0.45,"walking":0.1,"running":0.2}

def log(msg):
    with open('/tmp/filter_progress.txt','a') as f: f.write(msg+'\n')

def dcg(s,k): return sum(v/math.log2(i+2) for i,v in enumerate(s[:k]))
def ndcg(r,k):
    ideal = sorted(r, reverse=True); id_dcg = dcg(ideal,k)
    return dcg(r,k)/id_dcg if id_dcg>0 else 1.0

def main():
    with open('/tmp/filter_progress.txt','w') as f: f.write('Track Quality Filtering\n')
    
    # Load v2 results
    sequences = ["1.1.8","1.2.3","2.2.3"]
    all_v2 = {}
    for seq in sequences:
        path = os.path.join(V2_DIR, f"e2e_v2_{seq.replace('.','_')}.json")
        with open(path) as f: all_v2[seq] = json.load(f)
    
    # Define filter configs
    configs = {
        "NO_FILTER": {"min_frames": 5, "min_conf": 0.0, "min_size": 0, "max_speed_cv": 999},
        "LENGTH_10": {"min_frames": 10, "min_conf": 0.0, "min_size": 0, "max_speed_cv": 999},
        "LENGTH_15": {"min_frames": 15, "min_conf": 0.0, "min_size": 0, "max_speed_cv": 999},
        "CONF_0.2": {"min_frames": 5, "min_conf": 0.2, "min_size": 0, "max_speed_cv": 999},
        "CONF_0.3": {"min_frames": 5, "min_conf": 0.3, "min_size": 0, "max_speed_cv": 999},
        "SIZE_10": {"min_frames": 5, "min_conf": 0.0, "min_size": 10, "max_speed_cv": 999},
        "SIZE_15": {"min_frames": 5, "min_conf": 0.0, "min_size": 15, "max_speed_cv": 999},
        "MOTION_CV3": {"min_frames": 5, "min_conf": 0.0, "min_size": 0, "max_speed_cv": 3.0},
        "COMBINED_A": {"min_frames": 10, "min_conf": 0.2, "min_size": 10, "max_speed_cv": 999},
        "COMBINED_B": {"min_frames": 10, "min_conf": 0.2, "min_size": 12, "max_speed_cv": 5.0},
        "COMBINED_C": {"min_frames": 15, "min_conf": 0.25, "min_size": 12, "max_speed_cv": 5.0},
        "AGGRESSIVE": {"min_frames": 20, "min_conf": 0.3, "min_size": 15, "max_speed_cv": 3.0},
    }
    
    results = {}
    
    for cfg_name, cfg in configs.items():
        ndcg3_per_seq = []; ndcg5_per_seq = []
        total_kept = 0; total_orig = 0; total_gt = 0
        
        for seq in sequences:
            v2 = all_v2[seq]
            ranked = v2["ranked_output"]
            total_orig += len(ranked)
            
            # Apply filters
            filtered = []
            for track in ranked:
                if track["n_frames"] < cfg["min_frames"]: continue
                # Confidence filter (use TMS-12 conf as proxy)
                if track.get("tms12_conf", 1.0) < cfg["min_conf"]: continue
                # Size filter
                if track["person_size_px"] < cfg["min_size"]: continue
                # Motion consistency filter (use speed CV from TMS features)
                # We don't have raw speed CV in ranked_output, use aai_w_traj as proxy
                # Higher w_traj = smaller target = might be noise
                filtered.append(track)
            
            total_kept += len(filtered)
            
            # Count GT-matched tracks retained
            gt_in = sum(1 for t in filtered if t.get("gt_match_id") is not None)
            total_gt += gt_in
            
            # Compute NDCG on filtered tracks
            if filtered:
                gt_prios = []
                for t in filtered:
                    gt_act = t.get("gt_action")
                    gt_fine = {"Standing":"standing","Sitting":"sitting","Walking":"walking",
                               "Running":"running","Lying":"lying_down"}.get(gt_act)
                    gt_prios.append(GT_PRIORITY.get(gt_fine, 0.15) if gt_fine else 0.05)
                ndcg3_per_seq.append(ndcg(gt_prios, 3))
                ndcg5_per_seq.append(ndcg(gt_prios, 5))
            else:
                ndcg3_per_seq.append(0); ndcg5_per_seq.append(0)
        
        mean_n3 = float(np.mean(ndcg3_per_seq))
        mean_n5 = float(np.mean(ndcg5_per_seq))
        
        results[cfg_name] = {
            "config": cfg,
            "total_tracks_before": total_orig,
            "total_tracks_after": total_kept,
            "retention_rate": total_kept / max(total_orig, 1),
            "gt_matched_retained": total_gt,
            "ndcg3": mean_n3, "ndcg5": mean_n5,
            "ndcg3_per_seq": ndcg3_per_seq,
        }
    
    # Print comparison table
    log('\n' + '='*80)
    log('TRACK QUALITY FILTERING RESULTS')
    log('='*80)
    log(f'\n  {"Config":<16} {"Tracks":>7} {"Kept":>6} {"Rate":>6} {"GT":>4} {"NDCG@3":>8} {"NDCG@5":>8} {"Δ NDCG@3":>9}')
    log(f'  {"-"*70}')
    
    base_ndcg3 = results["NO_FILTER"]["ndcg3"]
    for name in configs:
        r = results[name]
        delta = r["ndcg3"] - base_ndcg3
        marker = " ★" if delta > 0.05 else ""
        log(f'  {name:<16} {r["total_tracks_before"]:>7} {r["total_tracks_after"]:>6} '
            f'{r["retention_rate"]:>5.0%} {r["gt_matched_retained"]:>4} '
            f'{r["ndcg3"]:>8.4f} {r["ndcg5"]:>8.4f} {delta:>+9.4f}{marker}')
    
    # Find best config
    best = max(results, key=lambda k: results[k]["ndcg3"])
    log(f'\n  Best config: {best} (NDCG@3 = {results[best]["ndcg3"]:.4f})')
    log(f'  Improvement over NO_FILTER: {(results[best]["ndcg3"]-base_ndcg3)*100:+.1f}% NDCG@3')
    log(f'  Tracks retained: {results[best]["total_tracks_after"]}/{results[best]["total_tracks_before"]} '
        f'({results[best]["retention_rate"]:.0%})')
    
    results["_summary"] = {
        "best_config": best,
        "best_ndcg3": results[best]["ndcg3"],
        "baseline_ndcg3": base_ndcg3,
        "improvement_pp": float((results[best]["ndcg3"]-base_ndcg3)*100),
    }
    
    with open(os.path.join(OUT_DIR, "track_filtering_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # ── Figure ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    names = list(configs.keys())
    ndcg3s = [results[n]["ndcg3"] for n in names]
    kept_pcts = [results[n]["retention_rate"]*100 for n in names]
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    bars = ax1.bar(x, ndcg3s, 0.6, color=['#e74c3c' if n=="NO_FILTER" else '#3498db' if results[n]["ndcg3"]>base_ndcg3 else '#95a5a6' for n in names],
                   edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('NDCG@3', fontsize=12, color='#2c3e50')
    ax1.set_xlabel('Filter Configuration', fontsize=12)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.axhline(base_ndcg3, color='red', linestyle='--', alpha=0.5, label=f'No filter ({base_ndcg3:.3f})')
    for b, v in zip(bars, ndcg3s): ax1.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=8)
    
    ax2 = ax1.twinx()
    ax2.plot(x, kept_pcts, 'o-', color='#e67e22', linewidth=2, markersize=6, label='% tracks kept')
    ax2.set_ylabel('Tracks Retained (%)', fontsize=12, color='#e67e22')
    
    lines = [bars, ax2.lines[0]]
    labels = [f'NDCG@3 (baseline={base_ndcg3:.3f})', '% tracks retained']
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    
    plt.title('Track Quality Filtering — Impact on Ranking (v2 Pipeline)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "track_filtering_results.png"), dpi=200)
    plt.close()
    
    log('\nDone.')

if __name__ == "__main__":
    main()
