import os
import sys
import json
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

feature_names = [
    "Net displacement", "Mean speed", "Speed CV", "Max acceleration",
    "Vertical dominance", "Direction change rate", "Stationarity ratio", 
    "Aspect ratio change", "Speed decay", "Oscillation index", 
    "Mean aspect ratio", "Mean normalised size"
]

def map_bin(sz):
    if sz < 20: return "<20px"
    elif sz < 35: return "20-35px"
    elif sz < 50: return "35-50px"
    elif sz < 75: return "50-75px"
    else: return ">75px"

def run_visdrone_eval():
    print("STEP A: Load and Extract Tracking Data")
    with open("evaluation/real_data/visdrone_mot_tracks.json", "r") as f:
        vd_data = json.load(f)
        
    with open("evaluation/real_data/okutama_all_tracks.json", "r") as f:
        oku_data = json.load(f)
        
    vd_bins = {b: {"valid": [], "failed": 0, "features": []} for b in ["<20px", "20-35px", "35-50px", "50-75px", ">75px"]}
    
    for tid, t in vd_data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        sz_bin = map_bin(t["mean_size_px"])
        
        traj = []
        for (cx,cy), box in zip(t["centroids"], t["bboxes"]):
            traj.append([cx, cy, box[2], box[3]])
            
        try:
            feats = extract_tms12(traj)
            if any(math.isnan(f) or math.isinf(f) for f in feats):
                vd_bins[sz_bin]["failed"] += 1
            else:
                vd_bins[sz_bin]["valid"].append(tid)
                vd_bins[sz_bin]["features"].append(feats)
        except:
            vd_bins[sz_bin]["failed"] += 1
            
    print("\nSTEP B & C: Resolution Degeneration Analysis")
    for b in ["<20px", "20-35px", "35-50px", "50-75px", ">75px"]:
        v_count = len(vd_bins[b]["valid"])
        f_count = vd_bins[b]["failed"]
        total = v_count + f_count
        fail_rate = (f_count / total * 100) if total > 0 else 0
        
        print(f"[{b:10s}] Valid: {v_count:3d} | Failed: {f_count:3d} | Failure Rate: {fail_rate:5.1f}%")
        
    # Collate for JSON dump
    json_stats = {}
    for b in ["<20px", "20-35px", "35-50px", "50-75px", ">75px"]:
        if len(vd_bins[b]["features"]) == 0: continue
        f_matrix = np.array(vd_bins[b]["features"])
        means = np.mean(f_matrix, axis=0)
        stds = np.std(f_matrix, axis=0)
        
        json_stats[b] = {
            "valid": len(vd_bins[b]["features"]),
            "failed": vd_bins[b]["failed"],
            "features": {}
        }
        for i, fname in enumerate(feature_names):
            json_stats[b]["features"][fname] = {
                "mean": float(means[i]),
                "std": float(stds[i])
            }

    print("\nSTEP D: Okutama vs VisDrone Cross-Dataset Consistency (50-75px)")
    oku_50_75 = []
    for tid, t in oku_data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        if 50 <= t["mean_size_px"] < 75:
            traj = [[cx, cy, box[2], box[3]] for (cx,cy), box in zip(t["centroids"], t["bboxes"])]
            feats = extract_tms12(traj)
            if not any(math.isnan(f) or math.isinf(f) for f in feats):
                oku_50_75.append(feats)
                
    oku_50_75 = np.array(oku_50_75)
    vd_50_75 = np.array(vd_bins["50-75px"]["features"])
    
    print(f"Okutama 50-75px N={len(oku_50_75)} | VisDrone 50-75px N={len(vd_50_75)}")
    print("Kolmogorov-Smirnov Test (H0: distributions are identical):")
    ks_results = {}
    for i, fname in enumerate(feature_names):
        stat, pval = stats.ks_2samp(oku_50_75[:, i], vd_50_75[:, i])
        ks_results[fname] = {"stat": float(stat), "pval": float(pval)}
        sig = "*** (Reject H0)" if pval < 0.05 else "--- (Accept H0)"
        print(f"  {fname:25s} | KS Stat: {stat:.4f} | p-value: {pval:.4e} {sig}")

    # Save to disk
    with open(os.path.join(FULL_DIR, "tms12_visdrone_feature_quality.json"), "w") as f:
        json.dump({
            "bins": json_stats,
            "ks_test_50_75px": ks_results
        }, f, indent=2)

    # Plot feature vs scale
    valid_bins = [b for b in ["<20px", "20-35px", "35-50px", "50-75px", ">75px"] if len(vd_bins[b]["features"]) > 0]
    
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, fname in enumerate(feature_names):
        plot_data = []
        labels = []
        for b in valid_bins:
            vals = np.array(vd_bins[b]["features"])[:, i]
            plot_data.extend(vals)
            labels.extend([b]*len(vals))
            
        sns.boxplot(x=labels, y=plot_data, ax=axes[i], order=valid_bins, palette="muted", showfliers=False)
        axes[i].set_title(fname)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "tms12_feature_vs_scale.png"))
    plt.close()
    
    print(f"\nSaved analysis to {FULL_DIR}/")

if __name__ == "__main__":
    run_visdrone_eval()
