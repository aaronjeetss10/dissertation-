import os
import sys
import json
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

# Copied from TCE v2 pilot
def assess_initial_state(track_first_n_frames, speed_threshold=40.0):
    if len(track_first_n_frames) < 2: return "UNKNOWN", 0.1
        
    cxs = [t[0] for t in track_first_n_frames]
    cys = [t[1] for t in track_first_n_frames]
    ws = [t[2] for t in track_first_n_frames]
    hs = [t[3] for t in track_first_n_frames]
    
    dxs = [cxs[i] - cxs[i-1] for i in range(1, len(track_first_n_frames))]
    dys = [cys[i] - cys[i-1] for i in range(1, len(track_first_n_frames))]
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    
    initial_speed = np.mean(speeds) if speeds else 0.0
    initial_stationarity = sum(1 for s in speeds if s < speed_threshold) / len(speeds) if speeds else 1.0
    
    ars = [h / (w + 1e-8) for w, h in zip(ws, hs)]
    initial_aspect_ratio = np.mean(ars)
    
    if initial_stationarity > 0.8 and initial_aspect_ratio < 0.6:  # W > H
        return "CRITICAL_STATIC", 0.8
    elif initial_stationarity > 0.8:
        return "SUSTAINED_STILL", 0.5
    elif initial_speed > speed_threshold:
        return "MOVING_FAST", 0.3
    else:
        return "MOVING_SLOW", 0.2

def tce_state_machine_v2(traj, speed_thresh=40.0):
    if len(traj) < 2: return "UNKNOWN", 0.1
    
    track_first_n = traj[:10]
    state, score = assess_initial_state(track_first_n, speed_thresh)
    
    cxs = [t[0] for t in traj]
    cys = [t[1] for t in traj]
    dxs = [cxs[i] - cxs[i-1] for i in range(1, len(traj))]
    dys = [cys[i] - cys[i-1] for i in range(1, len(traj))]
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    
    still_count = 0
    fast_count = 0
    
    for i, spd in enumerate(speeds):
        if spd < speed_thresh: cat = "STILL"
        elif spd < speed_thresh * 1.5: cat = "SLOW"
        else: cat = "FAST"
        
        if cat == "STILL":
            still_count += 1
            if still_count > 15:
                if state in ["MOVING_SLOW", "MOVING_FAST"]:
                    state = "COLLAPSED"
                    score = 0.9
                elif state == "STOPPED":
                    state = "SUSTAINED_STILL"
                    score = max(0.5, score)
                elif state == "CRITICAL_STATIC":
                    state = "CRITICAL_STATIC"
                    score = max(0.8, score)
            elif still_count > 5:
                if state not in ["COLLAPSED", "CRITICAL_STATIC", "SUSTAINED_STILL"]:
                    state = "STOPPED"
                    score = 0.3
        elif cat == "FAST":
            fast_count += 1
            still_count = 0
            if fast_count > 10:
                state = "FAST_MOVING"
                score = 0.2
            else:
                if state not in ["CRITICAL_STATIC", "COLLAPSED"]:
                    state = "MOVING_SLOW"
                    score = 0.1
        else:
            still_count = max(0, still_count - 1)
            # No persistent decay on CRITICAL_STATIC for small wiggles
            if state not in ["COLLAPSED", "CRITICAL_STATIC", "SUSTAINED_STILL"]:
                state = "MOVING_SLOW"
                score = 0.1

    score = min(0.99, max(0.1, score))
    return score

def wilson_ci(x, confidence=0.95):
    # Standard Wald CI since we compute mean of scores
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    n = len(x)
    cl = 1.96 * (std / math.sqrt(n)) if n > 0 else 0
    return mean - cl, mean + cl

def run_f():
    print("STEP 1: Load tracks")
    with open("evaluation/real_data/okutama_all_tracks.json", "r") as f:
        data = json.load(f)
        
    valid_tracks = []
    # Relevant classes
    mapping = {"Standing": "standing", "Sitting": "sitting", "Walking": "walking", "Running": "running", "Lying": "lying_down"}
    
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in mapping: continue
        
        traj = []
        for (cx,cy), box in zip(t["centroids"], t["bboxes"]):
            traj.append([cx, cy, box[2], box[3]])
            
        tce_score = tce_state_machine_v2(traj)
        
        valid_tracks.append({
            "id": tid,
            "cat": mapping[act],
            "tce": tce_score
        })
        
    # Stats
    cat_scores = {"lying_down": [], "running": [], "walking": [], "standing": [], "sitting": []}
    aggregate_stat = []
    
    for t in valid_tracks:
        cat = t["cat"]
        score = t["tce"]
        cat_scores[cat].append(score)
        if cat in ["standing", "sitting"]:
            aggregate_stat.append(score)
            
    print("EXPERIMENT 1: Final Priority Score Distributions\n")
    print(f"lying_down ({len(cat_scores['lying_down'])} tracks): {np.mean(cat_scores['lying_down']):.4f} +/- {np.std(cat_scores['lying_down']):.4f}")
    if aggregate_stat:
        print(f"stationary ({len(aggregate_stat)} tracks): {np.mean(aggregate_stat):.4f} +/- {np.std(aggregate_stat):.4f}")
    print(f"running    ({len(cat_scores['running'])} tracks): {np.mean(cat_scores['running']):.4f} +/- {np.std(cat_scores['running']):.4f}")
    print(f"walking    ({len(cat_scores['walking'])} tracks): {np.mean(cat_scores['walking']):.4f} +/- {np.std(cat_scores['walking']):.4f}")
    print(f"  > standing ({len(cat_scores['standing'])} tracks): {np.mean(cat_scores['standing']):.4f} +/- {np.std(cat_scores['standing']):.4f}")
    print(f"  > sitting  ({len(cat_scores['sitting'])} tracks): {np.mean(cat_scores['sitting']):.4f} +/- {np.std(cat_scores['sitting']):.4f}")

    print("\nEXPERIMENT 2: Ranking Evaluation")
    # Ground truth
    gt_map = {"lying_down": 3, "sitting": 2, "standing": 1, "running": 1, "walking": 0}
    np.random.seed(42)
    
    ndcg_results = {"FLAT": {3:[], 5:[], 10:[]}, "RANDOM": {3:[], 5:[], 10:[]}, "TCE": {3:[], 5:[], 10:[]}}
    recall_lying = {"FLAT": [], "RANDOM": [], "TCE": []}
    
    # We want 50 scenes of 20 tracks. Some scenes must have lying_down to measure recall.
    # To strictly follow proportional mixing without replacement per scene
    tracks_array = np.array(valid_tracks)
    
    for _ in range(50):
        scene_tracks = np.random.choice(tracks_array, 20, replace=False)
        y_true = [gt_map[t["cat"]] for t in scene_tracks]
        
        y_flat = [0.5 for _ in scene_tracks]
        y_rand = np.random.rand(20)
        y_tce = [t["tce"] for t in scene_tracks]
        
        # small noise to flat to break ties uniformly
        y_flat_noisy = np.array(y_flat) + np.random.uniform(0, 1e-5, 20)
        
        # NDCG
        for k in [3, 5, 10]:
            ndcg_results["FLAT"][k].append(ndcg_score([y_true], [y_flat_noisy], k=k))
            ndcg_results["RANDOM"][k].append(ndcg_score([y_true], [y_rand], k=k))
            ndcg_results["TCE"][k].append(ndcg_score([y_true], [y_tce], k=k))
            
        # Recall@3 for lying
        has_lying = any(t == 3 for t in y_true)
        if has_lying:
            def check_top3(scores):
                # get indices of top 3 scores
                top3_idx = np.argsort(scores)[::-1][:3]
                # is any top 3 a lying down?
                return 1.0 if any(y_true[idx] == 3 for idx in top3_idx) else 0.0
                
            recall_lying["FLAT"].append(check_top3(y_flat_noisy))
            recall_lying["RANDOM"].append(check_top3(y_rand))
            recall_lying["TCE"].append(check_top3(y_tce))
            
    print("NDCG Results (Mean +/- 95% CI):")
    pvals = {}
    for k in [3, 5, 10]:
        print(f" NDCG@{k}:")
        for method in ["FLAT", "RANDOM", "TCE"]:
            res = ndcg_results[method][k]
            low, high = wilson_ci(res)
            print(f"   {method:6s}: {np.mean(res):.4f} [{low:.4f}, {high:.4f}]")
            
        # Paired Wilcoxon
        w_stat, p = stats.wilcoxon(ndcg_results["TCE"][k], ndcg_results["FLAT"][k])
        print(f"   Wilcoxon TCE vs FLAT (k={k}) p-value: {p:.4e}")
        pvals[k] = p

    print("\nEXPERIMENT 3: Lying Detection Rate in Top-K (Recall@3)")
    print(f"Evaluated on {len(recall_lying['TCE'])}/50 scenes containing at least 1 casualty.")
    for method in ["FLAT", "RANDOM", "TCE"]:
        r_mean = np.mean(recall_lying[method])
        print(f" {method:6s}: {r_mean*100:.1f}%")

    # SAVE TO DISK
    with open(os.path.join(FULL_DIR, "tce_v2_full_results.json"), "w") as f:
        json.dump({
            "means_std": {
                "lying_down": [np.mean(cat_scores['lying_down']), np.std(cat_scores['lying_down'])],
                "running": [np.mean(cat_scores['running']), np.std(cat_scores['running'])],
                "walking": [np.mean(cat_scores['walking']), np.std(cat_scores['walking'])],
                "stationary": [np.mean(aggregate_stat), np.std(aggregate_stat)],
                "standing": [np.mean(cat_scores['standing']), np.std(cat_scores['standing'])],
                "sitting": [np.mean(cat_scores['sitting']), np.std(cat_scores['sitting'])]
            },
            "ndcg_results": {m: {str(k): np.mean(v) for k, v in res.items()} for m, res in ndcg_results.items()},
            "recall_lying": {m: np.mean(v) for m, v in recall_lying.items()}
        }, f, indent=2)
        
    # Plot 1: NDCG 
    labels = ["NDCG@3", "NDCG@5", "NDCG@10"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,6))
    
    means_flat = [np.mean(ndcg_results["FLAT"][k]) for k in [3,5,10]]
    means_rand = [np.mean(ndcg_results["RANDOM"][k]) for k in [3,5,10]]
    means_tce = [np.mean(ndcg_results["TCE"][k]) for k in [3,5,10]]
    
    ax.bar(x - width, means_flat, width, label='FLAT', color='grey')
    ax.bar(x, means_rand, width, label='RANDOM', color='orange')
    ax.bar(x + width, means_tce, width, label='TCE v2', color='red')
    
    ax.set_ylabel('NDCG Score')
    ax.set_title('NDCG by Ranking Method (50 Simulated Scenes)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "tce_ranking_comparison.png"))
    plt.close()
    
    # Plot 2: Lying Recall
    fig, ax = plt.subplots(figsize=(6,5))
    mets = ["FLAT", "RANDOM", "TCE"]
    vals = [np.mean(recall_lying[m]) for m in mets]
    ax.bar(mets, vals, color=['grey', 'orange', 'red'])
    ax.set_ylabel('Recall@3 (Casualty Found)')
    ax.set_title('Recall@3 for lying_down tracks')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "tce_lying_topk.png"))
    plt.close()
    
if __name__ == "__main__":
    run_f()
