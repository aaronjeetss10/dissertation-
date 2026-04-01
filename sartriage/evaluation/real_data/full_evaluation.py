import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

def wilson_ci(p, n, z=1.96):
    denominator = 1 + z**2/n
    center = p + z**2/(2*n)
    rad = z * math.sqrt( (p*(1-p))/n + z**2/(4*n**2) )
    return (center - rad)/denominator, (center + rad)/denominator

def run_full():
    print("SETUP: Loading and Filtering Okutama Dataset")
    with open("evaluation/real_data/okutama_all_tracks.json", "r") as f:
        data = json.load(f)
        
    valid_tracks = []
    
    # Mapping
    mapping = {
        "Standing": "stationary",
        "Sitting": "stationary",
        "Walking": "walking",
        "Running": "running",
        "Lying": "lying_down"
    }
    
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        
        act = t["primary_action"]
        if act not in mapping: continue
        mapped = mapping[act]
             
        traj = []
        for (cx,cy), box in zip(t["centroids"], t["bboxes"]):
            traj.append([cx, cy, box[2], box[3]])
            
        valid_tracks.append({
            "id": tid,
            "trajectory": traj,
            "label": mapped,
            "mean_size_px": t["mean_size_px"]
        })
        
    y_raw = [t["label"] for t in valid_tracks]
    u, c = np.unique(y_raw, return_counts=True)
    dist = dict(zip(u, c))
    print(f"Final track counts after filtering (>=20 frames): {dist}")
    
    print("\nEXPERIMENT 1: Feature Extraction")
    X, y, sizes = [], [], []
    valid_tids = []
    
    nan_inf_count = 0
    for t in valid_tracks:
        features = extract_tms12(t["trajectory"])
        # Check Nan or Inf
        if any(math.isnan(f) or math.isinf(f) for f in features):
            nan_inf_count += 1
            print(f"WARNING: NaN/Inf generated for track {t['id']}")
            continue
            
        X.append(features)
        y.append(t["label"])
        sizes.append(t["mean_size_px"])
        valid_tids.append(t["id"])
        
    print(f"Extraction dropped {nan_inf_count} tracks due to math discontinuity.")
    X = np.array(X)
    y = np.array(y)
    sizes = np.array(sizes)
    
    baseline = max(c) / len(y_raw)
    
    X_train, X_test, y_train, y_test, sizes_train, sizes_test = train_test_split(
        X, y, sizes, test_size=0.30, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    ci_low, ci_high = wilson_ci(acc, len(y_test))
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print("\n--- EXPERIMENT 1 REPORT ---")
    print(f"Majority Baseline: {baseline*100:.2f}%")
    print(f"Overall Accuracy : {acc*100:.2f}%  (95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%])")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    labels_order = ["lying_down", "stationary", "walking", "running"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    print("Confusion Matrix (Counts):")
    print(cm)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("\nConfusion Matrix (Percentages):")
    print(np.round(cm_pct, 3))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order)
    plt.title('TMS-12 Real Dataset Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(FULL_DIR, "tms12_confusion.png"))
    plt.close()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
    print(f"\n5-fold Stratified CV Accuracy: {np.mean(cv_scores)*100:.2f}% +/- {np.std(cv_scores)*100:.2f}%")

    print("\n--- EXPERIMENT 2: Accuracy by Person Size Bin (Test Set) ---")
    size_bins = [
        ("<50px",   lambda sz: sz < 50),
        ("50-75px", lambda sz: 50 <= sz < 75),
        ("75-100px",lambda sz: 75 <= sz < 100),
        (">100px",  lambda sz: sz >= 100)
    ]
    
    bin_results = {}
    for name, condition in size_bins:
        idx = np.array([i for i, sz in enumerate(sizes_test) if condition(sz)])
        if len(idx) == 0:
            print(f" Bin {name:10s} : N=0")
            continue
            
        y_test_bin = y_test[idx]
        y_pred_bin = y_pred[idx]
        b_acc = accuracy_score(y_test_bin, y_pred_bin)
        report = classification_report(y_test_bin, y_pred_bin, output_dict=True, zero_division=0)
        
        recall_dict = {cls: report.get(cls, {}).get('recall', 0.0) for cls in labels_order}
        bin_results[name] = {
             "n": len(idx),
             "accuracy": b_acc,
             "recalls": recall_dict
        }
        
        print(f" Bin {name:10s} : N={len(idx):3d} | Acc={b_acc*100:5.1f}% | Recalls: {np.round([recall_dict[c] for c in labels_order], 2)}")

    print("\n--- EXPERIMENT 3: Feature Importance ---")
    feature_names = [
        "Net displacement", "Mean speed", "Speed CV", "Max acceleration",
        "Vertical dominance", "Direction change rate", "Stationarity ratio", 
        "Aspect ratio change", "Speed decay", "Oscillation index", 
        "Mean aspect ratio", "Mean normalised size"
    ]
    impts = rf.feature_importances_
    sorted_idx = impts.argsort()[::-1]
    
    print("Gini Importances:")
    for i in sorted_idx:
        print(f"  {feature_names[i]:25s}: {impts[i]:.4f}")
        
    perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    print("\nPermutation Importances (Test Set):")
    perm_idx = perm_imp.importances_mean.argsort()[::-1]
    for i in perm_idx:
        print(f"  {feature_names[i]:25s}: {perm_imp.importances_mean[i]:.4f} +/- {perm_imp.importances_std[i]:.4f}")
        
    plt.figure(figsize=(10,6))
    sns.barplot(x=impts[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")
    plt.title('Gini Feature Importance for TMS-12')
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "tms12_feature_importance.png"))
    plt.close()

    print("\n--- EXPERIMENT 4: Visualisation ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, alpha=0.6, palette="Set1")
    plt.title('PCA of Entire TMS-12 Okutama Dataset')
    plt.savefig(os.path.join(FULL_DIR, "tms12_pca.png"))
    plt.close()
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, alpha=0.6, palette="Set1")
    plt.title('t-SNE of Entire TMS-12 Okutama Dataset')
    plt.savefig(os.path.join(FULL_DIR, "tms12_tsne.png"))
    plt.close()
    
    out_json = os.path.join(FULL_DIR, "tms12_full_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "overall_test_accuracy": acc,
            "wilson_ci_95": [ci_low, ci_high],
            "cohens_kappa": kappa,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": cm.tolist(),
            "cv_accuracy_mean": np.mean(cv_scores),
            "cv_accuracy_std": np.std(cv_scores),
            "size_bins_test_set": bin_results,
            "gini_importance": dict(zip(feature_names, impts.tolist())),
            "permutation_importance": dict(zip(feature_names, perm_imp.importances_mean.tolist()))
        }, f, indent=2)
        
    print(f"\nSaved all artifacts thoroughly to {FULL_DIR}/")

if __name__ == "__main__":
    run_full()
