import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Make sure we can load the feature extractor
sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

PILOT_DIR = "evaluation/real_data/pilot"
os.makedirs(PILOT_DIR, exist_ok=True)

def tce_state_machine(traj):
    """
    Standalone replica of TCE (Track Context Engine) State Machine.
    Expects traj = list of [cx, cy, w, h]
    """
    if len(traj) < 2:
        return "UNKNOWN", 0.1
        
    cxs = [t[0] for t in traj]
    cys = [t[1] for t in traj]
    dxs = [cxs[i] - cxs[i-1] for i in range(1, len(traj))]
    dys = [cys[i] - cys[i-1] for i in range(1, len(traj))]
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    
    state = "MOVING"
    score = 0.1
    still_count = 0
    fast_count = 0
    
    for i, spd in enumerate(speeds):
        # Categorise speed
        if spd < 1.0: cat = "STILL"
        elif spd < 3.0: cat = "SLOW"
        else: cat = "FAST"
        
        # State transitions
        if cat == "STILL":
            still_count += 1
            if still_count > 15:
                if state == "MOVING":
                    # Sudden drop
                    state = "COLLAPSED"
                    score = 0.9
                elif state == "STOPPED":
                    state = "SUSTAINED_STILL"
                    score = 0.5
                elif state == "FAST_MOVING":
                    state = "COLLAPSED"
                    score = 0.95
            elif still_count > 5:
                if state != "COLLAPSED":
                    state = "STOPPED"
                    score = 0.3
        elif cat == "FAST":
            fast_count += 1
            still_count = 0
            if fast_count > 10:
                state = "FAST_MOVING"
                score = 0.2
            else:
                state = "MOVING"
                score = 0.1
        else:
            still_count = max(0, still_count - 1)
            if state in ["COLLAPSED", "SUSTAINED_STILL"]:
                score *= 0.9 # slowly decay if they start moving slow
            else:
                state = "MOVING"
                score = 0.1

    # Apply ceiling logic to score
    score = min(0.99, max(0.1, score))
    return state, score

def run_pilot():
    print("STEP 1: Load tracks & extracted TMS-12 features")
    with open("evaluation/real_data/okutama_all_tracks.json", "r") as f:
        data = json.load(f)
    
    valid_tracks = []
    y_labels = []
    
    print("STEP 2: Prepare class labels")
    for tid, t in data["tracks"].items():
        act = t["primary_action"]
        
        # Mapping
        if act in ["Standing", "Sitting"]:
             mapped = "stationary"
        elif act == "Walking":
             mapped = "walking"
        elif act == "Lying":
             mapped = "lying_down"
        else:
             # Drop running or others
             continue
             
        # Extract features
        traj = []
        for (cx,cy), box in zip(t["centroids"], t["bboxes"]):
            traj.append([cx, cy, box[2], box[3]])
            
        features = extract_tms12(traj)
        
        valid_tracks.append({
            "id": tid,
            "trajectory": traj,
            "features": features,
            "label": mapped
        })
        y_labels.append(mapped)

    X = np.array([t["features"] for t in valid_tracks])
    y = np.array(y_labels)
    
    print(f"Total Valid Tracks: {len(X)}")
    unique, counts = np.unique(y, return_counts=True)
    baseline_acc = dict(zip(unique, counts)).get('stationary', 0) / len(y)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"CRITICAL CHECK - Majority Class Baseline: {baseline_acc*100:.1f}%\n")
    
    print("STEP 3: LOOCV Random Forest")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    loo = LeaveOneOut()
    
    y_true = []
    y_pred = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        y_true.append(y_test[0])
        y_pred.append(pred[0])
        
    acc = accuracy_score(y_true, y_pred)
    print(f"LOOCV Overall Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, output_dict=False)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred, labels=['stationary', 'walking', 'lying_down'])
    print("Confusion Matrix (stationary, walking, lying_down):")
    print(cm)
    
    # Save Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['stationary', 'walking', 'lying_down'], yticklabels=['stationary', 'walking', 'lying_down'])
    plt.title('TMS-12 Confusion Matrix (LOOCV)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(PILOT_DIR, "tms12_confusion_pilot.png"))
    plt.close()
    
    print("\nSTEP 4: Feature Importances")
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_final.fit(X, y)
    
    feature_names = [
        "Net displacement", "Mean speed", "Speed CV", "Max acceleration",
        "Vertical dominance", "Direction change rate", "Stationarity ratio", 
        "Aspect ratio change", "Speed decay", "Oscillation index", 
        "Mean aspect ratio", "Mean normalised size"
    ]
    
    importances = rf_final.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    print("Gini Feature Importance (All data):")
    for i in sorted_idx:
        print(f"  {feature_names[i]:25s}: {importances[i]:.4f}")
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf_perm = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    perm_imp = permutation_importance(rf_perm, X_val, y_val, n_repeats=10, random_state=42)
    
    print("\nPermutation Importance (30% Hold-out):")
    perm_sorted_idx = perm_imp.importances_mean.argsort()[::-1]
    for i in perm_sorted_idx:
        print(f"  {feature_names[i]:25s}: {perm_imp.importances_mean[i]:.4f} +/- {perm_imp.importances_std[i]:.4f}")

    print("\nSTEP 5: Feature Sanity Visualisation")
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls, alpha=0.7)
    plt.legend()
    plt.title('PCA of TMS-12 Features')
    plt.savefig(os.path.join(PILOT_DIR, "tms12_pca_pilot.png"))
    plt.close()
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=cls, alpha=0.7)
    plt.legend()
    plt.title('t-SNE of TMS-12 Features')
    plt.savefig(os.path.join(PILOT_DIR, "tms12_tsne_pilot.png"))
    plt.close()
    
    # Boxplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, feat_name in enumerate(feature_names):
        sns.boxplot(x=y, y=X[:, i], ax=axes[i], order=['stationary', 'walking', 'lying_down'])
        axes[i].set_title(feat_name)
    plt.tight_layout()
    plt.savefig(os.path.join(PILOT_DIR, "tms12_features_boxplot_pilot.png"))
    plt.close()
    
    with open(os.path.join(PILOT_DIR, "tms12_pilot_results.json"), "w") as f:
        json.dump({
            "loocv_accuracy": acc,
            "report": classification_report(y_true, y_pred, output_dict=True),
            "gini_importance": dict(zip(feature_names, importances.tolist())),
            "permutation_importance": dict(zip(feature_names, perm_imp.importances_mean.tolist()))
        }, f, indent=2)

    print("\nSTEP 6: TCE State Machine Sanity Check")
    tce_results = []
    class_tce_scores = defaultdict(list)
    
    for trk in valid_tracks:
        st, score = tce_state_machine(trk["trajectory"])
        cls = trk["label"]
        class_tce_scores[cls].append(score)
        tce_results.append({
            "id": trk["id"],
            "label": cls,
            "final_state": st,
            "final_score": round(score, 4)
        })
        
    print("Mean TCE Priority Score per Class:")
    for cls in ['lying_down', 'stationary', 'walking']:
        scores = class_tce_scores[cls]
        print(f"  {cls:15s}: {np.mean(scores):.4f}  (Min: {np.min(scores):.2f}, Max: {np.max(scores):.2f})")
        
    with open(os.path.join(PILOT_DIR, "tce_pilot_results.json"), "w") as f:
        json.dump({
            "class_means": {c: float(np.mean(s)) for c, s in class_tce_scores.items()},
            "tracks": tce_results
        }, f, indent=2)
        
    print("\nSaved all pilot artifacts to evaluation/real_data/pilot/")

if __name__ == "__main__":
    run_pilot()
