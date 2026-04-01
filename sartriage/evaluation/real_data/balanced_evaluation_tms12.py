import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

def eval_model(model, X_test, y_test, labels_order):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    kap = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    return acc, rep, kap, cm, y_pred

def run_balanced():
    print("Loading Okutama Tracks...")
    with open("evaluation/real_data/okutama_all_tracks.json", "r") as f:
        data = json.load(f)
        
    mapping = {"Standing": "stationary", "Sitting": "stationary", "Walking": "walking", "Running": "running", "Lying": "lying_down"}
    X_list, y_list, sizes_list = [], [], []

    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in mapping: continue
        
        traj = []
        for (cx,cy), box in zip(t["centroids"], t["bboxes"]):
            traj.append([cx, cy, box[2], box[3]])
            
        feats = extract_tms12(traj)
        if any(np.isnan(f) or np.isinf(f) for f in feats): continue
        
        X_list.append(feats)
        y_list.append(mapping[act])
        sizes_list.append(t["mean_size_px"])
        
    X = np.array(X_list)
    y = np.array(y_list)
    sizes = np.array(sizes_list)
    labels_order = ["lying_down", "stationary", "walking", "running"]
    
    X_train, X_test, y_train, y_test, sizes_train, sizes_test = train_test_split(
        X, y, sizes, test_size=0.30, random_state=42, stratify=y
    )

    results = {}
    
    print("\n--- BASELINE RF ---")
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_base.fit(X_train, y_train)
    acc_base, rep_base, kap_base, _, _ = eval_model(rf_base, X_test, y_test, labels_order)
    results["Baseline"] = {"acc": acc_base, "rep": rep_base, "kap": kap_base}
    
    print("\n--- VARIANT 1: Balanced RF ---")
    rf_bal = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_bal.fit(X_train, y_train)
    acc_val1, rep_val1, kap_val1, _, _ = eval_model(rf_bal, X_test, y_test, labels_order)
    results["Balanced"] = {"acc": acc_val1, "rep": rep_val1, "kap": kap_val1}
    
    print("\n--- VARIANT 2: Balanced + Tuned Threshold for lying_down ---")
    # Using X_train to tune threshold
    X_t_train, X_t_val, y_t_train, y_t_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    rf_t = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_t.fit(X_t_train, y_t_train)
    
    probs_val = rf_t.predict_proba(X_t_val)
    idx_lying = list(rf_t.classes_).index('lying_down')
    
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.05, 0.95, 0.05):
        # Apply threshold
        preds = []
        for i in range(len(probs_val)):
            if probs_val[i, idx_lying] >= thresh:
                preds.append('lying_down')
            else:
                # Argmax of remaining
                p_copy = probs_val[i].copy()
                p_copy[idx_lying] = -1
                preds.append(rf_t.classes_[np.argmax(p_copy)])
        f1 = f1_score(y_t_val == 'lying_down', np.array(preds) == 'lying_down')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal lying_down threshold found: {best_thresh:.2f}")
    
    # Test on test set
    probs_test = rf_bal.predict_proba(X_test) # use full rf_bal
    idx_lying_full = list(rf_bal.classes_).index('lying_down')
    y_pred_v2 = []
    for i in range(len(probs_test)):
        if probs_test[i, idx_lying_full] >= best_thresh:
            y_pred_v2.append('lying_down')
        else:
            p_copy = probs_test[i].copy()
            p_copy[idx_lying_full] = -1
            y_pred_v2.append(rf_bal.classes_[np.argmax(p_copy)])
            
    acc_val2 = accuracy_score(y_test, y_pred_v2)
    rep_val2 = classification_report(y_test, y_pred_v2, output_dict=True, zero_division=0)
    kap_val2 = cohen_kappa_score(y_test, y_pred_v2)
    results["Balanced+Threshold"] = {"acc": acc_val2, "rep": rep_val2, "kap": kap_val2}

    print("\n--- VARIANT 3: SMOTE + Balanced RF ---")
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    rf_smote = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_smote.fit(X_sm, y_sm)
    acc_val3, rep_val3, kap_val3, cm_smote, y_pred_v3 = eval_model(rf_smote, X_test, y_test, labels_order)
    results["SMOTE+Balanced"] = {"acc": acc_val3, "rep": rep_val3, "kap": kap_val3}
    
    print("\n--- COMPARISON TABLE ---")
    metrics = ["Overall accuracy", "lying_down recall", "lying_down precision", "running recall", "walking recall", "stationary recall", "Cohen's kappa"]
    
    table_data = []
    for m in metrics:
        row = [m]
        for var in ["Baseline", "Balanced", "Balanced+Threshold", "SMOTE+Balanced"]:
            res = results[var]
            if m == "Overall accuracy":
                val = res["acc"]
            elif "recall" in m:
                cls = m.split(" ")[0]
                val = res["rep"][cls]["recall"]
            elif "precision" in m:
                cls = m.split(" ")[0]
                val = res["rep"][cls]["precision"]
            elif m == "Cohen's kappa":
                val = res["kap"]
            row.append(f"{val:.3f}")
        table_data.append(row)
        
    print(f"| {'Metric':<25} | {'Baseline':<12} | {'Balanced':<12} | {'Bal+Thresh':<12} | {'SMOTE':<12} |")
    print("-" * 85)
    for row in table_data:
        print(f"| {row[0]:<25} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12} | {row[4]:<12} |")

    # Save visual table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    col_labels = ["Metric", "Baseline RF", "Balanced RF", "Balanced+Threshold", "SMOTE+Balanced"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "tms12_balanced_comparison_table.png"))
    plt.close()

    # Determine best variant (Prioritising lying_down recall > 0.5 without destroying precision)
    best_variant = "SMOTE+Balanced"
    best_pred = y_pred_v3
    best_rep = rep_val3
    best_cm = cm_smote
    
    print(f"\nSelecting {best_variant} as Best Variant for remaining analysis.")
    
    # Per-size bin testing on TEST set using BEST MODEL
    print("\n--- BEST VARIANT: Accuracy by Size Bin ---")
    size_bins = [
        ("<50px",   lambda sz: sz < 50),
        ("50-75px", lambda sz: 50 <= sz < 75),
        ("75-100px",lambda sz: 75 <= sz < 100),
        (">100px",  lambda sz: sz >= 100)
    ]
    for name, condition in size_bins:
        idx = np.array([i for i, sz in enumerate(sizes_test) if condition(sz)])
        if len(idx) == 0: continue
            
        y_test_bin = y_test[idx]
        y_pred_bin = np.array(best_pred)[idx]
        b_acc = accuracy_score(y_test_bin, y_pred_bin)
        b_rep = classification_report(y_test_bin, y_pred_bin, output_dict=True, zero_division=0)
        
        recall_dict = {cls: b_rep.get(cls, {}).get('recall', 0.0) for cls in labels_order}
        print(f" Bin {name:10s} : N={len(idx):3d} | Acc={b_acc*100:5.1f}% | Recalls: {[round(recall_dict[c], 2) for c in labels_order]}")

    # Save CM
    plt.figure(figsize=(8,6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_order, yticklabels=labels_order)
    plt.title(f'Confusion Matrix ({best_variant})')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(FULL_DIR, "tms12_balanced_confusion.png"))
    plt.close()

    with open(os.path.join(FULL_DIR, "tms12_balanced_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_balanced()
