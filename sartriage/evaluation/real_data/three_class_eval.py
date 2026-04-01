"""
3-Class SAR-Aligned Classification: critical / upright_stationary / mobile
"""
import os, sys, json, math, time
import numpy as np
import torch
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms16
from evaluation.scte import SCTEModel

OUT_DIR = "evaluation/real_data/full"

# ── SAR 3-class mapping ──
# Okutama → 3-class
SAR3_MAP = {
    "Standing": "upright_stationary", "Sitting": "upright_stationary",
    "Walking": "mobile", "Running": "mobile",
    "Lying": "critical",
}
SAR3_CLASSES = ["critical", "upright_stationary", "mobile"]
SAR3_PRIORITY = {"critical": 1.0, "upright_stationary": 0.4, "mobile": 0.1}

# Old 4-class for comparison
SAR4_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
            "Running":"running","Lying":"lying_down"}
SAR4_CLASSES = ["lying_down","stationary","walking","running"]

def log(msg):
    with open('/tmp/three_class_progress.txt','a') as f: f.write(msg+'\n')

def dcg(s,k): return sum(v/math.log2(i+2) for i,v in enumerate(s[:k]))
def ndcg(r,k):
    ideal = sorted(r, reverse=True); d = dcg(ideal,k)
    return dcg(r,k)/d if d>0 else 1.0

def main():
    with open('/tmp/three_class_progress.txt','w') as f: f.write('3-Class SAR Classification\n')

    # Load data
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)

    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR3_MAP: continue
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        f16 = extract_tms16(traj)
        f16 = [0 if (math.isnan(f) or math.isinf(f)) else f for f in f16]

        centroids = t["centroids"]; bboxes = t["bboxes"]
        scte_tokens = []
        for i in range(1, len(centroids)):
            dx = (centroids[i][0]-centroids[i-1][0])/1280.0
            dy = (centroids[i][1]-centroids[i-1][1])/720.0
            aspect = bboxes[i][3]/max(bboxes[i][2],1)
            sn = math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            scte_tokens.append([dx,dy,aspect,sn])
        arr_s = np.zeros((50,4), dtype=np.float32)
        for j in range(min(len(scte_tokens),50)): arr_s[j] = scte_tokens[j]

        tracks.append({
            "tms16": np.array(f16, dtype=np.float32),
            "scte_tokens": arr_s,
            "label_3": SAR3_CLASSES.index(SAR3_MAP[act]),
            "label_4": SAR4_CLASSES.index(SAR4_MAP[act]),
            "gt_action": act, "tid": tid,
            "sar3_class": SAR3_MAP[act], "sar4_class": SAR4_MAP[act],
        })

    y3 = np.array([t["label_3"] for t in tracks])
    y4 = np.array([t["label_4"] for t in tracks])
    log(f'Tracks: {len(tracks)}')
    log(f'4-class dist: {Counter(y4.tolist())} → {dict(zip(SAR4_CLASSES, [sum(y4==i) for i in range(4)]))}')
    log(f'3-class dist: {Counter(y3.tolist())} → {dict(zip(SAR3_CLASSES, [sum(y3==i) for i in range(3)]))}')

    # Load SCTE
    scte = SCTEModel(input_dim=4, d_model=32, proj_dim=16, n_heads=2, n_layers=2, dropout=0.1, max_len=50)
    scte.load_state_dict(torch.load("evaluation/results/scte_encoder_trained.pt",
                                     map_location="cpu", weights_only=True))
    scte.eval()
    scte_arr = np.stack([t["scte_tokens"] for t in tracks])
    with torch.no_grad():
        embs = []
        for s in range(0, len(scte_arr), 64):
            embs.append(scte.get_embedding(torch.FloatTensor(scte_arr[s:s+64])).cpu().numpy())
    X_scte = np.concatenate(embs)  # (N, 32)

    X_tms = np.stack([t["tms16"] for t in tracks])  # (N, 16)
    X_48 = np.hstack([X_tms, X_scte])  # (N, 48)

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: 3-class vs 4-class with TMS-16
    # ══════════════════════════════════════════════════════════════════
    log('\n' + '='*70)
    log('EXPERIMENT 1: 3-class vs 4-class (5-fold CV)')
    log('='*70)

    configs = [
        ("TMS-16, 4-class", X_tms, y4, SAR4_CLASSES),
        ("TMS-16, 3-class", X_tms, y3, SAR3_CLASSES),
        ("TMS-16+SCTE, 4-class", X_48, y4, SAR4_CLASSES),
        ("TMS-16+SCTE, 3-class", X_48, y3, SAR3_CLASSES),
    ]

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, X, y, classes in configs:
        all_pred = []; all_true = []; fold_accs = []
        for tr_idx, te_idx in skf.split(X, y):
            try:
                sm = SMOTE(random_state=42)
                X_sm, y_sm = sm.fit_resample(X[tr_idx], y[tr_idx])
            except:
                X_sm, y_sm = X[tr_idx], y[tr_idx]
            rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            rf.fit(X_sm, y_sm)
            preds = rf.predict(X[te_idx])
            fold_accs.append(accuracy_score(y[te_idx], preds))
            all_pred.extend(preds); all_true.extend(y[te_idx])

        all_pred = np.array(all_pred); all_true = np.array(all_true)
        acc = np.mean(fold_accs); ci = 1.96*np.std(fold_accs)
        kappa = cohen_kappa_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(all_true, all_pred)

        # Critical class recall
        if "critical" in classes:
            crit_r = report["critical"]["recall"]
        else:
            crit_r = report["lying_down"]["recall"]

        results[name] = {
            "accuracy": float(acc), "accuracy_ci": float(ci),
            "kappa": float(kappa),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "critical_recall": float(crit_r),
            "per_class": {c: {"p":round(report[c]["precision"],3),
                              "r":round(report[c]["recall"],3),
                              "f1":round(report[c]["f1-score"],3),
                              "n":int(report[c]["support"])} for c in classes},
            "confusion": cm.tolist(), "classes": classes,
        }

        log(f'\n  {name}:')
        log(f'    Acc: {acc:.1%} ± {ci:.1%}  Kappa: {kappa:.3f}  Macro-F1: {report["macro avg"]["f1-score"]:.3f}')
        log(f'    Critical/lying recall: {crit_r:.1%}')
        for c in classes:
            log(f'    {c:<22} P={report[c]["precision"]:.3f} R={report[c]["recall"]:.3f} F1={report[c]["f1-score"]:.3f} n={report[c]["support"]}')

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Ranking with 3-class predictions
    # ══════════════════════════════════════════════════════════════════
    log('\n' + '='*70)
    log('EXPERIMENT 2: 3-class ranking (100 random scenes)')
    log('='*70)

    # Train final 3-class RF on full data
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_48, y3)
    rf3 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf3.fit(X_sm, y_sm)

    # Also train 4-class for comparison
    sm4 = SMOTE(random_state=42)
    X_sm4, y_sm4 = sm4.fit_resample(X_48, y4)
    rf4 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf4.fit(X_sm4, y_sm4)

    # Create 100 scenes of 20 tracks
    np.random.seed(42)
    ndcg3_3c = []; ndcg5_3c = []; r3_crit_3c = []
    ndcg3_4c = []; ndcg5_4c = []; r3_crit_4c = []

    for scene in range(100):
        idx = np.random.choice(len(tracks), 20, replace=True)

        # 3-class ranking
        feats = X_48[idx]
        preds_3 = rf3.predict(feats)
        scores_3 = [SAR3_PRIORITY[SAR3_CLASSES[p]] for p in preds_3]
        # GT priorities
        gt_3 = [SAR3_PRIORITY[tracks[i]["sar3_class"]] for i in idx]
        ranked_idx = np.argsort(-np.array(scores_3))
        ranked_gt = [gt_3[i] for i in ranked_idx]
        ndcg3_3c.append(ndcg(ranked_gt, 3))
        ndcg5_3c.append(ndcg(ranked_gt, 5))
        top3_classes = [tracks[idx[i]]["sar3_class"] for i in ranked_idx[:3]]
        has_crit = any(tracks[i]["sar3_class"]=="critical" for i in idx)
        r3_crit_3c.append(1.0 if "critical" in top3_classes else (0.0 if has_crit else float('nan')))

        # 4-class ranking
        preds_4 = rf4.predict(feats)
        RF4_PRIORITY = {"lying_down":1.0,"stationary":0.45,"walking":0.1,"running":0.2}
        scores_4 = [RF4_PRIORITY[SAR4_CLASSES[p]] for p in preds_4]
        gt_4_prio = {"lying_down":1.0,"stationary":0.45,"walking":0.1,"running":0.2}
        gt_4 = [gt_4_prio[tracks[i]["sar4_class"]] for i in idx]
        ranked_idx4 = np.argsort(-np.array(scores_4))
        ranked_gt4 = [gt_4[i] for i in ranked_idx4]
        ndcg3_4c.append(ndcg(ranked_gt4, 3))
        ndcg5_4c.append(ndcg(ranked_gt4, 5))
        top3_4 = [tracks[idx[i]]["sar4_class"] for i in ranked_idx4[:3]]
        has_lying = any(tracks[i]["sar4_class"]=="lying_down" for i in idx)
        r3_crit_4c.append(1.0 if "lying_down" in top3_4 else (0.0 if has_lying else float('nan')))

    valid_3 = [v for v in r3_crit_3c if not math.isnan(v)]
    valid_4 = [v for v in r3_crit_4c if not math.isnan(v)]

    ranking = {
        "3_class": {
            "ndcg3": float(np.mean(ndcg3_3c)), "ndcg5": float(np.mean(ndcg5_3c)),
            "critical_recall3": float(np.mean(valid_3)) if valid_3 else None,
            "n_scenes_with_critical": len(valid_3),
        },
        "4_class": {
            "ndcg3": float(np.mean(ndcg3_4c)), "ndcg5": float(np.mean(ndcg5_4c)),
            "lying_recall3": float(np.mean(valid_4)) if valid_4 else None,
            "n_scenes_with_lying": len(valid_4),
        },
    }

    log(f'\n  {"Metric":<25} {"4-class":>10} {"3-class":>10} {"Δ":>8}')
    log(f'  {"-"*55}')
    log(f'  {"NDCG@3":<25} {ranking["4_class"]["ndcg3"]:>10.4f} {ranking["3_class"]["ndcg3"]:>10.4f} {ranking["3_class"]["ndcg3"]-ranking["4_class"]["ndcg3"]:>+8.4f}')
    log(f'  {"NDCG@5":<25} {ranking["4_class"]["ndcg5"]:>10.4f} {ranking["3_class"]["ndcg5"]:>10.4f} {ranking["3_class"]["ndcg5"]-ranking["4_class"]["ndcg5"]:>+8.4f}')
    r3c = ranking["3_class"]["critical_recall3"] or 0
    r4c = ranking["4_class"]["lying_recall3"] or 0
    log(f'  {"Critical/Lying R@3":<25} {r4c:>10.1%} {r3c:>10.1%} {(r3c-r4c)*100:>+8.1f}pp')

    # ── Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 3-class confusion matrix
    ax = axes[0,0]
    cm3 = np.array(results["TMS-16+SCTE, 3-class"]["confusion"])
    im = ax.imshow(cm3, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm3[i,j]),ha='center',va='center',fontsize=14,
                    color='white' if cm3[i,j]>cm3.max()/2 else 'black')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(SAR3_CLASSES,rotation=45,ha='right'); ax.set_yticklabels(SAR3_CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'3-Class (TMS-16+SCTE)\nAcc: {results["TMS-16+SCTE, 3-class"]["accuracy"]:.1%}',fontsize=11,fontweight='bold')
    plt.colorbar(im,ax=ax)

    # 4-class confusion matrix
    ax = axes[0,1]
    cm4 = np.array(results["TMS-16+SCTE, 4-class"]["confusion"])
    im = ax.imshow(cm4, cmap='Oranges')
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(cm4[i,j]),ha='center',va='center',fontsize=13,
                    color='white' if cm4[i,j]>cm4.max()/2 else 'black')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(SAR4_CLASSES,rotation=45,ha='right'); ax.set_yticklabels(SAR4_CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'4-Class (TMS-16+SCTE)\nAcc: {results["TMS-16+SCTE, 4-class"]["accuracy"]:.1%}',fontsize=11,fontweight='bold')
    plt.colorbar(im,ax=ax)

    # Accuracy comparison bar chart
    ax = axes[1,0]
    names = list(results.keys()); accs = [results[n]["accuracy"] for n in names]
    colors = ['#e74c3c' if '3-class' in n else '#3498db' for n in names]
    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=9)
    for b,v in zip(bars,accs): ax.text(v+0.005,b.get_y()+b.get_height()/2,f'{v:.1%}',va='center',fontsize=9)
    ax.set_xlabel('Accuracy'); ax.set_title('3-Class vs 4-Class Accuracy',fontsize=11,fontweight='bold')
    ax.invert_yaxis(); ax.set_xlim(0,max(accs)*1.12)

    # Ranking comparison
    ax = axes[1,1]
    metrics = ['NDCG@3','NDCG@5','Critical R@3']
    vals_4 = [ranking["4_class"]["ndcg3"], ranking["4_class"]["ndcg5"], r4c]
    vals_3 = [ranking["3_class"]["ndcg3"], ranking["3_class"]["ndcg5"], r3c]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x-w/2, vals_4, w, label='4-class', color='#3498db', edgecolor='black')
    ax.bar(x+w/2, vals_3, w, label='3-class', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    for i in range(len(metrics)):
        ax.text(i-w/2, vals_4[i]+0.01, f'{vals_4[i]:.3f}', ha='center', fontsize=9)
        ax.text(i+w/2, vals_3[i]+0.01, f'{vals_3[i]:.3f}', ha='center', fontsize=9)
    ax.set_ylabel('Score'); ax.set_title('Ranking Comparison (100 scenes)',fontsize=11,fontweight='bold')
    ax.legend(); ax.set_ylim(0,1)

    plt.suptitle('3-Class SAR-Aligned vs 4-Class Classification', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"three_class_confusion.png"),dpi=200); plt.close()

    # Save
    all_results = {"classification": results, "ranking": ranking}
    with open(os.path.join(OUT_DIR,"three_class_results.json"),"w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(OUT_DIR,"three_class_ranking.json"),"w") as f:
        json.dump(ranking, f, indent=2)

    # Summary
    log('\n'+'='*70)
    log('SUMMARY')
    log('='*70)
    log(f'\n  {"Config":<28} {"Acc":>7} {"CritR":>7} {"κ":>7} {"F1":>7}')
    log(f'  {"-"*55}')
    for n in names:
        r = results[n]
        log(f'  {n:<28} {r["accuracy"]:>7.1%} {r["critical_recall"]:>7.1%} {r["kappa"]:>7.3f} {r["macro_f1"]:>7.3f}')
    log('\nDone.')

if __name__ == "__main__":
    main()
