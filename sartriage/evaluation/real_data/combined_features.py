"""
Multi-Stream Feature Fusion: TMS-16 + SCTE + TrajMAE embeddings → RF
"""
import os, sys, json, math, time
import numpy as np
import torch
from collections import Counter
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
from evaluation.traj_mae import TrajMAE

OUT_DIR = "evaluation/real_data/full"
SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
RF_CLASSES = ["lying_down","stationary","walking","running"]

def log(msg):
    with open('/tmp/combined_progress.txt','a') as f: f.write(msg+'\n')

def main():
    with open('/tmp/combined_progress.txt','w') as f: f.write('Multi-Stream Feature Fusion\n')

    # ── Load data ──
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)

    log('Loading models...')
    scte = SCTEModel(input_dim=4, d_model=32, proj_dim=16, n_heads=2, n_layers=2, dropout=0.1, max_len=50)
    scte.load_state_dict(torch.load("evaluation/results/scte_encoder_trained.pt",
                                     map_location="cpu", weights_only=True))
    scte.eval()
    log('  SCTE ✓')

    trajmae = TrajMAE(num_classes=4, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=50)
    trajmae.encoder.load_state_dict(torch.load("evaluation/results/trajmae_encoder_pretrained.pt",
                                                map_location="cpu", weights_only=True))
    trajmae.eval()
    log('  TrajMAE ✓')

    # ── Extract all features ──
    log('\nExtracting features for all tracks...')
    tms16_all = []; scte_tokens_all = []; delta_tokens_all = []; labels = []
    
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR_MAP: continue

        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        f16 = extract_tms16(traj)
        f16 = [0 if (math.isnan(f) or math.isinf(f)) else f for f in f16]
        tms16_all.append(f16)

        centroids = t["centroids"]; bboxes = t["bboxes"]
        # SCTE tokens (normalised)
        st = []
        for i in range(1, len(centroids)):
            dx = (centroids[i][0]-centroids[i-1][0])/1280.0
            dy = (centroids[i][1]-centroids[i-1][1])/720.0
            aspect = bboxes[i][3]/max(bboxes[i][2],1)
            sn = math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            st.append([dx,dy,aspect,sn])
        arr_s = np.zeros((50,4), dtype=np.float32)
        for j in range(min(len(st),50)): arr_s[j] = st[j]
        scte_tokens_all.append(arr_s)

        # TrajMAE tokens (raw deltas)
        dt = []
        for i in range(1, len(centroids)):
            dt.append([centroids[i][0]-centroids[i-1][0], centroids[i][1]-centroids[i-1][1],
                       bboxes[i][2]-bboxes[i-1][2], bboxes[i][3]-bboxes[i-1][3]])
        arr_d = np.zeros((50,4), dtype=np.float32)
        for j in range(min(len(dt),50)): arr_d[j] = dt[j]
        delta_tokens_all.append(arr_d)

        labels.append(RF_CLASSES.index(SAR_MAP[act]))

    X_tms = np.array(tms16_all, dtype=np.float32)
    y = np.array(labels)
    log(f'Tracks: {len(y)}, classes: {Counter(y.tolist())}')

    # Extract SCTE embeddings (batch)
    log('Extracting SCTE embeddings...')
    scte_arr = np.stack(scte_tokens_all)
    with torch.no_grad():
        scte_embs = []
        for s in range(0, len(scte_arr), 64):
            scte_embs.append(scte.get_embedding(torch.FloatTensor(scte_arr[s:s+64])).cpu().numpy())
    X_scte = np.concatenate(scte_embs)  # (N, 32)
    log(f'  SCTE: {X_scte.shape}')

    # Extract TrajMAE CLS embeddings (batch)
    log('Extracting TrajMAE embeddings...')
    delta_arr = np.stack(delta_tokens_all)
    with torch.no_grad():
        mae_embs = []
        for s in range(0, len(delta_arr), 64):
            batch = torch.FloatTensor(delta_arr[s:s+64])
            tokens, pad_mask = trajmae.encoder.embed_tokens(batch)
            encoded = trajmae.encoder(tokens, padding_mask=pad_mask)
            cls_emb = trajmae.encoder.cls_embedding(encoded)
            mae_embs.append(cls_emb.cpu().numpy())
    X_mae = np.concatenate(mae_embs)  # (N, 64)
    log(f'  TrajMAE: {X_mae.shape}')

    # ── Build feature combinations ──
    configs = {
        "TMS-16 only":          X_tms,                                    # 16
        "SCTE only":            X_scte,                                   # 32
        "TrajMAE-CLS only":     X_mae,                                    # 64
        "TMS-16 + SCTE":        np.hstack([X_tms, X_scte]),              # 48
        "TMS-16 + TrajMAE":     np.hstack([X_tms, X_mae]),               # 80
        "SCTE + TrajMAE":       np.hstack([X_scte, X_mae]),              # 96
        "TMS-16+SCTE+TrajMAE":  np.hstack([X_tms, X_scte, X_mae]),      # 112
    }

    # ── 5-fold CV ──
    log('\n' + '='*70)
    log('5-FOLD STRATIFIED CV')
    log('='*70)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, X in configs.items():
        fold_accs = []; all_pred = []; all_true = []
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            try:
                sm = SMOTE(random_state=42)
                X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
            except:
                X_sm, y_sm = X_tr, y_tr
            rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            rf.fit(X_sm, y_sm)
            preds = rf.predict(X_te)
            fold_accs.append(accuracy_score(y_te, preds))
            all_pred.extend(preds); all_true.extend(y_te)

        all_pred = np.array(all_pred); all_true = np.array(all_true)
        acc = np.mean(fold_accs); ci = 1.96*np.std(fold_accs)
        lying_r = recall_score(all_true==0, all_pred==0, zero_division=0)
        kappa = cohen_kappa_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, target_names=RF_CLASSES, output_dict=True)

        results[name] = {
            "n_features": int(X.shape[1]),
            "accuracy": float(acc), "accuracy_ci": float(ci),
            "lying_recall": float(lying_r),
            "kappa": float(kappa),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {c: {"p":round(report[c]["precision"],3),
                              "r":round(report[c]["recall"],3),
                              "f1":round(report[c]["f1-score"],3)} for c in RF_CLASSES},
            "confusion": confusion_matrix(all_true, all_pred).tolist(),
        }
        log(f'  {name:<25} ({X.shape[1]:>3}d)  Acc={acc:.1%}±{ci:.1%}  lying_R={lying_r:.1%}  κ={kappa:.3f}  F1={report["macro avg"]["f1-score"]:.3f}')

    # ── Best config confusion matrix ──
    best = max(results, key=lambda k: results[k]["accuracy"])
    log(f'\n  BEST: {best} ({results[best]["accuracy"]:.1%})')

    X_best = configs[best]
    # Full train/test for confusion matrix figure
    np.random.seed(42); idx = np.random.permutation(len(y))
    split = int(0.7*len(idx))
    X_tr, X_te = X_best[idx[:split]], X_best[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]
    sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_sm, y_sm); preds = rf.predict(X_te)
    cm = confusion_matrix(y_te, preds)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(RF_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(RF_CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{best}\n(Acc: {accuracy_score(y_te,preds):.1%})', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Bar chart comparison
    ax = axes[1]
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    dims = [results[n]["n_features"] for n in names]
    colors = ['#95a5a6' if results[n]["n_features"]<=32 else '#3498db' if results[n]["n_features"]<=48
              else '#2ecc71' if results[n]["n_features"]<=80 else '#e74c3c' for n in names]
    bars = ax.barh(range(len(names)), accs, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f'{n} ({dims[i]}d)' for i,n in enumerate(names)], fontsize=9)
    for b, v in zip(bars, accs):
        ax.text(v+0.005, b.get_y()+b.get_height()/2, f'{v:.1%}', va='center', fontsize=9)
    ax.set_xlabel('Accuracy (5-fold CV)'); ax.set_title('Feature Fusion Comparison', fontsize=12, fontweight='bold')
    ax.axvline(results["TMS-16 only"]["accuracy"], color='red', linestyle='--', alpha=0.5,
               label=f'TMS-16 baseline ({results["TMS-16 only"]["accuracy"]:.1%})')
    ax.legend(fontsize=9); ax.set_xlim(0, max(accs)*1.12); ax.invert_yaxis()

    plt.suptitle('Multi-Stream Trajectory Feature Fusion', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "combined_features_confusion.png"), dpi=200); plt.close()

    # Save
    with open(os.path.join(OUT_DIR, "combined_features_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log('\n' + '='*70)
    log('SUMMARY')
    log('='*70)
    log(f'\n  {"Config":<25} {"Dims":>5} {"Acc":>7} {"Δ vs TMS16":>11} {"lying_R":>8} {"κ":>6}')
    log(f'  {"-"*65}')
    base = results["TMS-16 only"]["accuracy"]
    for name in names:
        r = results[name]
        delta = (r["accuracy"]-base)*100
        log(f'  {name:<25} {r["n_features"]:>5} {r["accuracy"]:>7.1%} {delta:>+10.1f}pp {r["lying_recall"]:>8.1%} {r["kappa"]:>6.3f}')
    log('\nDone.')

if __name__ == "__main__":
    main()
