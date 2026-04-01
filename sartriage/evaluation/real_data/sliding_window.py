"""
Sliding Window TMS Features: temporal consistency for classification.
"""
import os, sys, json, math, time
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms16

OUT_DIR = "evaluation/real_data/full"
SAR3_MAP = {"Standing":"upright_stationary","Sitting":"upright_stationary",
            "Walking":"mobile","Running":"mobile","Lying":"critical"}
SAR3_CLASSES = ["critical","upright_stationary","mobile"]
WINDOW = 30; STRIDE = 15

def log(msg):
    with open('/tmp/sliding_progress.txt','a') as f: f.write(msg+'\n')

def main():
    with open('/tmp/sliding_progress.txt','w') as f: f.write('Sliding Window Features\n')

    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)

    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR3_MAP: continue
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        label = SAR3_CLASSES.index(SAR3_MAP[act])

        # Whole-track features
        f16 = extract_tms16(traj)
        f16 = [0 if(math.isnan(f)or math.isinf(f))else f for f in f16]

        # Sliding window features
        windows = []
        if len(traj) >= WINDOW:
            for start in range(0, len(traj)-WINDOW+1, STRIDE):
                w = traj[start:start+WINDOW]
                wf = extract_tms16(w)
                wf = [0 if(math.isnan(f)or math.isinf(f))else f for f in wf]
                windows.append(wf)
        else:
            windows.append(f16)  # Track too short for windowing

        tracks.append({
            "tms16": np.array(f16, dtype=np.float32),
            "windows": [np.array(w, dtype=np.float32) for w in windows],
            "n_windows": len(windows),
            "label": label, "traj_len": len(traj),
        })

    y = np.array([t["label"] for t in tracks])
    log(f'Tracks: {len(tracks)}, classes: {dict(zip(SAR3_CLASSES,[sum(y==i) for i in range(3)]))}')
    log(f'Windows per track: min={min(t["n_windows"] for t in tracks)} max={max(t["n_windows"] for t in tracks)} '
        f'mean={np.mean([t["n_windows"] for t in tracks]):.1f}')

    # ═══════════════════════════════════════════════════════════════
    # 5-fold CV
    # ═══════════════════════════════════════════════════════════════
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_whole = np.stack([t["tms16"] for t in tracks])

    methods = {}

    # ── M1: Whole-track TMS-16 (baseline) ──
    all_pred=[]; all_true=[]
    for tr_i, te_i in skf.split(X_whole, y):
        sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_whole[tr_i], y[tr_i])
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(X_sm, y_sm)
        all_pred.extend(rf.predict(X_whole[te_i])); all_true.extend(y[te_i])
    all_pred=np.array(all_pred); all_true=np.array(all_true)
    methods["Whole-track TMS-16"] = {
        "accuracy": float(accuracy_score(all_true,all_pred)),
        "kappa": float(cohen_kappa_score(all_true,all_pred)),
        "critical_recall": float(classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)["critical"]["recall"]),
        "macro_f1": float(classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)["macro avg"]["f1-score"]),
    }

    # ── M2: Window-level classifier → majority vote ──
    all_pred=[]; all_true=[]
    for tr_i, te_i in skf.split(X_whole, y):
        # Build window-level training data
        X_w_tr=[]; y_w_tr=[]
        for i in tr_i:
            for w in tracks[i]["windows"]:
                X_w_tr.append(w); y_w_tr.append(y[i])
        X_w_tr=np.stack(X_w_tr); y_w_tr=np.array(y_w_tr)
        sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_w_tr, y_w_tr)
        rf_w = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_w.fit(X_sm, y_sm)
        
        for i in te_i:
            w_preds = rf_w.predict(np.stack(tracks[i]["windows"]))
            vote = Counter(w_preds.tolist()).most_common(1)[0][0]
            all_pred.append(vote); all_true.append(y[i])
    all_pred=np.array(all_pred); all_true=np.array(all_true)
    methods["Window majority vote"] = {
        "accuracy": float(accuracy_score(all_true,all_pred)),
        "kappa": float(cohen_kappa_score(all_true,all_pred)),
        "critical_recall": float(classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)["critical"]["recall"]),
        "macro_f1": float(classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)["macro avg"]["f1-score"]),
    }

    # ── M3: Window-level soft average (probabilities) ──
    all_pred=[]; all_true=[]
    for tr_i, te_i in skf.split(X_whole, y):
        X_w_tr=[]; y_w_tr=[]
        for i in tr_i:
            for w in tracks[i]["windows"]:
                X_w_tr.append(w); y_w_tr.append(y[i])
        X_w_tr=np.stack(X_w_tr); y_w_tr=np.array(y_w_tr)
        sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_w_tr, y_w_tr)
        rf_w = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_w.fit(X_sm, y_sm)
        
        for i in te_i:
            w_probs = rf_w.predict_proba(np.stack(tracks[i]["windows"]))
            avg_prob = w_probs.mean(axis=0)
            all_pred.append(avg_prob.argmax()); all_true.append(y[i])
    all_pred=np.array(all_pred); all_true=np.array(all_true)
    methods["Window soft average"] = {
        "accuracy": float(accuracy_score(all_true,all_pred)),
        "kappa": float(cohen_kappa_score(all_true,all_pred)),
        "critical_recall": float(classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)["critical"]["recall"]),
        "macro_f1": float(classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)["macro avg"]["f1-score"]),
    }

    # ── M4: Window statistics as features ──
    # For each track: compute mean, std, min, max of each of the 16 window features
    # Plus: prediction_consistency, n_windows, temporal_trend
    all_pred=[]; all_true=[]
    for tr_i, te_i in skf.split(X_whole, y):
        # First train window-level RF
        X_w_tr=[]; y_w_tr=[]
        for i in tr_i:
            for w in tracks[i]["windows"]:
                X_w_tr.append(w); y_w_tr.append(y[i])
        X_w_tr=np.stack(X_w_tr); y_w_tr=np.array(y_w_tr)
        sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_w_tr, y_w_tr)
        rf_w = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_w.fit(X_sm, y_sm)
        
        # Build meta-features
        def build_meta(indices):
            metas = []
            for i in indices:
                ws = np.stack(tracks[i]["windows"])  # (n_win, 16)
                # Stats per feature
                feat_mean = ws.mean(0)  # 16
                feat_std = ws.std(0)    # 16
                feat_min = ws.min(0)    # 16
                feat_max = ws.max(0)    # 16
                # Window predictions
                w_preds = rf_w.predict(ws)
                w_probs = rf_w.predict_proba(ws)
                # Prediction consistency
                most_common = Counter(w_preds.tolist()).most_common(1)[0]
                pred_consistency = most_common[1] / len(w_preds)
                # Mean confidence
                mean_conf = w_probs.max(1).mean()
                # Class probability trends (first half vs second half)
                if len(w_probs) >= 2:
                    half = len(w_probs)//2
                    prob_trend = w_probs[half:].mean(0) - w_probs[:half].mean(0)  # 3
                else:
                    prob_trend = np.zeros(3)
                # Number of class switches
                switches = sum(1 for j in range(1,len(w_preds)) if w_preds[j]!=w_preds[j-1])
                switch_rate = switches / max(len(w_preds)-1, 1)
                
                meta = np.concatenate([
                    feat_mean, feat_std, feat_min, feat_max,  # 64
                    [pred_consistency, mean_conf, switch_rate, len(ws)],  # 4
                    prob_trend,  # 3
                    tracks[i]["tms16"],  # 16 (whole-track features too)
                ])
                metas.append(meta)
            return np.stack(metas)
        
        X_meta_tr = build_meta(tr_i); X_meta_te = build_meta(te_i)
        y_meta_tr = y[tr_i]
        sm2 = SMOTE(random_state=42); X_sm2, y_sm2 = sm2.fit_resample(X_meta_tr, y_meta_tr)
        rf_meta = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_meta.fit(X_sm2, y_sm2)
        p = rf_meta.predict(X_meta_te)
        all_pred.extend(p); all_true.extend(y[te_i])
    all_pred=np.array(all_pred); all_true=np.array(all_true)
    rpt = classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)
    methods["Window meta-features"] = {
        "accuracy": float(accuracy_score(all_true,all_pred)),
        "kappa": float(cohen_kappa_score(all_true,all_pred)),
        "critical_recall": float(rpt["critical"]["recall"]),
        "macro_f1": float(rpt["macro avg"]["f1-score"]),
        "n_features": 87,
    }

    # ── M5: Whole-track + consistency only ──
    # Add just pred_consistency and switch_rate to TMS-16
    all_pred=[]; all_true=[]
    for tr_i, te_i in skf.split(X_whole, y):
        X_w_tr=[]; y_w_tr=[]
        for i in tr_i:
            for w in tracks[i]["windows"]:
                X_w_tr.append(w); y_w_tr.append(y[i])
        X_w_tr=np.stack(X_w_tr); y_w_tr=np.array(y_w_tr)
        sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_w_tr, y_w_tr)
        rf_w = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_w.fit(X_sm, y_sm)
        
        def add_consistency(indices):
            feats = []
            for i in indices:
                ws = np.stack(tracks[i]["windows"])
                w_preds = rf_w.predict(ws)
                w_probs = rf_w.predict_proba(ws)
                most_common = Counter(w_preds.tolist()).most_common(1)[0]
                pred_cons = most_common[1] / len(w_preds)
                mean_conf = w_probs.max(1).mean()
                switches = sum(1 for j in range(1,len(w_preds)) if w_preds[j]!=w_preds[j-1])
                switch_rate = switches / max(len(w_preds)-1, 1)
                feats.append(np.concatenate([tracks[i]["tms16"], [pred_cons, mean_conf, switch_rate]]))
            return np.stack(feats)
        
        X_tr = add_consistency(tr_i); X_te = add_consistency(te_i)
        sm2 = SMOTE(random_state=42); X_sm2, y_sm2 = sm2.fit_resample(X_tr, y[tr_i])
        rf2 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf2.fit(X_sm2, y_sm2)
        all_pred.extend(rf2.predict(X_te)); all_true.extend(y[te_i])
    all_pred=np.array(all_pred); all_true=np.array(all_true)
    rpt = classification_report(all_true,all_pred,target_names=SAR3_CLASSES,output_dict=True)
    methods["TMS-16 + consistency"] = {
        "accuracy": float(accuracy_score(all_true,all_pred)),
        "kappa": float(cohen_kappa_score(all_true,all_pred)),
        "critical_recall": float(rpt["critical"]["recall"]),
        "macro_f1": float(rpt["macro avg"]["f1-score"]),
        "n_features": 19,
    }

    # ── Summary ──
    log('\n' + '='*70)
    log('SLIDING WINDOW RESULTS (3-class, 5-fold CV)')
    log('='*70)
    log(f'\n  {"Method":<28} {"Acc":>7} {"CritR":>7} {"κ":>7} {"F1":>7}')
    log(f'  {"-"*58}')
    base_acc = methods["Whole-track TMS-16"]["accuracy"]
    for name in ["Whole-track TMS-16","Window majority vote","Window soft average",
                  "Window meta-features","TMS-16 + consistency"]:
        r = methods[name]
        delta = (r["accuracy"]-base_acc)*100
        log(f'  {name:<28} {r["accuracy"]:>7.1%} {r["critical_recall"]:>7.1%} {r["kappa"]:>7.3f} {r["macro_f1"]:>7.3f}  ({delta:+.1f}pp)')

    with open(os.path.join(OUT_DIR,"sliding_window_results.json"),"w") as f:
        json.dump(methods, f, indent=2)
    log('\nDone.')

if __name__ == "__main__":
    main()
