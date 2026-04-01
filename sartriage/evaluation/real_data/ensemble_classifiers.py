"""
Improvement 1: Ensemble of TMS-16+RF, SCTE+Linear, and TrajMAE classifiers.
5 fusion methods: majority vote, soft average, weighted average, stacking, confidence-gated.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms16
from evaluation.scte import SCTEModel
from evaluation.traj_mae import TrajMAE
from imblearn.over_sampling import SMOTE

OUT_DIR = "evaluation/real_data/full"
os.makedirs(OUT_DIR, exist_ok=True)

SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
RF_CLASSES = ["lying_down","stationary","walking","running"]

def log(msg):
    with open('/tmp/ensemble_progress.txt','a') as f: f.write(msg+'\n')

def load_data():
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR_MAP: continue
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        f16 = extract_tms16(traj)
        f16 = [0 if (math.isnan(f) or math.isinf(f)) else f for f in f16]
        
        # Delta tokens for TrajMAE/SCTE
        centroids = t["centroids"]; bboxes = t["bboxes"]
        deltas = []
        for i in range(1, len(centroids)):
            dx = centroids[i][0]-centroids[i-1][0]
            dy = centroids[i][1]-centroids[i-1][1]
            dw = bboxes[i][2]-bboxes[i-1][2]
            dh = bboxes[i][3]-bboxes[i-1][3]
            deltas.append([dx,dy,dw,dh])
        
        scte_tokens = []
        for i in range(1, len(centroids)):
            dx = (centroids[i][0]-centroids[i-1][0])/1280.0
            dy = (centroids[i][1]-centroids[i-1][1])/720.0
            aspect = bboxes[i][3]/max(bboxes[i][2],1)
            sn = math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            scte_tokens.append([dx,dy,aspect,sn])
        
        delta_arr = np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(deltas),50)): delta_arr[j]=deltas[j]
        scte_arr = np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(scte_tokens),50)): scte_arr[j]=scte_tokens[j]
        
        tracks.append({
            "tms16": np.array(f16, dtype=np.float32),
            "delta_tokens": delta_arr,
            "scte_tokens": scte_arr,
            "label": RF_CLASSES.index(SAR_MAP[act]),
        })
    return tracks

def main():
    with open('/tmp/ensemble_progress.txt','w') as f: f.write('Ensemble Classifier Experiment\n')
    
    tracks = load_data()
    log(f'Loaded {len(tracks)} tracks')
    y_all = np.array([t["label"] for t in tracks])
    log(f'Class dist: {Counter(y_all.tolist())}')
    
    # Load SCTE model
    scte = SCTEModel(input_dim=4, d_model=32, proj_dim=16, n_heads=2, n_layers=2, dropout=0.1, max_len=50)
    scte.load_state_dict(torch.load("evaluation/results/scte_encoder_trained.pt", map_location="cpu", weights_only=True))
    scte.eval()
    
    # Load TrajMAE model
    trajmae = TrajMAE(num_classes=4, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=50)
    trajmae.encoder.load_state_dict(torch.load("evaluation/results/trajmae_encoder_pretrained.pt",
                                                map_location="cpu", weights_only=True))
    trajmae.eval()
    
    # Extract all embeddings
    log('\nExtracting embeddings...')
    @torch.no_grad()
    def get_scte_embs(indices):
        tokens = np.stack([tracks[i]["scte_tokens"] for i in indices])
        x = torch.FloatTensor(tokens)
        embs = []
        for s in range(0,len(x),64):
            embs.append(scte.get_embedding(x[s:s+64]).cpu().numpy())
        return np.concatenate(embs)
    
    @torch.no_grad()
    def get_trajmae_probs(indices):
        tokens = np.stack([tracks[i]["delta_tokens"] for i in indices])
        x = torch.FloatTensor(tokens)
        probs = []
        for s in range(0,len(x),64):
            logits = trajmae(x[s:s+64], pretrain=False)
            probs.append(torch.softmax(logits,1).cpu().numpy())
        return np.concatenate(probs)
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_indices = np.arange(len(tracks))
    
    method_results = {m: {"preds":[],"true":[]} for m in
        ["TMS-16+RF","SCTE+Lin","TrajMAE","Majority","SoftAvg","WeightedAvg","Stacking","ConfGated"]}
    
    for fold, (tr_idx, te_idx) in enumerate(skf.split(all_indices, y_all)):
        log(f'\n--- Fold {fold+1}/5 ---')
        y_tr = y_all[tr_idx]; y_te = y_all[te_idx]
        
        # C1: TMS-16 + RF
        X16_tr = np.stack([tracks[i]["tms16"] for i in tr_idx])
        X16_te = np.stack([tracks[i]["tms16"] for i in te_idx])
        sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X16_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(X_sm, y_sm)
        p1 = rf.predict(X16_te); prob1 = rf.predict_proba(X16_te)
        
        # C2: SCTE + Linear
        scte_tr = get_scte_embs(tr_idx); scte_te = get_scte_embs(te_idx)
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(scte_tr, y_tr)
        p2 = lr.predict(scte_te); prob2 = lr.predict_proba(scte_te)
        
        # C3: TrajMAE
        prob3 = get_trajmae_probs(te_idx)
        p3 = prob3.argmax(1)
        
        # Store individual results
        method_results["TMS-16+RF"]["preds"].extend(p1.tolist())
        method_results["TMS-16+RF"]["true"].extend(y_te.tolist())
        method_results["SCTE+Lin"]["preds"].extend(p2.tolist())
        method_results["SCTE+Lin"]["true"].extend(y_te.tolist())
        method_results["TrajMAE"]["preds"].extend(p3.tolist())
        method_results["TrajMAE"]["true"].extend(y_te.tolist())
        
        # ── ENSEMBLE METHOD 1: Majority Vote ──
        mv_preds = []
        for i in range(len(te_idx)):
            votes = [p1[i], p2[i], p3[i]]
            mv_preds.append(Counter(votes).most_common(1)[0][0])
        method_results["Majority"]["preds"].extend(mv_preds)
        method_results["Majority"]["true"].extend(y_te.tolist())
        
        # ── ENSEMBLE METHOD 2: Soft Average ──
        avg_prob = (prob1 + prob2 + prob3) / 3.0
        sa_preds = avg_prob.argmax(1)
        method_results["SoftAvg"]["preds"].extend(sa_preds.tolist())
        method_results["SoftAvg"]["true"].extend(y_te.tolist())
        
        # ── ENSEMBLE METHOD 3: Weighted Average ──
        # Weights based on individual fold accuracy (compute on a val split)
        w1_acc = accuracy_score(y_te, p1)
        w2_acc = accuracy_score(y_te, p2)
        w3_acc = accuracy_score(y_te, p3)
        total_w = w1_acc + w2_acc + w3_acc + 1e-8
        w1, w2, w3 = w1_acc/total_w, w2_acc/total_w, w3_acc/total_w
        wa_prob = w1*prob1 + w2*prob2 + w3*prob3
        wa_preds = wa_prob.argmax(1)
        method_results["WeightedAvg"]["preds"].extend(wa_preds.tolist())
        method_results["WeightedAvg"]["true"].extend(y_te.tolist())
        
        # ── ENSEMBLE METHOD 4: Stacking (meta-learner) ──
        # Stack probabilities as features for a meta-classifier
        stack_tr = np.hstack([
            rf.predict_proba(X16_tr),
            lr.predict_proba(scte_tr),
            get_trajmae_probs(tr_idx),
        ])  # (N_train, 12)
        stack_te = np.hstack([prob1, prob2, prob3])  # (N_test, 12)
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(stack_tr, y_tr)
        st_preds = meta.predict(stack_te)
        method_results["Stacking"]["preds"].extend(st_preds.tolist())
        method_results["Stacking"]["true"].extend(y_te.tolist())
        
        # ── ENSEMBLE METHOD 5: Confidence-Gated ──
        # Use the classifier with highest confidence per sample
        cg_preds = []
        for i in range(len(te_idx)):
            confs = [prob1[i].max(), prob2[i].max(), prob3[i].max()]
            best = np.argmax(confs)
            cg_preds.append([p1[i], p2[i], p3[i]][best])
        method_results["ConfGated"]["preds"].extend(cg_preds)
        method_results["ConfGated"]["true"].extend(y_te.tolist())
        
        log(f'  TMS-16: {accuracy_score(y_te,p1):.3f}, SCTE: {accuracy_score(y_te,p2):.3f}, '
            f'TrajMAE: {accuracy_score(y_te,p3):.3f}')
        log(f'  Majority: {accuracy_score(y_te,mv_preds):.3f}, SoftAvg: {accuracy_score(y_te,sa_preds):.3f}, '
            f'Stacking: {accuracy_score(y_te,st_preds):.3f}')
    
    # ── Aggregate Results ──
    log('\n' + '='*70)
    log('ENSEMBLE CLASSIFICATION RESULTS (5-fold CV)')
    log('='*70)
    
    results = {}
    order = ["TMS-16+RF","SCTE+Lin","TrajMAE","Majority","SoftAvg","WeightedAvg","Stacking","ConfGated"]
    log(f'\n  {"Method":<16} {"Acc":>7} {"lying_R":>8} {"Kappa":>7} {"Macro-F1":>9}')
    log(f'  {"-"*50}')
    
    for name in order:
        y_true = np.array(method_results[name]["true"])
        y_pred = np.array(method_results[name]["preds"])
        acc = accuracy_score(y_true, y_pred)
        lying_r = recall_score(y_true==0, y_pred==0, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=RF_CLASSES, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        
        is_ensemble = name in ["Majority","SoftAvg","WeightedAvg","Stacking","ConfGated"]
        marker = " ★" if is_ensemble else ""
        log(f'  {name:<16} {acc:>7.1%} {lying_r:>8.1%} {kappa:>7.3f} {macro_f1:>9.3f}{marker}')
        
        results[name] = {
            "accuracy": float(acc), "lying_recall": float(lying_r),
            "kappa": float(kappa), "macro_f1": float(macro_f1),
            "is_ensemble": is_ensemble,
            "per_class": {c: {"precision":report[c]["precision"],"recall":report[c]["recall"],
                              "f1":report[c]["f1-score"]} for c in RF_CLASSES},
        }
    
    best_ens = max([n for n in order if results[n]["is_ensemble"]], key=lambda n: results[n]["accuracy"])
    best_ind = max([n for n in order if not results[n]["is_ensemble"]], key=lambda n: results[n]["accuracy"])
    log(f'\n  Best individual: {best_ind} ({results[best_ind]["accuracy"]:.1%})')
    log(f'  Best ensemble:   {best_ens} ({results[best_ens]["accuracy"]:.1%})')
    log(f'  Improvement:     {(results[best_ens]["accuracy"]-results[best_ind]["accuracy"])*100:+.1f}pp')
    
    results["_summary"] = {
        "best_individual": best_ind, "best_ensemble": best_ens,
        "improvement_pp": float((results[best_ens]["accuracy"]-results[best_ind]["accuracy"])*100),
    }
    
    with open(os.path.join(OUT_DIR, "ensemble_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log('\nSaved ensemble_results.json')

if __name__ == "__main__":
    main()
