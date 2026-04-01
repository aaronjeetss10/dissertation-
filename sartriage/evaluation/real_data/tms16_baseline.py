"""
TMS-16 Evaluation + R3D-18 Cache + Baseline Comparison Table
Final classification results for the dissertation.
"""
import os, sys, json, math, time, glob, warnings
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             cohen_kappa_score, recall_score)
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12, extract_tms16

warnings.filterwarnings("ignore")

OUT_DIR = "evaluation/real_data/full"
os.makedirs(OUT_DIR, exist_ok=True)

SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
RF_CLASSES = ["lying_down","stationary","walking","running"]

def log(msg):
    with open('/tmp/baseline_progress.txt','a') as f: f.write(msg+'\n')

# ═══════════════════════════════════════════════════════════════════════
# Load Okutama data
# ═══════════════════════════════════════════════════════════════════════
def load_data():
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR_MAP: continue
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        f12 = extract_tms12(traj)
        f16 = extract_tms16(traj)
        if any(math.isnan(f) or math.isinf(f) for f in f16):
            f16 = [0 if (math.isnan(f) or math.isinf(f)) else f for f in f16]
        if any(math.isnan(f) or math.isinf(f) for f in f12):
            f12 = [0 if (math.isnan(f) or math.isinf(f)) else f for f in f12]
        tracks.append({
            "tid": tid, "tms12": f12, "tms16": f16,
            "label": RF_CLASSES.index(SAR_MAP[act]),
            "sar_class": SAR_MAP[act], "gt_action": act,
            "centroids": t["centroids"], "bboxes": t["bboxes"],
            "mean_size": float(np.mean([math.sqrt(max(b[2],1)*max(b[3],1)) for b in t["bboxes"]])),
        })
    return tracks

# ═══════════════════════════════════════════════════════════════════════
# TASK 1: TMS-12 vs TMS-16 comparison
# ═══════════════════════════════════════════════════════════════════════
def task1_tms16(tracks):
    log('\n' + '='*60)
    log('TASK 1: TMS-12 vs TMS-16 (5-fold stratified CV)')
    log('='*60)
    
    X12 = np.array([t["tms12"] for t in tracks])
    X16 = np.array([t["tms16"] for t in tracks])
    y = np.array([t["label"] for t in tracks])
    
    from imblearn.over_sampling import SMOTE
    
    configs = [
        ("TMS-12 + RF", X12, "rf"),
        ("TMS-16 + RF", X16, "rf"),
        ("TMS-16 + GBM", X16, "gbm"),
    ]
    
    results = {}
    for name, X, clf_type in configs:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []; fold_lying = []; fold_kappas = []; all_preds = []; all_true = []
        
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            
            try:
                sm = SMOTE(random_state=42)
                X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
            except:
                X_sm, y_sm = X_tr, y_tr
            
            if clf_type == "rf":
                clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            else:
                clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
            clf.fit(X_sm, y_sm)
            preds = clf.predict(X_te)
            
            fold_accs.append(accuracy_score(y_te, preds))
            lying_mask = y_te == 0
            if lying_mask.sum() > 0:
                fold_lying.append(recall_score(y_te[lying_mask]==0, preds[lying_mask]==0, zero_division=0))
            fold_kappas.append(cohen_kappa_score(y_te, preds))
            all_preds.extend(preds); all_true.extend(y_te)
        
        acc = np.mean(fold_accs)
        acc_ci = 1.96 * np.std(fold_accs)
        lying_r = np.mean(fold_lying) if fold_lying else 0
        kappa = np.mean(fold_kappas)
        
        all_preds = np.array(all_preds); all_true = np.array(all_true)
        lying_recall = recall_score(all_true==0, all_preds==0, zero_division=0)
        
        results[name] = {
            "accuracy": float(acc), "accuracy_ci": float(acc_ci),
            "lying_recall": float(lying_recall),
            "kappa": float(kappa),
            "report": classification_report(all_true, all_preds, target_names=RF_CLASSES, output_dict=True),
        }
        log(f'\n  {name}:')
        log(f'    Accuracy: {acc:.4f} ± {acc_ci:.4f}')
        log(f'    lying_down Recall: {lying_recall:.4f}')
        log(f'    Kappa: {kappa:.4f}')
    
    # Best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    log(f'\n  BEST: {best_name} ({results[best_name]["accuracy"]:.4f})')
    
    # Save TMS-16 confusion matrix
    X16_full = np.array([t["tms16"] for t in tracks])
    y_full = np.array([t["label"] for t in tracks])
    np.random.seed(42)
    idx = np.random.permutation(len(y_full))
    split = int(0.7 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X16_full[tr_idx], y_full[tr_idx])
    rf16 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf16.fit(X_sm, y_sm)
    preds16 = rf16.predict(X16_full[te_idx])
    cm = confusion_matrix(y_full[te_idx], preds16)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(len(RF_CLASSES)):
        for j in range(len(RF_CLASSES)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_xticks(range(len(RF_CLASSES))); ax.set_yticks(range(len(RF_CLASSES)))
    ax.set_xticklabels(RF_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(RF_CLASSES)
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'TMS-16 + RF Confusion Matrix (Acc: {accuracy_score(y_full[te_idx], preds16):.1%})', fontsize=13)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tms16_confusion.png"), dpi=200)
    plt.close()
    
    return results, best_name

# ═══════════════════════════════════════════════════════════════════════
# R3D-18 Cache (separate model load)
# ═══════════════════════════════════════════════════════════════════════
def cache_r3d18(tracks):
    log('\n' + '='*60)
    log('R3D-18 Cache Generation')
    log('='*60)
    
    R3D_WEIGHTS = "models/action_r3d18_sar.pt"
    if not os.path.exists(R3D_WEIGHTS):
        log('  R3D-18 weights not found!'); return None
    
    from torchvision.models.video import r3d_18
    ckpt = torch.load(R3D_WEIGHTS, map_location="cpu", weights_only=False)
    model = r3d_18()
    # Adjust head
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    
    # Figure out number of classes
    fc_key = [k for k in sd if "fc" in k and "weight" in k]
    if fc_key:
        n_classes = sd[fc_key[0]].shape[0]
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        n_classes = 9
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    
    try:
        model.load_state_dict(sd, strict=True)
    except:
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            log(f'  R3D-18 load error: {e}'); return None
    
    model.eval()
    
    MVIT_9 = {0:"falling",1:"crawling",2:"lying_down",3:"running",4:"waving_hand",
              5:"climbing",6:"stumbling",7:"pushing",8:"pulling"}
    MVIT_TO_SAR = {"falling":"lying_down","crawling":"lying_down","lying_down":"lying_down",
                   "running":"running","waving_hand":"stationary","climbing":"walking",
                   "stumbling":"walking","pushing":"stationary","pulling":"stationary"}
    
    TEST_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
    LABELS_DIR = os.path.join(TEST_BASE, "Labels", "MultiActionLabels", "3840x2160")
    SCALE_X = 1280.0/3840.0; SCALE_Y = 720.0/2160.0
    
    r3d_cache = {}
    for seq in ["1.1.8","1.2.3","2.2.3"]:
        frame_dir = None
        for m in glob.glob(os.path.join(TEST_BASE,"**",seq), recursive=True):
            if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
        if not frame_dir: continue
        
        gt_tracks = defaultdict(lambda: {"frames":[],"bboxes":[],"action":None})
        with open(os.path.join(LABELS_DIR, f"{seq}.txt")) as f:
            for line in f:
                p = line.strip().split()
                if len(p)<11: continue
                tid=int(p[0]); xmin=float(p[1])*SCALE_X; ymin=float(p[2])*SCALE_Y
                xmax=float(p[3])*SCALE_X; ymax=float(p[4])*SCALE_Y
                fi=int(p[5]); act=p[10].strip('"')
                gt_tracks[tid]["frames"].append(fi)
                gt_tracks[tid]["bboxes"].append([xmin,ymin,xmax-xmin,ymax-ymin])
                gt_tracks[tid]["action"]=act
        
        seq_cache = {}
        for tid, tv in gt_tracks.items():
            avail = list(zip(tv["frames"], tv["bboxes"]))
            if len(avail) > 16:
                step = len(avail)//16; avail = avail[::step][:16]
            
            crops = []
            for fi, bb in avail:
                img_path = os.path.join(frame_dir, f"{fi}.jpg")
                if not os.path.exists(img_path):
                    crops.append(np.zeros((112,112,3),dtype=np.float32)); continue
                img = cv2.imread(img_path)
                if img is None:
                    crops.append(np.zeros((112,112,3),dtype=np.float32)); continue
                x,y,w,h = int(bb[0]),int(bb[1]),int(max(bb[2],1)),int(max(bb[3],1))
                crop = img[max(0,y):max(0,y+h), max(0,x):max(0,x+w)]
                if crop.size == 0:
                    crops.append(np.zeros((112,112,3),dtype=np.float32)); continue
                crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                crop = (crop-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
                crops.append(crop)
            while len(crops) < 16: crops.append(np.zeros((112,112,3),dtype=np.float32))
            
            clip = torch.tensor(np.transpose(np.stack(crops[:16]),(3,0,1,2))).unsqueeze(0).float()
            try:
                with torch.no_grad():
                    logits = model(clip)
                    probs = torch.softmax(logits,1)[0]
                pred_idx = probs.argmax().item()
                if n_classes == 9:
                    pred_raw = MVIT_9.get(pred_idx,"unknown")
                    pred_sar = MVIT_TO_SAR.get(pred_raw,"unknown")
                elif n_classes == 4:
                    pred_sar = RF_CLASSES[pred_idx]
                    pred_raw = pred_sar
                else:
                    pred_raw = f"class_{pred_idx}"; pred_sar = "unknown"
                seq_cache[str(tid)] = {
                    "predicted_sar": pred_sar, "predicted_raw": pred_raw,
                    "confidence": round(float(probs[pred_idx].item()),4),
                }
            except Exception as e:
                seq_cache[str(tid)] = {"predicted_sar":None,"error":str(e)}
        
        r3d_cache[seq] = seq_cache
        n_ok = sum(1 for v in seq_cache.values() if v.get("predicted_sar"))
        log(f'  {seq}: {n_ok} tracks classified')
    
    with open(os.path.join(OUT_DIR, "r3d18_predictions_cache.json"), "w") as f:
        json.dump(r3d_cache, f, indent=2)
    log('  Saved R3D-18 cache')
    
    del model; import gc; gc.collect()
    return r3d_cache

# ═══════════════════════════════════════════════════════════════════════
# TASK 2: Baseline Comparison Table
# ═══════════════════════════════════════════════════════════════════════
def task2_baseline(tracks, tms_results, r3d_cache):
    log('\n' + '='*60)
    log('TASK 2: Baseline Comparison Table')
    log('='*60)
    
    # Split data (same 70/30 as previous experiments)
    np.random.seed(42)
    idx = np.random.permutation(len(tracks))
    split = int(0.7 * len(idx))
    test_tracks = [tracks[i] for i in idx[split:]]
    train_tracks = [tracks[i] for i in idx[:split]]
    
    y_test = np.array([t["label"] for t in test_tracks])
    n_test = len(y_test)
    log(f'  Test set: {n_test} tracks')
    
    baselines = {}
    
    # 1. Random
    np.random.seed(42)
    rand_preds = np.random.randint(0, 4, n_test)
    baselines["Random"] = {
        "type": "—",
        "accuracy": float(accuracy_score(y_test, rand_preds)),
        "lying_recall": float(recall_score(y_test==0, rand_preds==0, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_test, rand_preds)),
    }
    
    # 2. Majority class
    majority_preds = np.ones(n_test, dtype=int)  # stationary = 1
    baselines["Majority Class"] = {
        "type": "—",
        "accuracy": float(accuracy_score(y_test, majority_preds)),
        "lying_recall": 0.0,
        "kappa": 0.0,
    }
    
    # 3. MViTv2-S (from cache)
    mvit_cache_path = "evaluation/real_data/full/end_to_end/mvit2s_predictions_cache.json"
    if os.path.exists(mvit_cache_path):
        with open(mvit_cache_path) as f: mc = json.load(f)
        # Aggregate all sequences
        mvit_map = {}
        for seq_data in mc.values():
            for tid, pred in seq_data.items():
                if pred.get("predicted_sar"):
                    mvit_map[tid] = pred["predicted_sar"]
        # Match to test tracks
        mvit_preds = []; mvit_true = []
        for t in test_tracks:
            if t["tid"] in mvit_map:
                mapped = RF_CLASSES.index(mvit_map[t["tid"]]) if mvit_map[t["tid"]] in RF_CLASSES else None
                if mapped is not None:
                    mvit_preds.append(mapped)
                    mvit_true.append(t["label"])
        if mvit_preds:
            mvit_true = np.array(mvit_true); mvit_preds = np.array(mvit_preds)
            baselines["MViTv2-S"] = {
                "type":"Pixel","n_eval":len(mvit_preds),
                "accuracy":float(accuracy_score(mvit_true, mvit_preds)),
                "lying_recall":float(recall_score(mvit_true==0, mvit_preds==0, zero_division=0)),
                "kappa":float(cohen_kappa_score(mvit_true, mvit_preds)),
            }
        else:
            baselines["MViTv2-S"] = {"type":"Pixel","accuracy":0.265,"lying_recall":0.0,"kappa":0.0,"note":"from prior eval"}
    
    # 4. R3D-18
    if r3d_cache:
        r3d_map = {}
        for seq_data in r3d_cache.values():
            for tid, pred in seq_data.items():
                if pred.get("predicted_sar"):
                    r3d_map[tid] = pred["predicted_sar"]
        r3d_preds = []; r3d_true = []
        for t in test_tracks:
            if t["tid"] in r3d_map and r3d_map[t["tid"]] in RF_CLASSES:
                r3d_preds.append(RF_CLASSES.index(r3d_map[t["tid"]]))
                r3d_true.append(t["label"])
        if r3d_preds:
            r3d_true = np.array(r3d_true); r3d_preds = np.array(r3d_preds)
            baselines["R3D-18"] = {
                "type":"Pixel","n_eval":len(r3d_preds),
                "accuracy":float(accuracy_score(r3d_true, r3d_preds)),
                "lying_recall":float(recall_score(r3d_true==0, r3d_preds==0, zero_division=0)),
                "kappa":float(cohen_kappa_score(r3d_true, r3d_preds)),
            }
    
    # 5. TMS-12 + RF
    from imblearn.over_sampling import SMOTE
    X12_tr = np.array([t["tms12"] for t in train_tracks])
    y_tr = np.array([t["label"] for t in train_tracks])
    X12_te = np.array([t["tms12"] for t in test_tracks])
    sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X12_tr, y_tr)
    rf12 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf12.fit(X_sm, y_sm)
    p12 = rf12.predict(X12_te)
    baselines["TMS-12 + RF"] = {
        "type":"Trajectory",
        "accuracy":float(accuracy_score(y_test, p12)),
        "lying_recall":float(recall_score(y_test==0, p12==0, zero_division=0)),
        "kappa":float(cohen_kappa_score(y_test, p12)),
    }
    
    # 6. TMS-16 + RF
    X16_tr = np.array([t["tms16"] for t in train_tracks])
    X16_te = np.array([t["tms16"] for t in test_tracks])
    sm16 = SMOTE(random_state=42); X16_sm, y16_sm = sm16.fit_resample(X16_tr, y_tr)
    rf16 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf16.fit(X16_sm, y16_sm)
    p16 = rf16.predict(X16_te)
    baselines["TMS-16 + RF"] = {
        "type":"Trajectory",
        "accuracy":float(accuracy_score(y_test, p16)),
        "lying_recall":float(recall_score(y_test==0, p16==0, zero_division=0)),
        "kappa":float(cohen_kappa_score(y_test, p16)),
    }
    
    # 7. TMS-16 + GBM
    gbm16 = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gbm16.fit(X16_sm, y16_sm)
    p16g = gbm16.predict(X16_te)
    baselines["TMS-16 + GBM"] = {
        "type":"Trajectory",
        "accuracy":float(accuracy_score(y_test, p16g)),
        "lying_recall":float(recall_score(y_test==0, p16g==0, zero_division=0)),
        "kappa":float(cohen_kappa_score(y_test, p16g)),
    }
    
    # 8. TrajMAE (from prior eval)
    baselines["TrajMAE"] = {
        "type":"Traj+SSL",
        "accuracy":0.246, "lying_recall":0.47, "kappa":0.02,
        "note":"from prior evaluation",
    }
    
    # 9. SCTE + Linear (from prior eval)
    baselines["SCTE + Linear"] = {
        "type":"Traj+CL",
        "accuracy":0.565, "lying_recall":0.0, "kappa":0.0,
        "note":"from scte_real_eval.py",
    }
    
    # Print table
    log(f'\n  {"Method":<20} {"Type":<10} {"Acc":>7} {"lying_R":>8} {"Kappa":>7}')
    log(f'  {"-"*55}')
    for name in ["Random","Majority Class","MViTv2-S","R3D-18","TMS-12 + RF","TMS-16 + RF","TMS-16 + GBM","TrajMAE","SCTE + Linear"]:
        b = baselines.get(name)
        if b:
            log(f'  {name:<20} {b.get("type","?"):<10} {b["accuracy"]:>7.1%} {b["lying_recall"]:>8.1%} {b["kappa"]:>7.3f}')
    
    # ── BASELINE COMPARISON FIGURE ──
    methods = [k for k in ["Random","Majority Class","MViTv2-S","R3D-18",
               "TMS-12 + RF","TMS-16 + RF","TMS-16 + GBM","TrajMAE","SCTE + Linear"]
               if k in baselines]
    accs = [baselines[m]["accuracy"] for m in methods]
    lyings = [baselines[m]["lying_recall"] for m in methods]
    kappas = [baselines[m]["kappa"] for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    type_colors = {"—":"#95a5a6","Pixel":"#e74c3c","Trajectory":"#3498db",
                   "Traj+SSL":"#9b59b6","Traj+CL":"#2ecc71"}
    bar_colors = [type_colors.get(baselines[m].get("type","—"),"grey") for m in methods]
    
    # Accuracy bars
    ax = axes[0]
    bars = ax.barh(range(len(methods)), accs, color=bar_colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=11); ax.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    for b, v in zip(bars, accs): ax.text(v+0.01, b.get_y()+b.get_height()/2, f'{v:.1%}', va='center', fontsize=9)
    ax.set_xlim(0, max(accs)*1.15); ax.axvline(0.25, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.invert_yaxis()
    
    # Lying recall bars
    ax = axes[1]
    bars = ax.barh(range(len(methods)), lyings, color=bar_colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Lying_down Recall', fontsize=11); ax.set_title('Casualty Detection Rate', fontsize=12, fontweight='bold')
    for b, v in zip(bars, lyings): ax.text(v+0.01, b.get_y()+b.get_height()/2, f'{v:.1%}', va='center', fontsize=9)
    ax.set_xlim(0, max(lyings)*1.15 if max(lyings)>0 else 1); ax.invert_yaxis()
    
    # Kappa bars
    ax = axes[2]
    bars = ax.barh(range(len(methods)), kappas, color=bar_colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Cohen's Kappa", fontsize=11); ax.set_title("Cohen's Kappa", fontsize=12, fontweight='bold')
    for b, v in zip(bars, kappas): ax.text(max(v+0.01,0.02), b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim(min(min(kappas),-0.05),max(kappas)*1.2 if max(kappas)>0 else 0.5); ax.invert_yaxis()
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, edgecolor='black', label=l) for l,c in type_colors.items()]
    fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5,-0.02))
    
    plt.suptitle('SARTriage Classification Baselines — Okutama-Action (4 classes)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, "baseline_comparison_table.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    return baselines

def main():
    with open('/tmp/baseline_progress.txt','w') as f: f.write('TMS-16 + Baseline Comparison\n')
    
    log('\nLoading Okutama tracks...')
    tracks = load_data()
    log(f'Loaded {len(tracks)} tracks')
    
    # Task 1
    tms_results, best = task1_tms16(tracks)
    with open(os.path.join(OUT_DIR, "tms16_results.json"), "w") as f:
        json.dump(tms_results, f, indent=2)
    
    # R3D-18 cache
    r3d = cache_r3d18(tracks)
    
    # Task 2
    baselines = task2_baseline(tracks, tms_results, r3d)
    with open(os.path.join(OUT_DIR, "baseline_comparison.json"), "w") as f:
        json.dump(baselines, f, indent=2, default=str)
    
    log('\n\nALL DONE.')

if __name__ == "__main__":
    main()
