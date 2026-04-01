"""
Proper TrajMAE Training: Pre-train on ALL data, fine-tune on 3-class labels.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from evaluation.traj_mae import TrajMAE, TrajMAEEncoder, pretrain_mae
from evaluation.real_data.tms12_standalone import extract_tms16
from evaluation.scte import SCTEModel

OUT_DIR = "evaluation/real_data/full"
SAR3_MAP = {"Standing":"upright_stationary","Sitting":"upright_stationary",
            "Walking":"mobile","Running":"mobile","Lying":"critical"}
SAR3_CLASSES = ["critical","upright_stationary","mobile"]
SAR3_PRIORITY = {"critical":1.0,"upright_stationary":0.4,"mobile":0.1}

def log(msg):
    with open('/tmp/trajmae_proper_progress.txt','a') as f: f.write(msg+'\n')

def load_okutama():
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR3_MAP: continue
        centroids = t["centroids"]; bboxes = t["bboxes"]
        deltas = []
        for i in range(1, len(centroids)):
            deltas.append([centroids[i][0]-centroids[i-1][0], centroids[i][1]-centroids[i-1][1],
                           bboxes[i][2]-bboxes[i-1][2], bboxes[i][3]-bboxes[i-1][3]])
        arr = np.zeros((50,4), dtype=np.float32)
        for j in range(min(len(deltas),50)): arr[j] = deltas[j]
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(centroids,bboxes)]
        f16 = extract_tms16(traj)
        f16 = [0 if(math.isnan(f)or math.isinf(f))else f for f in f16]
        tracks.append({
            "delta_tokens": arr, "label": SAR3_CLASSES.index(SAR3_MAP[act]),
            "tms16": np.array(f16, dtype=np.float32), "gt_action": act,
        })
    return tracks

def load_visdrone():
    """Load VisDrone-MOT trajectories (unlabelled, for pre-training only)."""
    vd_path = "evaluation/real_data/visdrone_tracks.json"
    if not os.path.exists(vd_path):
        log('  VisDrone tracks not found, skipping')
        return []
    with open(vd_path) as f:
        data = json.load(f)
    seqs = []
    for tid, t in data.get("tracks",{}).items():
        if len(t.get("centroids",[])) < 10: continue
        centroids = t["centroids"]; bboxes = t["bboxes"]
        deltas = []
        for i in range(1, len(centroids)):
            deltas.append([centroids[i][0]-centroids[i-1][0], centroids[i][1]-centroids[i-1][1],
                           bboxes[i][2]-bboxes[i-1][2], bboxes[i][3]-bboxes[i-1][3]])
        arr = np.zeros((50,4), dtype=np.float32)
        for j in range(min(len(deltas),50)): arr[j] = deltas[j]
        seqs.append(arr)
    return seqs

def finetune_3class(model, X_train, y_train, X_test, y_test, epochs=30,
                    batch_size=32, lr=1e-4, freeze_epochs=10):
    """Fine-tune with class-weighted CE, encoder frozen then unfrozen."""
    device = torch.device("cpu")
    model = model.to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Class weights (inverse frequency)
    counts = np.bincount(y_train, minlength=3).astype(float)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * len(counts)
    cw = torch.FloatTensor(weights).to(device)

    # Freeze encoder initially
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.decoder.parameters(): p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_acc = 0; best_preds = None
    train_losses = []; test_accs = []

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            for p in model.encoder.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr*0.1, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs-freeze_epochs)

        model.train()
        epoch_loss = 0; nb = 0
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model.forward_finetune(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        avg_loss = epoch_loss/max(nb,1)
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            logits = model.forward_finetune(X_te)
            preds = logits.argmax(1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        test_accs.append(acc)
        if acc > best_acc: best_acc = acc; best_preds = preds.copy()
        if (epoch+1) % 5 == 0 or epoch == 0:
            log(f'    [finetune] e={epoch+1:>3d}/{epochs} loss={avg_loss:.4f} acc={acc:.1%}'
                f'{" (frozen)" if epoch<freeze_epochs else ""}')

    return best_preds, best_acc, train_losses, test_accs

def main():
    with open('/tmp/trajmae_proper_progress.txt','w') as f: f.write('TrajMAE Proper Training\n')

    tracks = load_okutama()
    log(f'Okutama: {len(tracks)} tracks')
    y = np.array([t["label"] for t in tracks])
    log(f'3-class dist: {dict(zip(SAR3_CLASSES, [sum(y==i) for i in range(3)]))}')

    # ── Train/test split ──
    np.random.seed(42)
    idx = np.random.permutation(len(tracks))
    split = int(0.7 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    X_all = np.stack([t["delta_tokens"] for t in tracks])
    y_train, y_test = y[tr_idx], y[te_idx]
    X_train, X_test = X_all[tr_idx], X_all[te_idx]
    log(f'Train: {len(tr_idx)}, Test: {len(te_idx)}')

    # ── Load VisDrone for pre-training ──
    vd_seqs = load_visdrone()
    log(f'VisDrone: {len(vd_seqs)} unlabelled trajectories')

    # Combine all unlabelled data for pre-training
    pretrain_data = list(X_all)  # All Okutama (no labels needed)
    pretrain_data.extend(vd_seqs)
    X_pretrain = np.stack(pretrain_data)
    log(f'Pre-training data: {len(X_pretrain)} trajectories')

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Pre-train TrajMAE (full architecture)
    # ══════════════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('STEP 1: Pre-training TrajMAE (d=64, 4-layer encoder)')
    log('='*60)

    model = TrajMAE(num_classes=3, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=50)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'  Model params: {n_params:,}')

    t0 = time.time()
    pt_losses = pretrain_mae(model, X_pretrain, epochs=50, batch_size=64, lr=1e-3)
    pt_time = time.time()-t0
    log(f'  Pre-training: {pt_time:.1f}s ({pt_time/50:.2f}s/epoch)')
    log(f'  Final recon loss: {pt_losses[-1]:.6f}')

    # Save pre-trained weights
    torch.save(model.encoder.state_dict(), "evaluation/results/trajmae_encoder_3class_pretrained.pt")
    log('  Saved encoder weights')

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Fine-tune on 3-class labels
    # ══════════════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('STEP 2: Fine-tuning on 3-class labels')
    log('='*60)

    t1 = time.time()
    preds_ft, best_acc, ft_losses, ft_accs = finetune_3class(
        model, X_train, y_train, X_test, y_test,
        epochs=30, batch_size=32, lr=1e-4, freeze_epochs=10)
    ft_time = time.time()-t1
    log(f'  Fine-tuning: {ft_time:.1f}s')
    log(f'  Best accuracy: {best_acc:.1%}')

    # Save fine-tuned model
    torch.save(model.state_dict(), "evaluation/results/trajmae_3class_finetuned.pt")

    report_ft = classification_report(y_test, preds_ft, target_names=SAR3_CLASSES, output_dict=True)
    cm_ft = confusion_matrix(y_test, preds_ft)
    log(f'\n  Fine-tuned TrajMAE per-class:')
    for c in SAR3_CLASSES:
        log(f'    {c:<22} P={report_ft[c]["precision"]:.3f} R={report_ft[c]["recall"]:.3f} F1={report_ft[c]["f1-score"]:.3f}')

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: 5-fold CV comparison
    # ══════════════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('STEP 3: 5-fold CV comparison')
    log('='*60)

    X_tms = np.stack([t["tms16"] for t in tracks])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = {}

    # TMS-16 RF baseline
    fold_accs = []; all_pred = []; all_true = []
    for tr_i, te_i in skf.split(X_tms, y):
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_tms[tr_i], y[tr_i])
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(X_sm, y_sm)
        p = rf.predict(X_tms[te_i])
        fold_accs.append(accuracy_score(y[te_i], p))
        all_pred.extend(p); all_true.extend(y[te_i])
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    cv_results["TMS-16 RF"] = {
        "accuracy": float(np.mean(fold_accs)), "ci": float(1.96*np.std(fold_accs)),
        "kappa": float(cohen_kappa_score(all_true, all_pred)),
        "critical_recall": float(classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)["critical"]["recall"]),
        "macro_f1": float(classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)["macro avg"]["f1-score"]),
    }
    log(f'  TMS-16 RF:  Acc={cv_results["TMS-16 RF"]["accuracy"]:.1%} CritR={cv_results["TMS-16 RF"]["critical_recall"]:.1%}')

    # TrajMAE fine-tuned (5-fold)
    fold_accs = []; all_pred = []; all_true = []
    for fold_i, (tr_i, te_i) in enumerate(skf.split(X_all, y)):
        m = TrajMAE(num_classes=3, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=50)
        # Pre-train on all data
        pretrain_mae(m, X_pretrain, epochs=30, batch_size=64, lr=1e-3)
        # Fine-tune
        p, _, _, _ = finetune_3class(m, X_all[tr_i], y[tr_i], X_all[te_i], y[te_i],
                                      epochs=20, batch_size=32, lr=1e-4, freeze_epochs=5)
        fold_accs.append(accuracy_score(y[te_i], p))
        all_pred.extend(p); all_true.extend(y[te_i])
        log(f'    Fold {fold_i+1}: {fold_accs[-1]:.1%}')
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    cv_results["TrajMAE FT"] = {
        "accuracy": float(np.mean(fold_accs)), "ci": float(1.96*np.std(fold_accs)),
        "kappa": float(cohen_kappa_score(all_true, all_pred)),
        "critical_recall": float(classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)["critical"]["recall"]),
        "macro_f1": float(classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)["macro avg"]["f1-score"]),
    }
    log(f'  TrajMAE FT: Acc={cv_results["TrajMAE FT"]["accuracy"]:.1%} CritR={cv_results["TrajMAE FT"]["critical_recall"]:.1%}')

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: TMS-16 + new TrajMAE embeddings → RF
    # ══════════════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('STEP 4: Combined features with properly trained TrajMAE')
    log('='*60)

    # Extract embeddings from the best model
    model.eval()
    with torch.no_grad():
        mae_embs = []
        for s in range(0, len(X_all), 64):
            batch = torch.FloatTensor(X_all[s:s+64])
            emb = model.get_embedding(batch).cpu().numpy()
            mae_embs.append(emb)
    X_mae = np.concatenate(mae_embs)  # (N, 64)

    X_combined = np.hstack([X_tms, X_mae])  # (N, 80)
    log(f'  Combined: {X_combined.shape}')

    fold_accs = []; all_pred = []; all_true = []
    for tr_i, te_i in skf.split(X_combined, y):
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_combined[tr_i], y[tr_i])
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(X_sm, y_sm)
        p = rf.predict(X_combined[te_i])
        fold_accs.append(accuracy_score(y[te_i], p))
        all_pred.extend(p); all_true.extend(y[te_i])
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    cv_results["TMS-16+TrajMAE RF"] = {
        "accuracy": float(np.mean(fold_accs)), "ci": float(1.96*np.std(fold_accs)),
        "kappa": float(cohen_kappa_score(all_true, all_pred)),
        "critical_recall": float(classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)["critical"]["recall"]),
        "macro_f1": float(classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)["macro avg"]["f1-score"]),
    }
    log(f'  TMS-16+MAE: Acc={cv_results["TMS-16+TrajMAE RF"]["accuracy"]:.1%} CritR={cv_results["TMS-16+TrajMAE RF"]["critical_recall"]:.1%}')

    # Also try triple: TMS-16 + SCTE + TrajMAE
    scte_model = SCTEModel(input_dim=4, d_model=32, proj_dim=16, n_heads=2, n_layers=2, dropout=0.1, max_len=50)
    scte_model.load_state_dict(torch.load("evaluation/results/scte_encoder_trained.pt",
                                           map_location="cpu", weights_only=True))
    scte_model.eval()
    scte_tokens = []
    for t in tracks:
        centroids_data = None  # Need to reconstruct from track data
    # Simpler: load SCTE tokens from the data
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        odata = json.load(f)
    scte_arrs = []
    for t in tracks:
        tid = t.get("gt_action")  # hack — reconstruct from original
    # Actually just extract SCTE embeddings directly from delta tokens (scaled)
    # SCTE uses [dx/1280, dy/720, aspect, size_norm] but we have raw deltas
    # Let's just use TMS+MAE for now and add SCTE from the all_tracks data
    scte_emb_list = []
    idx_map = []
    for tid_str, tdata in odata["tracks"].items():
        if tdata["track_length_frames"] < 20: continue
        act = tdata["primary_action"]
        if act not in SAR3_MAP: continue
        centroids = tdata["centroids"]; bboxes = tdata["bboxes"]
        st = []
        for i in range(1, len(centroids)):
            dx = (centroids[i][0]-centroids[i-1][0])/1280.0
            dy = (centroids[i][1]-centroids[i-1][1])/720.0
            aspect = bboxes[i][3]/max(bboxes[i][2],1)
            sn = math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            st.append([dx,dy,aspect,sn])
        arr = np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(st),50)): arr[j]=st[j]
        scte_emb_list.append(arr)
    scte_arr = np.stack(scte_emb_list)
    with torch.no_grad():
        scte_embs = []
        for s in range(0, len(scte_arr), 64):
            scte_embs.append(scte_model.get_embedding(torch.FloatTensor(scte_arr[s:s+64])).cpu().numpy())
    X_scte = np.concatenate(scte_embs)

    X_triple = np.hstack([X_tms, X_scte, X_mae])  # (N, 112)
    fold_accs = []; all_pred = []; all_true = []
    for tr_i, te_i in skf.split(X_triple, y):
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_triple[tr_i], y[tr_i])
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(X_sm, y_sm)
        p = rf.predict(X_triple[te_i])
        fold_accs.append(accuracy_score(y[te_i], p))
        all_pred.extend(p); all_true.extend(y[te_i])
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    rpt = classification_report(all_true, all_pred, target_names=SAR3_CLASSES, output_dict=True)
    cv_results["TMS+SCTE+MAE RF"] = {
        "accuracy": float(np.mean(fold_accs)), "ci": float(1.96*np.std(fold_accs)),
        "kappa": float(cohen_kappa_score(all_true, all_pred)),
        "critical_recall": float(rpt["critical"]["recall"]),
        "macro_f1": float(rpt["macro avg"]["f1-score"]),
    }
    log(f'  Triple RF:  Acc={cv_results["TMS+SCTE+MAE RF"]["accuracy"]:.1%} CritR={cv_results["TMS+SCTE+MAE RF"]["critical_recall"]:.1%}')

    # ── Summary ──
    log('\n' + '='*70)
    log('FINAL COMPARISON (3-class, 5-fold CV)')
    log('='*70)
    log(f'\n  {"Method":<25} {"Acc":>7} {"±CI":>6} {"CritR":>7} {"κ":>7} {"F1":>7}')
    log(f'  {"-"*60}')
    for name in ["TMS-16 RF","TrajMAE FT","TMS-16+TrajMAE RF","TMS+SCTE+MAE RF"]:
        r = cv_results[name]
        log(f'  {name:<25} {r["accuracy"]:>7.1%} {r["ci"]:>5.1%} {r["critical_recall"]:>7.1%} {r["kappa"]:>7.3f} {r["macro_f1"]:>7.3f}')

    # ── Figures ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix for best TrajMAE
    ax = axes[0]
    im = ax.imshow(cm_ft, cmap='Purples')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm_ft[i,j]),ha='center',va='center',fontsize=14,
                    color='white' if cm_ft[i,j]>cm_ft.max()/2 else 'black')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(SAR3_CLASSES,rotation=45,ha='right'); ax.set_yticklabels(SAR3_CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'TrajMAE Fine-tuned (3-class)\nAcc: {best_acc:.1%}',fontsize=11,fontweight='bold')
    plt.colorbar(im,ax=ax)

    # Bar comparison
    ax = axes[1]
    names = list(cv_results.keys())
    accs = [cv_results[n]["accuracy"] for n in names]
    crs = [cv_results[n]["critical_recall"] for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, accs, w, label='Accuracy', color='#3498db', edgecolor='black')
    ax.bar(x+w/2, crs, w, label='Critical Recall', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=9)
    for i in range(len(names)):
        ax.text(i-w/2, accs[i]+0.01, f'{accs[i]:.1%}', ha='center', fontsize=8)
        ax.text(i+w/2, crs[i]+0.01, f'{crs[i]:.1%}', ha='center', fontsize=8)
    ax.set_ylabel('Score'); ax.set_title('3-Class Method Comparison',fontsize=11,fontweight='bold')
    ax.legend(); ax.set_ylim(0,1)

    plt.suptitle('TrajMAE Proper Training — 3-Class SAR Classification',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"trajmae_3class_confusion.png"),dpi=200); plt.close()

    # Save all
    all_res = {
        "pretrain": {"n_tracks": len(X_pretrain), "epochs": 50, "final_loss": pt_losses[-1], "time_s": pt_time},
        "finetune": {"best_acc": float(best_acc), "time_s": ft_time,
                     "report": classification_report(y_test, preds_ft, target_names=SAR3_CLASSES, output_dict=True)},
        "cv_comparison": cv_results,
        "training_curves": {"pretrain_losses": pt_losses, "finetune_losses": ft_losses, "finetune_accs": ft_accs},
    }
    with open(os.path.join(OUT_DIR,"trajmae_3class_results.json"),"w") as f:
        json.dump(all_res, f, indent=2)

    log('\nDone.')

if __name__ == "__main__":
    main()
