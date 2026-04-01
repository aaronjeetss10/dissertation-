"""
TrajMAE FULL EVALUATION on Real Okutama Data (v2 — fixed)
Track A: Pre-train → fine-tune → evaluate
Track B: Few-shot learning curve
"""
import os, sys, json, math, time, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.abspath("."))
from evaluation.traj_mae import TrajMAE
from evaluation.real_data.tms12_standalone import extract_tms12

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

MAPPING = {"Standing":"stationary","Sitting":"stationary","Walking":"walking","Running":"running","Lying":"lying_down"}
CLASS_LIST = ["lying_down","stationary","walking","running"]
MAX_LEN = 50

def load_all_tracks():
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        oku = json.load(f)
    
    labelled = []
    unlabelled_tokens = []
    
    for tid, t in oku["tracks"].items():
        if t["track_length_frames"] < 20: continue
        c = t["centroids"]; b = t["bboxes"]
        # Delta tokens for TrajMAE
        tk = [[c[i][0]-c[i-1][0],c[i][1]-c[i-1][1],b[i][2]-b[i-1][2],b[i][3]-b[i-1][3]] for i in range(1,len(c))]
        if len(tk) < 5: continue
        
        arr = np.zeros((MAX_LEN,4),dtype=np.float32)
        for j in range(min(len(tk),MAX_LEN)): arr[j]=tk[j]
        unlabelled_tokens.append(arr)
        
        # Raw trajectory for TMS-12 (centroids + bbox)
        raw_traj = [[c[i][0],c[i][1],b[i][2],b[i][3]] for i in range(len(c))]
        
        act = t["primary_action"]
        if act in MAPPING:
            labelled.append({"tokens":arr,"raw_traj":raw_traj,"label":MAPPING[act],"size":t["mean_size_px"],"id":tid})
    
    # VisDrone unlabelled
    vd_path = "evaluation/real_data/visdrone_mot_tracks.json"
    if os.path.exists(vd_path):
        with open(vd_path) as f:
            vd = json.load(f)
        for tid, t in vd["tracks"].items():
            if t["track_length_frames"] < 20: continue
            c = t["centroids"]; b = t["bboxes"]
            tk = [[c[i][0]-c[i-1][0],c[i][1]-c[i-1][1],b[i][2]-b[i-1][2],b[i][3]-b[i-1][3]] for i in range(1,len(c))]
            if len(tk) < 5: continue
            arr = np.zeros((MAX_LEN,4),dtype=np.float32)
            for j in range(min(len(tk),MAX_LEN)): arr[j]=tk[j]
            unlabelled_tokens.append(arr)
    
    return labelled, np.array(unlabelled_tokens)

def make_model():
    return TrajMAE(num_classes=4, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=MAX_LEN)

def pretrain(model, X_all, epochs=30, lr=1e-3, bs=256, log=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(X_all))
        ep_loss = 0; nb = 0
        for i in range(0, len(X_all), bs):
            batch = X_all[perm[i:i+bs]]
            recon, target, ids = model(batch, pretrain=True)
            loss = nn.MSELoss()(recon, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item(); nb += 1
        scheduler.step()
        losses.append(ep_loss/nb)
        if log and (ep+1)%10==0: log(f'PT {ep+1}/{epochs}: loss={losses[-1]:.4f}')
    return losses

def finetune(model, X_train, y_train, epochs=20, lr=5e-4, bs=128, freeze_encoder_epochs=5):
    counts = Counter(y_train.numpy().tolist())
    total = len(y_train)
    weights = torch.tensor([total/(counts.get(i,1)*len(CLASS_LIST)) for i in range(len(CLASS_LIST))])
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Phase 1: freeze encoder, train head only
    for p in model.encoder.parameters(): p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr*5)
    model.train()
    for ep in range(freeze_encoder_epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), bs):
            idx = perm[i:i+bs]
            logits = model(X_train[idx], pretrain=False)
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    # Phase 2: unfreeze all, lower LR
    for p in model.encoder.parameters(): p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs - freeze_encoder_epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), bs):
            idx = perm[i:i+bs]
            logits = model(X_train[idx], pretrain=False)
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_test), 256):
            logits = model(X_test[i:i+256], pretrain=False)
            preds.extend(logits.argmax(1).numpy())
    return (np.array(preds)==y_test.numpy()).mean(), np.array(preds), y_test.numpy()

def log(msg):
    with open('/tmp/trajmae_progress.txt','a') as f: f.write(msg+'\n')

def main():
    with open('/tmp/trajmae_progress.txt','w') as f: f.write('Starting TrajMAE v2\n')
    
    labelled, all_tokens = load_all_tracks()
    log(f'Labelled: {len(labelled)}, Unlabelled: {len(all_tokens)}')
    
    np.random.seed(42)
    idx = np.random.permutation(len(labelled))
    split = int(0.7*len(labelled))
    train_data = [labelled[i] for i in idx[:split]]
    test_data = [labelled[i] for i in idx[split:]]
    
    X_train = torch.tensor(np.array([t["tokens"] for t in train_data]))
    y_train = torch.tensor([CLASS_LIST.index(t["label"]) for t in train_data])
    X_test = torch.tensor(np.array([t["tokens"] for t in test_data]))
    y_test = torch.tensor([CLASS_LIST.index(t["label"]) for t in test_data])
    X_all = torch.tensor(all_tokens)
    
    # Normalize
    mask = X_all != 0
    mean_val = X_all[mask].mean(); std_val = X_all[mask].std()
    X_all = (X_all - mean_val) / std_val
    X_train = (X_train - mean_val) / std_val
    X_test = (X_test - mean_val) / std_val
    
    log(f'Train: {len(X_train)} Test: {len(X_test)} All: {len(X_all)}')
    
    # ═══════════════════════════════════════════════════════════════════
    # TRACK A: Pre-train → Fine-tune → Evaluate
    # ═══════════════════════════════════════════════════════════════════
    log('TRACK A: Pre-training (30 epochs)...')
    t0 = time.time()
    model_pt = make_model()
    pt_losses = pretrain(model_pt, X_all, epochs=30, lr=1e-3, log=log)
    t_pretrain = time.time() - t0
    log(f'Pre-train: {t_pretrain:.1f}s, final_loss={pt_losses[-1]:.4f}')
    
    torch.save(model_pt.encoder.state_dict(), "evaluation/results/trajmae_encoder_pretrained.pt")
    log('Saved encoder weights.')
    
    # Fine-tune (with frozen encoder warm-up)
    t1 = time.time()
    model_ft = make_model()
    model_ft.encoder.load_state_dict(torch.load("evaluation/results/trajmae_encoder_pretrained.pt", weights_only=True))
    finetune(model_ft, X_train, y_train, epochs=20, freeze_encoder_epochs=5)
    t_finetune = time.time() - t1
    log(f'Fine-tune: {t_finetune:.1f}s')
    
    acc_pt, preds_pt, true_pt = evaluate(model_ft, X_test, y_test)
    
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
    kappa = cohen_kappa_score(true_pt, preds_pt)
    report = classification_report(true_pt, preds_pt, target_names=CLASS_LIST, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_pt, preds_pt)
    log(f'Pre-trained: acc={acc_pt*100:.1f}% kappa={kappa:.4f}')
    log(classification_report(true_pt, preds_pt, target_names=CLASS_LIST, zero_division=0))
    
    # From-scratch
    model_scratch = make_model()
    finetune(model_scratch, X_train, y_train, epochs=20, freeze_encoder_epochs=0)
    acc_scratch, _, _ = evaluate(model_scratch, X_test, y_test)
    log(f'From-scratch: acc={acc_scratch*100:.1f}%')
    
    # Size bins
    sizes = np.array([t["size"] for t in test_data])
    size_bins = {}
    for name, lo, hi in [("<50px",0,50),("50-75px",50,75),("75-100px",75,100),(">100px",100,9999)]:
        idx_b = [i for i in range(len(sizes)) if lo<=sizes[i]<hi]
        if idx_b: size_bins[name] = {"n":len(idx_b),"accuracy":float((preds_pt[idx_b]==true_pt[idx_b]).mean())}
    
    # ═══════════════════════════════════════════════════════════════════
    # TRACK B: Few-shot Learning Curve
    # ═══════════════════════════════════════════════════════════════════
    log('\nTRACK B: Few-shot learning curve...')
    
    # TMS-12 features from raw trajectories
    log('Extracting TMS-12 features...')
    tms_train = np.array([extract_tms12(t["raw_traj"]) for t in train_data])
    tms_test = np.array([extract_tms12(t["raw_traj"]) for t in test_data])
    tms_train = np.nan_to_num(tms_train, nan=0, posinf=0, neginf=0)
    tms_test = np.nan_to_num(tms_test, nan=0, posinf=0, neginf=0)
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    log('TMS-12 features extracted.')
    
    k_values = [5, 10, 20, 50, 100, len(train_data)]
    seeds = [42, 123, 456]
    results_curve = {k: {"pretrained":[], "scratch":[], "rf":[]} for k in k_values}
    
    for k in k_values:
        kname = "ALL" if k >= len(train_data) else str(k)
        log(f'  k={kname}...')
        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            
            sub_idx = []
            for ci in range(len(CLASS_LIST)):
                class_idx = [i for i in range(len(train_data)) if y_train_np[i]==ci]
                n_sample = min(k, len(class_idx))
                sub_idx.extend(np.random.choice(class_idx, n_sample, replace=n_sample>len(class_idx)).tolist())
            
            X_sub = X_train[sub_idx]; y_sub = y_train[sub_idx]
            
            # Pre-trained
            m1 = make_model()
            m1.encoder.load_state_dict(torch.load("evaluation/results/trajmae_encoder_pretrained.pt", weights_only=True))
            finetune(m1, X_sub, y_sub, epochs=20, freeze_encoder_epochs=5)
            a1, _, _ = evaluate(m1, X_test, y_test)
            results_curve[k]["pretrained"].append(a1)
            
            # Scratch
            m2 = make_model()
            finetune(m2, X_sub, y_sub, epochs=20, freeze_encoder_epochs=0)
            a2, _, _ = evaluate(m2, X_test, y_test)
            results_curve[k]["scratch"].append(a2)
            
            # RF
            from sklearn.ensemble import RandomForestClassifier
            tms_sub = tms_train[sub_idx]; y_sub_np = y_train_np[sub_idx]
            rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)
            rf.fit(tms_sub, y_sub_np)
            a3 = (rf.predict(tms_test)==y_test_np).mean()
            results_curve[k]["rf"].append(a3)
        
        log(f'    PT={np.mean(results_curve[k]["pretrained"])*100:.1f}±{np.std(results_curve[k]["pretrained"])*100:.1f}  '
            f'Scratch={np.mean(results_curve[k]["scratch"])*100:.1f}±{np.std(results_curve[k]["scratch"])*100:.1f}  '
            f'RF={np.mean(results_curve[k]["rf"])*100:.1f}±{np.std(results_curve[k]["rf"])*100:.1f}')
    
    # Save
    results = {
        "track_a": {
            "pretrained_accuracy": float(acc_pt), "scratch_accuracy": float(acc_scratch),
            "kappa": float(kappa), "report": report, "confusion_matrix": cm.tolist(),
            "size_bins": size_bins, "pretrain_time_s": t_pretrain, "finetune_time_s": t_finetune,
            "pretrain_losses": [float(l) for l in pt_losses],
            "n_train": len(train_data), "n_test": len(test_data), "n_unlabelled": len(all_tokens),
        },
        "track_b": {}
    }
    for k in k_values:
        kstr = "ALL" if k >= len(train_data) else str(k)
        results["track_b"][kstr] = {
            "pretrained": {"mean":float(np.mean(results_curve[k]["pretrained"])),"std":float(np.std(results_curve[k]["pretrained"]))},
            "scratch": {"mean":float(np.mean(results_curve[k]["scratch"])),"std":float(np.std(results_curve[k]["scratch"]))},
            "rf": {"mean":float(np.mean(results_curve[k]["rf"])),"std":float(np.std(results_curve[k]["rf"]))},
        }
    
    with open(os.path.join(FULL_DIR, "trajmae_real_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_LIST, rotation=45); ax.set_yticklabels(CLASS_LIST)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'TrajMAE Confusion (acc={acc_pt*100:.1f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR,"trajmae_confusion.png"),dpi=150); plt.close()
    
    # Few-shot curve
    fig, ax = plt.subplots(figsize=(9,5.5))
    k_plot = []
    pt_m, pt_s, sc_m, sc_s, rf_m, rf_s = [],[],[],[],[],[]
    for k in k_values:
        kstr = "ALL" if k >= len(train_data) else str(k)
        r = results["track_b"][kstr]
        k_plot.append(k)
        pt_m.append(r["pretrained"]["mean"]*100); pt_s.append(r["pretrained"]["std"]*100)
        sc_m.append(r["scratch"]["mean"]*100); sc_s.append(r["scratch"]["std"]*100)
        rf_m.append(r["rf"]["mean"]*100); rf_s.append(r["rf"]["std"]*100)
    
    ax.errorbar(k_plot, pt_m, yerr=pt_s, marker='o', lw=2.5, capsize=5, label='TrajMAE (pre-trained)', color='#27ae60', markersize=8)
    ax.errorbar(k_plot, sc_m, yerr=sc_s, marker='s', lw=2.5, capsize=5, label='TrajMAE (from scratch)', color='#e74c3c', markersize=8)
    ax.errorbar(k_plot, rf_m, yerr=rf_s, marker='^', lw=2.5, capsize=5, label='TMS-12 + RF (baseline)', color='#3498db', markersize=8)
    
    ax.set_xlabel('Labelled samples per class (k)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Few-Shot Learning Curve — Real Okutama-Action Data\nPre-training on 3,775 unlabelled trajectories (Okutama + VisDrone)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xscale('log')
    ax.set_xticks(k_plot)
    ax.set_xticklabels([str(k) if k<len(train_data) else 'ALL' for k in k_plot])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR,"trajmae_fewshot_curve.png"),dpi=200); plt.close()
    
    log('\nALL DONE.')

if __name__ == "__main__":
    main()
