"""
SCTE Real-Data Training + Cross-Scale Evaluation on Okutama Tracks
Trains contrastive embeddings, evaluates cross-scale transfer.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12
from evaluation.scte import SCTEModel, InfoNCELoss, altitude_scale_trajectory

OUT_DIR = "evaluation/real_data/full"
os.makedirs(OUT_DIR, exist_ok=True)

SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
RF_CLASSES = ["lying_down","stationary","walking","running"]

def log(msg):
    with open('/tmp/scte_progress.txt','a') as f: f.write(msg+'\n')

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Extract real trajectories and create contrastive pairs
# ═══════════════════════════════════════════════════════════════════════

def load_okutama_trajectories():
    """Load all Okutama tracks, compute delta tokens and metadata."""
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    
    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR_MAP: continue
        sar_class = SAR_MAP[act]
        
        centroids = t["centroids"]
        bboxes = t["bboxes"]
        sizes = [math.sqrt(max(b[2],1)*max(b[3],1)) for b in bboxes]
        mean_size = np.mean(sizes)
        
        # Compute delta tokens (dx, dy, dw, dh) normalised by frame size
        deltas = []
        for i in range(1, len(centroids)):
            dx = (centroids[i][0] - centroids[i-1][0]) / 1280.0
            dy = (centroids[i][1] - centroids[i-1][1]) / 720.0
            aspect = bboxes[i][3] / max(bboxes[i][2], 1)
            size_norm = math.sqrt(bboxes[i][2]*bboxes[i][3]) / 200.0
            deltas.append([dx, dy, aspect, size_norm])
        
        if len(deltas) < 10: continue
        
        # Pad/truncate to 50 timesteps
        arr = np.zeros((50, 4), dtype=np.float32)
        for j in range(min(len(deltas), 50)):
            arr[j] = deltas[j]
        
        tracks.append({
            "tid": tid,
            "tokens": arr,
            "sar_class": sar_class,
            "class_idx": RF_CLASSES.index(sar_class),
            "mean_size": mean_size,
            "gt_action": act,
            "raw_traj": [[c[0],c[1],b[2],b[3]] for c,b in zip(centroids, bboxes)],
        })
    
    return tracks

def create_contrastive_pairs(tracks, min_pairs=30):
    """Create positive pairs from real data: same action at different scales."""
    by_class = defaultdict(list)
    for t in tracks:
        by_class[t["sar_class"]].append(t)
    
    log('\nScale distribution per class:')
    for cls, ts in by_class.items():
        sizes = [t["mean_size"] for t in ts]
        large = sum(1 for s in sizes if s > 50)
        small = sum(1 for s in sizes if s < 50)
        log(f'  {cls}: {len(ts)} tracks, large(>50px)={large}, small(<50px)={small}, range=[{min(sizes):.0f}-{max(sizes):.0f}px]')
    
    # For Okutama, most tracks are 15-40px range (drone footage)
    # Use median split instead of 50/90px thresholds
    all_sizes = [t["mean_size"] for t in tracks]
    median_size = np.median(all_sizes)
    q25 = np.percentile(all_sizes, 25)
    q75 = np.percentile(all_sizes, 75)
    log(f'\nSize stats: median={median_size:.0f}, Q25={q25:.0f}, Q75={q75:.0f}')
    
    # Split: "smaller" = below median, "larger" = above median
    pairs = []  # (anchor_tokens, positive_tokens, class_idx)
    for cls_idx, cls in enumerate(RF_CLASSES):
        ts = by_class[cls]
        smaller = [t for t in ts if t["mean_size"] <= median_size]
        larger = [t for t in ts if t["mean_size"] > median_size]
        
        real_pairs = 0
        # Create natural cross-scale pairs
        for s in smaller:
            for l in larger:
                pairs.append((s["tokens"], l["tokens"], cls_idx))
                real_pairs += 1
                if real_pairs >= min_pairs * 3: break
            if real_pairs >= min_pairs * 3: break
        
        # Augment if needed: scale-simulate from existing tracks
        while real_pairs < min_pairs:
            t = ts[np.random.randint(len(ts))]
            # Create altitude-scaled version
            scaled = altitude_scale_trajectory(
                t["tokens"], source_alt=50.0,
                target_alt=np.random.uniform(30, 200),
                noise_std=0.001
            )
            pairs.append((t["tokens"], scaled, cls_idx))
            real_pairs += 1
        
        log(f'  {cls}: {real_pairs} pairs created')
    
    return pairs

class ContrastivePairDataset(Dataset):
    """Dataset yielding (anchor, positive, label) for InfoNCE training."""
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        a, p, l = self.pairs[idx]
        # Add noise augmentation
        a_aug = a.copy() + np.random.normal(0, 0.001, a.shape).astype(np.float32)
        p_aug = p.copy() + np.random.normal(0, 0.001, p.shape).astype(np.float32)
        return torch.FloatTensor(a_aug), torch.FloatTensor(p_aug), l

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Train SCTE
# ═══════════════════════════════════════════════════════════════════════

def train_scte_real(model, pairs, epochs=50, batch_size=64, lr=3e-4, temperature=0.07):
    """Train SCTE on real contrastive pairs."""
    dataset = ContrastivePairDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0; n_bat = 0
        for anchors, positives, labels in loader:
            _, z_a = model(anchors)
            _, z_p = model(positives)
            loss = criterion(z_a, z_p, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_bat += 1
        
        scheduler.step()
        avg = epoch_loss / max(n_bat, 1)
        losses.append(avg)
        if (epoch+1) % 10 == 0 or epoch == 0:
            log(f'  Epoch {epoch+1:3d}/{epochs}  InfoNCE={avg:.4f}')
    
    return losses

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Evaluate
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, tracks):
    """Extract SCTE embeddings for all tracks."""
    model.eval()
    tokens = np.stack([t["tokens"] for t in tracks])
    x = torch.FloatTensor(tokens)
    embs = []
    for i in range(0, len(x), 64):
        batch = x[i:i+64]
        embs.append(model.get_embedding(batch).cpu().numpy())
    return np.concatenate(embs, axis=0)

def main():
    with open('/tmp/scte_progress.txt','w') as f: f.write('SCTE Real-Data Training\n')
    
    # Step 1: Load data
    log('\n=== STEP 1: Load Okutama trajectories ===')
    tracks = load_okutama_trajectories()
    log(f'Loaded {len(tracks)} valid tracks')
    
    class_dist = defaultdict(int)
    for t in tracks: class_dist[t["sar_class"]] += 1
    log(f'Class distribution: {dict(class_dist)}')
    
    # Split into train/test (stratified)
    np.random.seed(42)
    by_class = defaultdict(list)
    for t in tracks: by_class[t["sar_class"]].append(t)
    
    train_tracks = []; test_tracks = []
    for cls, ts in by_class.items():
        np.random.shuffle(ts)
        split = int(0.7 * len(ts))
        train_tracks.extend(ts[:split])
        test_tracks.extend(ts[split:])
    
    log(f'Train: {len(train_tracks)}, Test: {len(test_tracks)}')
    
    # Create contrastive pairs from training set
    log('\n=== Creating contrastive pairs ===')
    pairs = create_contrastive_pairs(train_tracks, min_pairs=50)
    log(f'Total pairs: {len(pairs)}')
    
    # Step 2: Train SCTE
    log('\n=== STEP 2: Train SCTE (reduced architecture) ===')
    model = SCTEModel(
        input_dim=4, d_model=32, proj_dim=16,
        n_heads=2, n_layers=2, dropout=0.1, max_len=50
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f'Model params: {n_params:,}')
    
    t0 = time.time()
    losses = train_scte_real(model, pairs, epochs=50, batch_size=64, lr=3e-4, temperature=0.07)
    train_time = time.time() - t0
    log(f'Training: {train_time:.1f}s, final InfoNCE={losses[-1]:.4f}')
    
    # Save weights
    torch.save(model.state_dict(), "evaluation/results/scte_encoder_trained.pt")
    log('Saved weights: evaluation/results/scte_encoder_trained.pt')
    
    # Step 3a: Extract embeddings
    log('\n=== STEP 3a: Linear probe on SCTE embeddings ===')
    train_embs = extract_embeddings(model, train_tracks)
    test_embs = extract_embeddings(model, test_tracks)
    
    y_train = np.array([t["class_idx"] for t in train_tracks])
    y_test = np.array([t["class_idx"] for t in test_tracks])
    
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(train_embs, y_train)
    preds = clf.predict(test_embs)
    scte_acc = accuracy_score(y_test, preds)
    log(f'SCTE linear probe accuracy: {scte_acc:.4f}')
    log(classification_report(y_test, preds, target_names=RF_CLASSES))
    
    # Step 3b: TMS-12 + RF baseline for comparison
    log('\n=== STEP 3b: TMS-12 + RF baseline ===')
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    
    def get_tms12(tracks_list):
        X = []; valid = []
        for t in tracks_list:
            feats = extract_tms12(t["raw_traj"])
            feats = [0 if (math.isnan(f) or math.isinf(f)) else f for f in feats]
            X.append(feats); valid.append(True)
        return np.array(X)
    
    X_train_tms = get_tms12(train_tracks)
    X_test_tms = get_tms12(test_tracks)
    
    sm = SMOTE(random_state=42); X_sm, y_sm = sm.fit_resample(X_train_tms, y_train)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_sm, y_sm)
    rf_preds = rf.predict(X_test_tms)
    tms_acc = accuracy_score(y_test, rf_preds)
    log(f'TMS-12 + RF accuracy: {tms_acc:.4f}')
    
    # Step 3c: Cross-scale transfer test (KEY EXPERIMENT)
    log('\n=== STEP 3c: Cross-scale transfer test ===')
    all_sizes = [t["mean_size"] for t in tracks]
    median_size = np.median(all_sizes)
    
    large_train = [t for t in train_tracks if t["mean_size"] > median_size]
    small_test = [t for t in test_tracks if t["mean_size"] <= median_size]
    
    if len(large_train) > 10 and len(small_test) > 10:
        y_large = np.array([t["class_idx"] for t in large_train])
        y_small = np.array([t["class_idx"] for t in small_test])
        
        # SCTE cross-scale
        large_embs = extract_embeddings(model, large_train)
        small_embs = extract_embeddings(model, small_test)
        
        clf_cross = LogisticRegression(max_iter=1000, random_state=42)
        clf_cross.fit(large_embs, y_large)
        scte_cross_preds = clf_cross.predict(small_embs)
        scte_cross_acc = accuracy_score(y_small, scte_cross_preds)
        
        # TMS-12 cross-scale
        X_large_tms = get_tms12(large_train)
        X_small_tms = get_tms12(small_test)
        
        # Need SMOTE if classes are imbalanced
        try:
            sm2 = SMOTE(random_state=42)
            X_l_sm, y_l_sm = sm2.fit_resample(X_large_tms, y_large)
        except:
            X_l_sm, y_l_sm = X_large_tms, y_large
        
        rf_cross = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf_cross.fit(X_l_sm, y_l_sm)
        tms_cross_preds = rf_cross.predict(X_small_tms)
        tms_cross_acc = accuracy_score(y_small, tms_cross_preds)
        
        log(f'\nCROSS-SCALE TRANSFER (train on large >{median_size:.0f}px, test on small ≤{median_size:.0f}px)')
        log(f'  Large train tracks: {len(large_train)}')
        log(f'  Small test tracks:  {len(small_test)}')
        log(f'  SCTE cross-scale accuracy:  {scte_cross_acc:.4f}')
        log(f'  TMS-12 cross-scale accuracy: {tms_cross_acc:.4f}')
        log(f'  Δ = {(scte_cross_acc-tms_cross_acc)*100:+.1f}pp')
        
        crossscale = {
            "median_size_px": float(median_size),
            "large_train_n": len(large_train),
            "small_test_n": len(small_test),
            "scte_cross_scale_acc": float(scte_cross_acc),
            "tms12_cross_scale_acc": float(tms_cross_acc),
            "delta_pp": float((scte_cross_acc-tms_cross_acc)*100),
        }
    else:
        log('Not enough tracks for cross-scale split')
        crossscale = {"error": "insufficient tracks"}
    
    # Step 3d: t-SNE visualisation
    log('\n=== STEP 3d: t-SNE visualisation ===')
    all_embs = np.concatenate([train_embs, test_embs])
    all_labels = np.concatenate([y_train, y_test])
    all_sizes_arr = np.array([t["mean_size"] for t in train_tracks + test_tracks])
    
    # Subsample for t-SNE if too many
    n_tsne = min(1000, len(all_embs))
    idx = np.random.choice(len(all_embs), n_tsne, replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(all_embs[idx])
    
    # Plot 1: Coloured by action
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c','#3498db','#2ecc71','#f39c12']
    for ci, cls in enumerate(RF_CLASSES):
        mask = all_labels[idx] == ci
        ax.scatter(coords[mask,0], coords[mask,1], c=colors[ci], label=cls, alpha=0.6, s=15)
    ax.legend(fontsize=10); ax.set_title('SCTE Embeddings — Coloured by Action', fontsize=13)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scte_tsne_by_action.png"), dpi=200)
    plt.close()
    
    # Plot 2: Coloured by size
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:,0], coords[:,1], c=all_sizes_arr[idx], cmap='viridis', alpha=0.6, s=15)
    plt.colorbar(sc, label='Person Size (px)')
    ax.set_title('SCTE Embeddings — Coloured by Person Size', fontsize=13)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scte_tsne_by_scale.png"), dpi=200)
    plt.close()
    
    # Also do TMS-12 t-SNE for comparison
    all_tms = np.concatenate([X_train_tms, X_test_tms])
    tsne_tms = TSNE(n_components=2, random_state=42, perplexity=30)
    coords_tms = tsne_tms.fit_transform(all_tms[idx])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ci, cls in enumerate(RF_CLASSES):
        mask = all_labels[idx] == ci
        axes[0].scatter(coords[mask,0], coords[mask,1], c=colors[ci], label=cls, alpha=0.6, s=15)
        axes[1].scatter(coords_tms[mask,0], coords_tms[mask,1], c=colors[ci], label=cls, alpha=0.6, s=15)
    axes[0].set_title('SCTE Embeddings', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[1].set_title('TMS-12 Features', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    plt.suptitle('Embedding Space Comparison — Action Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scte_vs_tms12_tsne.png"), dpi=200)
    plt.close()
    
    # Save all results
    results = {
        "n_tracks": len(tracks), "n_train": len(train_tracks), "n_test": len(test_tracks),
        "n_contrastive_pairs": len(pairs),
        "model_params": n_params, "train_time_s": train_time,
        "final_infonce": float(losses[-1]),
        "scte_linear_probe_acc": float(scte_acc),
        "tms12_rf_acc": float(tms_acc),
        "cross_scale": crossscale,
        "training_losses": [float(l) for l in losses],
    }
    
    with open(os.path.join(OUT_DIR, "scte_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(OUT_DIR, "scte_crossscale.json"), "w") as f:
        json.dump(crossscale, f, indent=2)
    
    # Final summary
    log('\n' + '='*60)
    log('SCTE EVALUATION SUMMARY')
    log('='*60)
    log(f'  Contrastive pairs:     {len(pairs)}')
    log(f'  Training time:         {train_time:.1f}s')
    log(f'  Final InfoNCE:         {losses[-1]:.4f}')
    log(f'  SCTE linear probe acc: {scte_acc:.4f}')
    log(f'  TMS-12+RF acc:         {tms_acc:.4f}')
    if "scte_cross_scale_acc" in crossscale:
        log(f'  SCTE cross-scale:      {crossscale["scte_cross_scale_acc"]:.4f}')
        log(f'  TMS-12 cross-scale:    {crossscale["tms12_cross_scale_acc"]:.4f}')
    log('\nDone.')

if __name__ == "__main__":
    main()
