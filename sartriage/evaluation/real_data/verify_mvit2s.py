"""
MViTv2-S VERIFICATION: Class mapping, input pipeline, and prediction diagnostics.
"""
import os, sys, json, math, glob
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import defaultdict, Counter

TEST_FRAMES_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_FRAMES_BASE, "Labels", "MultiActionLabels", "3840x2160")
SCALE_X = 1280.0 / 3840.0
SCALE_Y = 720.0 / 2160.0

# ── 1. CLASS MAPPING CHECK ──────────────────────────────────────────────
def check_class_mapping():
    print("="*70)
    print("1. CLASS MAPPING CHECK")
    print("="*70, flush=True)
    
    model_path = "models/action_mvit2_sar.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    print(f"\nModel file: {model_path}")
    print(f"Training epochs: {checkpoint.get('epoch')}")
    print(f"Val accuracy: {checkpoint.get('val_acc', 0)*100:.2f}%")
    print(f"Val loss: {checkpoint.get('val_loss', 0):.4f}")
    print(f"Num classes: {checkpoint.get('num_classes')}")
    print(f"Clip frames: {checkpoint.get('clip_frames')}")
    print(f"Clip size: {checkpoint.get('clip_size')}")
    
    # The DEFINITIVE class mapping from training
    labels = checkpoint.get('labels', [])
    idx_to_label = checkpoint.get('idx_to_label', {})
    label_to_idx = checkpoint.get('label_to_idx', {})
    
    print(f"\n--- DEFINITIVE 9-class mapping from checkpoint ---")
    SAR_9 = {}
    for i in range(9):
        lbl = idx_to_label.get(str(i), idx_to_label.get(i, labels[i] if i < len(labels) else "?"))
        SAR_9[i] = lbl
        print(f"  Index {i}: {lbl}")
    
    # SAR 4-class mapping
    MVIT_TO_SAR = {
        "falling":"lying_down", "crawling":"lying_down", "lying_down":"lying_down",
        "running":"running", "waving_hand":"stationary", "climbing":"walking",
        "stumbling":"walking", "pushing":"stationary", "pulling":"stationary"
    }
    
    print(f"\n--- 9→4 SAR mapping ---")
    for src, dst in MVIT_TO_SAR.items():
        print(f"  {src:15s} → {dst}")
    
    # Okutama action labels
    OKU_TO_SAR = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
                  "Running":"running","Lying":"lying_down","Carrying":"stationary",
                  "Pushing/Pulling":"stationary","Reading":"stationary",
                  "Calling":"stationary","Drinking":"stationary"}
    
    print(f"\n--- Okutama label → SAR mapping ---")
    for src, dst in OKU_TO_SAR.items():
        print(f"  {src:20s} → {dst}")
    
    # Check: what % of MViTv2 training classes overlap with Okutama test labels?
    mvit_sar = set(MVIT_TO_SAR.values())
    oku_sar = set(OKU_TO_SAR.values())
    print(f"\n--- Overlap check ---")
    print(f"  MViTv2 SAR classes: {mvit_sar}")
    print(f"  Okutama SAR classes: {oku_sar}")
    print(f"  Overlap: {mvit_sar & oku_sar}")
    print(f"  In MViTv2 not Okutama: {mvit_sar - oku_sar}")
    print(f"  In Okutama not MViTv2: {oku_sar - mvit_sar}")
    
    return SAR_9, MVIT_TO_SAR, OKU_TO_SAR, checkpoint


# ── BUILD TEST DATA ─────────────────────────────────────────────────────
def build_test_tracks(OKU_TO_SAR):
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    all_tracks = defaultdict(lambda: {"frames":[],"bboxes":[],"action":None,"seq":None,"dir":None})
    
    for lf in label_files:
        seq_name = os.path.basename(lf).replace(".txt","")
        matches = glob.glob(os.path.join(TEST_FRAMES_BASE, "**", seq_name), recursive=True)
        frame_dir = None
        for m in matches:
            if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
        if not frame_dir: continue
        
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 11: continue
                tid = int(parts[0])
                xmin = float(parts[1])*SCALE_X; ymin = float(parts[2])*SCALE_Y
                xmax = float(parts[3])*SCALE_X; ymax = float(parts[4])*SCALE_Y
                fi = int(parts[5])
                action1 = parts[10].strip('"')
                
                tk = f"{seq_name}_{tid}"
                all_tracks[tk]["frames"].append(fi)
                all_tracks[tk]["bboxes"].append([xmin, ymin, xmax-xmin, ymax-ymin])
                all_tracks[tk]["action"] = action1
                all_tracks[tk]["seq"] = seq_name
                all_tracks[tk]["dir"] = frame_dir
    
    valid = {}
    for tk, tv in all_tracks.items():
        if len(tv["frames"]) < 16: continue
        if tv["action"] in OKU_TO_SAR:
            tv["mean_size"] = np.mean([math.sqrt(max(b[2],1)*max(b[3],1)) for b in tv["bboxes"]])
            valid[tk] = tv
    
    print(f"\nTest set: {len(all_tracks)} raw tracks → {len(valid)} valid")
    
    # Distributions
    gt_dist = Counter(tv["action"] for tv in valid.values())
    print(f"\nGT distribution (Okutama labels):")
    for act, n in gt_dist.most_common():
        print(f"  {act:20s} → {OKU_TO_SAR[act]:12s}: {n:4d}")
    
    sar_dist = Counter(OKU_TO_SAR[tv["action"]] for tv in valid.values())
    print(f"\nSAR 4-class distribution:")
    for cls, n in sar_dist.most_common():
        print(f"  {cls:12s}: {n:4d} ({n/len(valid)*100:.1f}%)")
    
    sizes = [tv["mean_size"] for tv in valid.values()]
    print(f"\nSize distribution:")
    for lo, hi, label in [(0,50,"<50px"),(50,75,"50-75px"),(75,100,"75-100px"),(100,9999,">100px")]:
        n = sum(1 for s in sizes if lo<=s<hi)
        print(f"  {label:10s}: {n:4d} ({n/len(sizes)*100:.1f}%)")
    
    return valid


# ── LOAD MODEL ──────────────────────────────────────────────────────────
def load_model(checkpoint):
    from torchvision.models.video import mvit_v2_s
    model = mvit_v2_s()
    model.head[1] = nn.Linear(model.head[1].in_features, 9)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


# ── EXTRACT CLIP ────────────────────────────────────────────────────────
def extract_clip(tv):
    frames = tv["frames"]; bboxes = tv["bboxes"]
    start = max(0, len(frames)//2 - 8)
    
    crops = []; crop_sizes = []
    for i in range(start, min(start+16, len(frames))):
        fi = frames[i]; bb = bboxes[i]
        img_path = os.path.join(tv["dir"], f"{fi}.jpg")
        if not os.path.exists(img_path):
            crops.append(np.zeros((224,224,3), dtype=np.float32)); crop_sizes.append((0,0)); continue
        
        img = cv2.imread(img_path)
        if img is None:
            crops.append(np.zeros((224,224,3), dtype=np.float32)); crop_sizes.append((0,0)); continue
        
        x, y, w, h = int(bb[0]), int(bb[1]), int(max(bb[2],1)), int(max(bb[3],1))
        x = max(0, x); y = max(0, y)
        x2 = min(x+w, img.shape[1]); y2 = min(y+h, img.shape[0])
        crop = img[y:y2, x:x2]
        
        if crop.size == 0:
            crops.append(np.zeros((224,224,3), dtype=np.float32)); crop_sizes.append((0,0)); continue
        
        crop_sizes.append((crop.shape[1], crop.shape[0]))
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
        crops.append(crop)
    
    while len(crops) < 16:
        crops.append(np.zeros((224,224,3), dtype=np.float32)); crop_sizes.append((0,0))
    
    clip = np.transpose(np.stack(crops[:16]), (3,0,1,2))  # (3,16,224,224)
    return torch.tensor(clip).unsqueeze(0).float(), crop_sizes[:16]


# ── 2. INPUT VERIFICATION ──────────────────────────────────────────────
def input_verification(model, valid_tracks, SAR_9, MVIT_TO_SAR, OKU_TO_SAR):
    print("\n" + "="*70)
    print("2. INPUT VERIFICATION (5 sample tracks)")
    print("="*70, flush=True)
    
    SAR_CLASSES = ["lying_down","stationary","walking","running"]
    
    bins_wanted = {"<50px":None,"50-75px":None,"75-100px":None,">100px":None,"any":None}
    for tk, tv in valid_tracks.items():
        sz = tv["mean_size"]
        if sz < 50 and bins_wanted["<50px"] is None: bins_wanted["<50px"] = tk
        elif 50 <= sz < 75 and bins_wanted["50-75px"] is None: bins_wanted["50-75px"] = tk
        elif 75 <= sz < 100 and bins_wanted["75-100px"] is None: bins_wanted["75-100px"] = tk
        elif sz >= 100 and bins_wanted[">100px"] is None: bins_wanted[">100px"] = tk
        elif bins_wanted["any"] is None: bins_wanted["any"] = tk
    
    samples = [(k,v) for k,v in bins_wanted.items() if v is not None][:5]
    
    for bin_label, tk in samples:
        tv = valid_tracks[tk]
        gt_sar = OKU_TO_SAR[tv["action"]]
        print(f"\n--- Track: {tk} (bin: {bin_label}, mean: {tv['mean_size']:.1f}px) ---")
        print(f"  a) GT: {tv['action']} → SAR: {gt_sar}")
        
        clip, crop_sizes = extract_clip(tv)
        valid_sz = [(w,h) for w,h in crop_sizes if w > 0]
        if valid_sz:
            print(f"  b) Mean crop BEFORE resize: {np.mean([s[0] for s in valid_sz]):.0f}×{np.mean([s[1] for s in valid_sz]):.0f}px")
            print(f"     First 4 raw sizes: {valid_sz[:4]}")
        print(f"  c) Resize: cv2.INTER_LINEAR → 224×224. Tensor: {clip.shape}")
        
        with torch.no_grad():
            logits = model(clip)
            probs = torch.softmax(logits, dim=1)[0]
        
        print(f"  d) Raw output probabilities:")
        for i in range(9):
            marker = " ◄── PREDICTED" if i == probs.argmax().item() else ""
            print(f"     [{i}] {SAR_9[i]:15s}: {probs[i].item():.4f} ({probs[i].item()*100:.1f}%){marker}")
        
        pred_idx = probs.argmax().item()
        pred_sar = MVIT_TO_SAR[SAR_9[pred_idx]]
        correct = "✓ CORRECT" if pred_sar == gt_sar else "✗ WRONG"
        print(f"  e) Predicted: {SAR_9[pred_idx]} → SAR: {pred_sar} {correct}")


# ── 3. SANITY CHECK ON LARGE PERSONS ───────────────────────────────────
def sanity_check_large(model, valid_tracks, SAR_9, MVIT_TO_SAR, OKU_TO_SAR):
    print("\n" + "="*70)
    print("3. SANITY CHECK: LARGE PERSONS")
    print("="*70, flush=True)
    
    SAR_CLASSES = ["lying_down","stationary","walking","running"]
    
    for threshold in [100, 75, 50, 30]:
        large = {tk:tv for tk,tv in valid_tracks.items() if tv["mean_size"] >= threshold}
        if len(large) >= 5:
            print(f"\nTesting tracks with mean size >= {threshold}px (n={len(large)})")
            break
    
    if not large:
        print("No tracks above any threshold!"); return
    
    import random; random.seed(42)
    sample = dict(random.sample(list(large.items()), min(100, len(large))))
    
    all_preds=[]; all_true=[]
    for tk, tv in sample.items():
        gt_sar = OKU_TO_SAR[tv["action"]]
        clip, _ = extract_clip(tv)
        with torch.no_grad():
            pred_idx = model(clip).argmax(1).item()
        pred_sar = MVIT_TO_SAR[SAR_9[pred_idx]]
        all_preds.append(SAR_CLASSES.index(pred_sar))
        all_true.append(SAR_CLASSES.index(gt_sar))
    
    all_preds=np.array(all_preds); all_true=np.array(all_true)
    acc = (all_preds==all_true).mean()
    print(f"Accuracy on large persons: {acc*100:.1f}% (n={len(all_preds)})")
    
    from sklearn.metrics import classification_report
    print(classification_report(all_true, all_preds, target_names=SAR_CLASSES, zero_division=0))


# ── 4. DISTRIBUTION CHECK ──────────────────────────────────────────────
def distribution_check(model, valid_tracks, SAR_9, MVIT_TO_SAR, OKU_TO_SAR):
    print("\n" + "="*70)
    print("4. PREDICTION DISTRIBUTION CHECK")
    print("="*70, flush=True)
    
    SAR_CLASSES = ["lying_down","stationary","walking","running"]
    import random; random.seed(42)
    keys = list(valid_tracks.keys())
    if len(keys) > 200: keys = random.sample(keys, 200)
    
    raw_dist = Counter(); sar_pred_dist = Counter(); sar_gt_dist = Counter()
    max_probs = []
    
    for tk in keys:
        tv = valid_tracks[tk]
        sar_gt_dist[OKU_TO_SAR[tv["action"]]] += 1
        clip, _ = extract_clip(tv)
        with torch.no_grad():
            probs = torch.softmax(model(clip), dim=1)[0]
        idx = probs.argmax().item()
        max_probs.append(probs[idx].item())
        raw_dist[SAR_9[idx]] += 1
        sar_pred_dist[MVIT_TO_SAR[SAR_9[idx]]] += 1
    
    n = len(keys)
    print(f"\n9-class PREDICTION distribution:")
    for cls, cnt in raw_dist.most_common():
        bar = "█" * int(cnt/n*50)
        print(f"  {cls:15s}: {cnt:4d} ({cnt/n*100:5.1f}%) {bar}")
    
    print(f"\n4-class comparison (predicted vs ground truth):")
    print(f"  {'Class':12s} {'Predicted':>10s} {'GT':>10s} {'Match?':>8s}")
    for cls in SAR_CLASSES:
        p = sar_pred_dist.get(cls,0); g = sar_gt_dist.get(cls,0)
        match = "~" if abs(p-g)/max(g,1) < 0.3 else "!SKEW"
        print(f"  {cls:12s} {p:8d}   {g:8d}   {match:>8s}")
    
    print(f"\nPrediction confidence:")
    print(f"  Mean={np.mean(max_probs):.4f}  Std={np.std(max_probs):.4f}  Min={np.min(max_probs):.4f}  Max={np.max(max_probs):.4f}")
    
    top = raw_dist.most_common(1)[0]
    if top[1]/n > 0.8:
        print(f"\n⚠️  DEGENERATE: Predicts '{top[0]}' for {top[1]/n*100:.0f}% of inputs!")
    elif top[1]/n > 0.5:
        print(f"\n⚠️  BIASED: Predicts '{top[0]}' for {top[1]/n*100:.0f}% of inputs")
    else:
        print(f"\n✓ Predictions distributed across classes")


def main():
    SAR_9, MVIT_TO_SAR, OKU_TO_SAR, checkpoint = check_class_mapping()
    valid_tracks = build_test_tracks(OKU_TO_SAR)
    
    print("\nLoading MViTv2-S model...", flush=True)
    model = load_model(checkpoint)
    print("Model loaded (strict=True, all weights matched).", flush=True)
    
    input_verification(model, valid_tracks, SAR_9, MVIT_TO_SAR, OKU_TO_SAR)
    sanity_check_large(model, valid_tracks, SAR_9, MVIT_TO_SAR, OKU_TO_SAR)
    distribution_check(model, valid_tracks, SAR_9, MVIT_TO_SAR, OKU_TO_SAR)
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE.")
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
