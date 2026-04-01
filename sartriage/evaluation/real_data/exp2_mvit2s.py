"""
EXPERIMENT 2: MViTv2-S Pixel Classification on Real Okutama Person Crops
Loads action classifier and runs on cropped person regions from test set.
"""
import os, sys, json, math, glob
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

TEST_FRAMES_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_FRAMES_BASE, "Labels", "MultiActionLabels", "3840x2160")

SCALE_X = 1280.0 / 3840.0
SCALE_Y = 720.0 / 2160.0

# MViTv2 9-class → SAR 4-class mapping
MVIT_CLASSES = {0:"falling",1:"crawling",2:"lying_down",3:"running",4:"waving_hand",5:"climbing",6:"stumbling",7:"pushing",8:"pulling"}
MVIT_TO_SAR = {"falling":"lying_down","crawling":"lying_down","lying_down":"lying_down",
               "running":"running","waving_hand":"stationary","climbing":"walking",
               "stumbling":"walking","pushing":"stationary","pulling":"stationary"}
SAR_CLASSES = ["lying_down","stationary","walking","running"]

# Okutama label → SAR mapping
OKU_TO_SAR = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
              "Running":"running","Lying":"lying_down","Carrying":"stationary",
              "Pushing/Pulling":"stationary","Reading":"stationary",
              "Calling":"stationary","Drinking":"stationary"}

def main():
    print("="*60)
    print("EXPERIMENT 2: MViTv2-S on Real Okutama Person Crops")
    print("="*60, flush=True)
    
    # Try loading model
    model_path = "models/action_mvit2_sar.pt"
    print(f"Loading MViTv2-S from {model_path}...", flush=True)
    
    try:
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        model = mvit_v2_s()
        model.head[1] = nn.Linear(model.head[1].in_features, 9)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        print("MViTv2-S loaded successfully.", flush=True)
    except Exception as e:
        print(f"ERROR loading MViTv2-S: {e}", flush=True)
        # Fall back to R3D-18
        print("Falling back to R3D-18...", flush=True)
        try:
            import torchvision.models.video as video_models
            backbone = video_models.r3d_18(weights=None)
            backbone.fc = nn.Identity()
            model = nn.Sequential(
                backbone,
                nn.Dropout(0.3),
                nn.Linear(512, 9)
            )
            r3d_path = "models/action_r3d18_sar.pt"
            state = torch.load(r3d_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()
            print(f"R3D-18 loaded from {r3d_path}", flush=True)
        except Exception as e2:
            print(f"ERROR loading R3D-18: {e2}", flush=True)
            return
    
    print(f"\nClass mapping (MViTv2/R3D → SAR):")
    for k,v in MVIT_TO_SAR.items(): print(f"  {k} → {v}")
    
    # Parse test set annotations with per-track info
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    
    # Build per-frame, per-track GT
    all_tracks = defaultdict(lambda: {"frames":[], "bboxes":[], "action":None, "seq":None})
    
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
                frame_idx = int(parts[5])
                action = parts[10].strip('"')
                
                track_key = f"{seq_name}_{tid}"
                all_tracks[track_key]["frames"].append(frame_idx)
                all_tracks[track_key]["bboxes"].append([xmin, ymin, xmax-xmin, ymax-ymin])
                all_tracks[track_key]["action"] = action
                all_tracks[track_key]["seq"] = seq_name
                all_tracks[track_key]["dir"] = frame_dir
    
    # Filter to tracks with >= 16 frames and known SAR action
    valid_tracks = {}
    for tk, tv in all_tracks.items():
        if len(tv["frames"]) < 16: continue
        if tv["action"] not in OKU_TO_SAR: continue
        valid_tracks[tk] = tv
    
    print(f"\nTotal test tracks with >= 16 frames: {len(valid_tracks)}", flush=True)
    
    # Sample up to 200 tracks for inference
    import random
    random.seed(42)
    track_keys = list(valid_tracks.keys())
    if len(track_keys) > 200:
        track_keys = random.sample(track_keys, 200)
    
    print(f"Running inference on {len(track_keys)} tracks...", flush=True)
    
    all_preds = []; all_true = []; all_sizes = []; failures = 0
    
    for tk in track_keys:
        tv = valid_tracks[tk]
        gt_sar = OKU_TO_SAR[tv["action"]]
        
        # Pick 16 consecutive frames
        frames = tv["frames"]; bboxes = tv["bboxes"]
        start = 0
        if len(frames) > 16:
            start = len(frames) // 2 - 8  # center 16
        
        crops = []
        sizes = []
        for i in range(start, min(start+16, len(frames))):
            fi = frames[i]; bb = bboxes[i]
            img_path = os.path.join(tv["dir"], f"{fi}.jpg")
            if not os.path.exists(img_path): continue
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            x = max(0, x); y = max(0, y)
            crop = img[y:y+max(h,1), x:x+max(w,1)]
            if crop.size == 0: continue
            
            crop = cv2.resize(crop, (224, 224))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            crops.append(crop)
            sizes.append(math.sqrt(w*h))
        
        if len(crops) < 16:
            # Pad with zeros
            while len(crops) < 16:
                crops.append(np.zeros((224, 224, 3), dtype=np.float32))
                sizes.append(0)
        
        # Shape: (3, 16, 224, 224) for video models
        clip = np.stack(crops[:16])  # (16, 224, 224, 3)
        clip = np.transpose(clip, (3, 0, 1, 2))  # (3, 16, 224, 224)
        clip_t = torch.tensor(clip).unsqueeze(0).float()  # (1, 3, 16, 224, 224)
        
        try:
            with torch.no_grad():
                logits = model(clip_t)
            pred_idx = logits.argmax(1).item()
            pred_action = MVIT_CLASSES[pred_idx]
            pred_sar = MVIT_TO_SAR[pred_action]
            
            mean_size = np.mean([s for s in sizes if s > 0]) if any(s > 0 for s in sizes) else 0
            
            all_preds.append(SAR_CLASSES.index(pred_sar))
            all_true.append(SAR_CLASSES.index(gt_sar))
            all_sizes.append(mean_size)
        except Exception as e:
            failures += 1
    
    print(f"Inference complete: {len(all_preds)} tracks, {failures} failures", flush=True)
    
    if not all_preds:
        print("ERROR: No predictions generated. Cannot compute metrics.")
        return
    
    all_preds = np.array(all_preds); all_true = np.array(all_true)
    all_sizes = np.array(all_sizes)
    acc = (all_preds == all_true).mean()
    
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
    kappa = cohen_kappa_score(all_true, all_preds)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Failures: {failures}")
    
    report_str = classification_report(all_true, all_preds, target_names=SAR_CLASSES, zero_division=0)
    print("\nClassification Report:"); print(report_str)
    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:"); print(cm)
    
    # By size
    print("\nAccuracy by person size:", flush=True)
    bins_info = {}
    for name, lo, hi in [("<50px",0,50),("50-75px",50,75),("75-100px",75,100),(">100px",100,9999)]:
        idx = [i for i in range(len(all_sizes)) if lo<=all_sizes[i]<hi]
        if not idx: continue
        b_acc = (all_preds[idx]==all_true[idx]).mean()
        bins_info[name] = {"n":len(idx),"accuracy":float(b_acc)}
        print(f"  {name:10s}: N={len(idx):3d} Acc={b_acc*100:.1f}%")
    
    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(SAR_CLASSES))); ax.set_yticks(range(len(SAR_CLASSES)))
    ax.set_xticklabels(SAR_CLASSES, rotation=45); ax.set_yticklabels(SAR_CLASSES)
    for i in range(len(SAR_CLASSES)):
        for j in range(len(SAR_CLASSES)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('MViTv2-S / R3D-18 Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "mvit2s_confusion.png"), dpi=150)
    plt.close()
    
    results = {
        "accuracy": float(acc), "kappa": float(kappa), "failures": failures,
        "report": classification_report(all_true, all_preds, target_names=SAR_CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(), "size_bins": bins_info,
        "n_tested": len(all_preds), "note": "Person crops from GT bounding boxes on Okutama TestSet"
    }
    with open(os.path.join(FULL_DIR, "mvit2s_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("EXPERIMENT 2 COMPLETE.", flush=True)

if __name__ == "__main__":
    main()
