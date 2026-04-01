"""
Pre-compute MViTv2-S predictions for E2E pipeline.
Run standalone to avoid memory contention with YOLO/TrajMAE.
"""
import os, sys, json, math, glob, time
import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, os.path.abspath("."))

TEST_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_BASE, "Labels", "MultiActionLabels", "3840x2160")
MVIT_WEIGHTS = "models/action_mvit2_sar.pt"
OUT_DIR = "evaluation/real_data/full/end_to_end"
os.makedirs(OUT_DIR, exist_ok=True)

SCALE_X = 1280.0/3840.0; SCALE_Y = 720.0/2160.0
SEQUENCES = ["1.1.8", "1.2.3", "2.2.3"]

MVIT_9 = {0:"falling",1:"crawling",2:"lying_down",3:"running",4:"waving_hand",
           5:"climbing",6:"stumbling",7:"pushing",8:"pulling"}
MVIT_TO_SAR = {"falling":"lying_down","crawling":"lying_down","lying_down":"lying_down",
               "running":"running","waving_hand":"stationary","climbing":"walking",
               "stumbling":"walking","pushing":"stationary","pulling":"stationary"}

def log(msg):
    with open('/tmp/mvit_cache_progress.txt','a') as f: f.write(msg+'\n')

def main():
    with open('/tmp/mvit_cache_progress.txt','w') as f: f.write('MViTv2-S cache starting\n')
    
    # Load model FIRST (only model in memory)
    log('Loading MViTv2-S...')
    t0 = time.time()
    from torchvision.models.video import mvit_v2_s
    ckpt = torch.load(MVIT_WEIGHTS, map_location="cpu", weights_only=False)
    model = mvit_v2_s()
    model.head[1] = nn.Linear(model.head[1].in_features, 9)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    idx_to_label = ckpt.get('idx_to_label', MVIT_9)
    t_load = time.time() - t0
    log(f'MViTv2-S loaded in {t_load:.1f}s')
    
    cache = {}
    
    for seq in SEQUENCES:
        log(f'\nSequence: {seq}')
        
        # Find frames
        frame_dir = None
        for m in glob.glob(os.path.join(TEST_BASE,"**",seq), recursive=True):
            if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
        if not frame_dir:
            log(f'  No frames found!'); continue
        
        # Parse GT annotations
        label_file = os.path.join(LABELS_DIR, f"{seq}.txt")
        from collections import defaultdict
        gt_tracks = defaultdict(lambda: {"frames":[],"bboxes":[],"action":None})
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 11: continue
                tid = int(parts[0])
                xmin=float(parts[1])*SCALE_X; ymin=float(parts[2])*SCALE_Y
                xmax=float(parts[3])*SCALE_X; ymax=float(parts[4])*SCALE_Y
                fi = int(parts[5]); act = parts[10].strip('"')
                gt_tracks[tid]["frames"].append(fi)
                gt_tracks[tid]["bboxes"].append([xmin,ymin,xmax-xmin,ymax-ymin])
                gt_tracks[tid]["action"] = act
        
        all_frames = set()
        for t in gt_tracks.values(): all_frames.update(t["frames"])
        sorted_frames = sorted(all_frames)
        sampled_frames = sorted_frames[::6]
        
        seq_cache = {}
        n_run = 0; n_skip = 0
        
        for tid, tv in gt_tracks.items():
            # Check person size
            sizes = [math.sqrt(max(b[2],1)*max(b[3],1)) for b in tv["bboxes"]]
            mean_size = np.mean(sizes)
            
            if mean_size < 10:
                seq_cache[str(tid)] = {
                    "predicted_class": None,
                    "predicted_sar": None,
                    "confidence": 0.0,
                    "raw_probs": None,
                    "reason": f"below_minimum_scale ({mean_size:.0f}px)",
                    "person_size_px": round(float(mean_size), 1),
                }
                n_skip += 1
                continue
            
            # Sample 16 frames for this track
            track_frames = tv["frames"]
            track_bboxes = tv["bboxes"]
            
            # Get frames that are in our sampled set (or closest)
            avail = [(fi, bb) for fi, bb in zip(track_frames, track_bboxes) if fi in sampled_frames]
            if len(avail) < 4:
                # Fall back to any frames
                avail = list(zip(track_frames, track_bboxes))
            
            # Sample up to 16
            if len(avail) > 16:
                step = len(avail) // 16
                avail = avail[::step][:16]
            
            crops = []
            for fi, bb in avail:
                img_path = os.path.join(frame_dir, f"{fi}.jpg")
                if not os.path.exists(img_path):
                    crops.append(np.zeros((224,224,3), dtype=np.float32))
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    crops.append(np.zeros((224,224,3), dtype=np.float32))
                    continue
                x,y,w,h = int(bb[0]),int(bb[1]),int(max(bb[2],1)),int(max(bb[3],1))
                crop = img[max(0,y):max(0,y+h), max(0,x):max(0,x+w)]
                if crop.size == 0:
                    crops.append(np.zeros((224,224,3), dtype=np.float32))
                    continue
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                crop = (crop - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
                crops.append(crop)
            
            # Pad to 16
            while len(crops) < 16:
                crops.append(np.zeros((224,224,3), dtype=np.float32))
            
            # Build clip: (1, C, T, H, W)
            clip = torch.tensor(np.transpose(np.stack(crops[:16]), (3,0,1,2))).unsqueeze(0).float()
            
            try:
                with torch.no_grad():
                    logits = model(clip)
                    probs = torch.softmax(logits, 1)[0]
                pred_idx = probs.argmax().item()
                pred_raw = MVIT_9[pred_idx]
                pred_sar = MVIT_TO_SAR[pred_raw]
                conf = float(probs[pred_idx].item())
                
                seq_cache[str(tid)] = {
                    "predicted_class": pred_raw,
                    "predicted_sar": pred_sar,
                    "confidence": round(conf, 4),
                    "raw_probs": {MVIT_9[i]: round(float(probs[i].item()), 4) for i in range(9)},
                    "reason": "inference_complete",
                    "person_size_px": round(float(mean_size), 1),
                }
                n_run += 1
            except Exception as e:
                seq_cache[str(tid)] = {
                    "predicted_class": None,
                    "predicted_sar": None,
                    "confidence": 0.0,
                    "raw_probs": None,
                    "reason": f"inference_error: {e}",
                    "person_size_px": round(float(mean_size), 1),
                }
                n_skip += 1
        
        cache[seq] = seq_cache
        log(f'  {n_run} inferred, {n_skip} skipped. Total: {len(seq_cache)} tracks')
        
        # Show distribution
        pred_dist = {}
        for v in seq_cache.values():
            c = v.get("predicted_sar") or "skipped"
            pred_dist[c] = pred_dist.get(c, 0) + 1
        log(f'  Predictions: {pred_dist}')
    
    # Save cache
    cache_path = os.path.join(OUT_DIR, "mvit2s_predictions_cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    log(f'\nSaved to {cache_path}')
    log('DONE.')

if __name__ == "__main__":
    main()
