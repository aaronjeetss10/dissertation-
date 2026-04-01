"""
EXPERIMENT 4: SAR Preprocessing Impact on Detection
Compares raw vs CLAHE+gamma+sharpen preprocessing on YOLO person recall.
"""
import os, sys, glob, time, json, math, random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO
import scipy.stats as stats

FULL_DIR = os.path.join(os.path.dirname(__file__), "full")
os.makedirs(FULL_DIR, exist_ok=True)

TEST_FRAMES_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_FRAMES_BASE, "Labels", "MultiActionLabels", "3840x2160")
WEIGHTS = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/results/heridal_finetune/weights/best.pt"

SCALE_X = 1280.0 / 3840.0
SCALE_Y = 720.0 / 2160.0

def sar_preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gamma = 0.8
    table = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def bbox_iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[0]+b1[2],b2[0]+b2[2]); y2=min(b1[1]+b1[3],b2[1]+b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/union if union>0 else 0.0

def size_bin(w,h):
    s=math.sqrt(w*h)
    if s<50: return "<50px"
    elif s<75: return "50-75px"
    elif s<100: return "75-100px"
    else: return ">100px"

def compute_recall(model, frames_data, preprocess=False):
    """Returns per-frame recall list and per-bin recall."""
    bins = {"<50px":[0,0],"50-75px":[0,0],"75-100px":[0,0],">100px":[0,0]}
    per_frame_recall = []
    
    for img_path, gt_boxes in frames_data:
        if preprocess:
            img = cv2.imread(img_path)
            img = sar_preprocess(img)
            tmp = "/tmp/_sar_tmp.jpg"
            cv2.imwrite(tmp, img)
            res = model(tmp, verbose=False, conf=0.25, imgsz=1280)[0]
        else:
            res = model(img_path, verbose=False, conf=0.25, imgsz=1280)[0]
        
        preds = []
        for b in res.boxes:
            if int(b.cls[0]) != 0: continue
            xyxy = b.xyxy[0].tolist()
            preds.append([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]])
        
        gt_matched = [False]*len(gt_boxes)
        for p in preds:
            best_iou, best_j = 0, -1
            for j, g in enumerate(gt_boxes):
                if gt_matched[j]: continue
                iou = bbox_iou(p, g)
                if iou > best_iou: best_iou=iou; best_j=j
            if best_iou >= 0.5 and best_j >= 0: gt_matched[best_j] = True
        
        r = sum(gt_matched)/len(gt_matched) if gt_matched else 0
        per_frame_recall.append(r)
        
        for j, g in enumerate(gt_boxes):
            b = size_bin(g[2], g[3]); bins[b][1] += 1
            if gt_matched[j]: bins[b][0] += 1
    
    return per_frame_recall, {k:(v[0]/v[1] if v[1]>0 else 0) for k,v in bins.items()}, bins

def main():
    print("="*60)
    print("EXPERIMENT 4: SAR Preprocessing Impact on Detection")
    print("="*60)
    
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    frames_data = []
    
    for lf in label_files:
        name = os.path.basename(lf).replace(".txt","")
        matches = glob.glob(os.path.join(TEST_FRAMES_BASE, "**", name), recursive=True)
        frame_dir = None
        for m in matches:
            if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
        if not frame_dir: continue
        
        frame_dict = defaultdict(list)
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 11: continue
                xmin=float(parts[1])*SCALE_X; ymin=float(parts[2])*SCALE_Y
                xmax=float(parts[3])*SCALE_X; ymax=float(parts[4])*SCALE_Y
                fi=int(parts[5]); w=xmax-xmin; h=ymax-ymin
                if w>0 and h>0: frame_dict[fi].append([xmin,ymin,w,h])
        
        imgs = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        for img_path in imgs[::10][:20]:  # every 10th, max 20 per seq
            try: f_idx = int(os.path.basename(img_path).replace(".jpg",""))
            except: continue
            if f_idx in frame_dict: frames_data.append((img_path, frame_dict[f_idx]))
    
    random.seed(42)
    if len(frames_data) > 200: frames_data = random.sample(frames_data, 200)
    print(f"Using {len(frames_data)} frames for preprocessing comparison.")
    
    model = YOLO(WEIGHTS)
    print("Model loaded. Running raw inference...")
    raw_recall, raw_bins, raw_counts = compute_recall(model, frames_data, preprocess=False)
    print(f"  Raw mean recall: {np.mean(raw_recall)*100:.1f}%")
    
    print("Running preprocessed inference...")
    pre_recall, pre_bins, pre_counts = compute_recall(model, frames_data, preprocess=True)
    print(f"  Preprocessed mean recall: {np.mean(pre_recall)*100:.1f}%")
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(raw_recall, pre_recall)
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    print("\nRecall by size bin:")
    for k in ["<50px","50-75px","75-100px",">100px"]:
        print(f"  {k:10s}: Raw={raw_bins[k]*100:5.1f}% | Pre={pre_bins[k]*100:5.1f}%")
    
    results = {
        "raw": {"mean_recall": np.mean(raw_recall), "bins": raw_bins, "bin_counts": {k:v[1] for k,v in raw_counts.items()}},
        "preprocessed": {"mean_recall": np.mean(pre_recall), "bins": pre_bins, "bin_counts": {k:v[1] for k,v in pre_counts.items()}},
        "paired_ttest": {"t_stat": t_stat, "p_value": p_val},
        "n_frames": len(frames_data)
    }
    with open(os.path.join(FULL_DIR, "preprocessing_impact.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    labels = ["<50px","50-75px","75-100px",">100px"]
    r_v = [raw_bins[l] for l in labels]; p_v = [pre_bins[l] for l in labels]
    x = np.arange(4); w = 0.35
    fig,ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, r_v, w, label='Raw', color='grey')
    ax.bar(x+w/2, p_v, w, label='CLAHE+Gamma+Sharpen', color='green', alpha=0.8)
    ax.set_ylabel('Recall'); ax.set_title(f'Preprocessing Impact (p={p_val:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
    plt.ylim(0,1.15); plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR,"preprocessing_impact.png"), dpi=150)
    plt.close()
    print("EXPERIMENT 4 COMPLETE.")

if __name__ == "__main__":
    main()
