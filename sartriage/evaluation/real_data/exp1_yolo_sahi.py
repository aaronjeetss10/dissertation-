"""
EXPERIMENT 1 (FAST): YOLO + SAHI Person Detection on Okutama Frames
- Uses a SINGLE model instance for SAHI tiles (no reload per tile)
- Limits to 30 frames per sequence max
"""
import os, sys, glob, time, json, math, random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

FULL_DIR = os.path.join(os.path.dirname(__file__), "full")
EXAMPLES_DIR = os.path.join(FULL_DIR, "yolo_examples")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

TEST_FRAMES_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_FRAMES_BASE, "Labels", "MultiActionLabels", "3840x2160")
WEIGHTS = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/results/heridal_finetune/weights/best.pt"

SCALE_X = 1280.0 / 3840.0
SCALE_Y = 720.0 / 2160.0

def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0

def size_bin(w, h):
    s = math.sqrt(w*h)
    if s < 50: return "<50px"
    elif s < 75: return "50-75px"
    elif s < 100: return "75-100px"
    else: return ">100px"

def sahi_inference(model, img_path, slice_size=640, overlap=0.2):
    """Tiled inference using a pre-loaded model."""
    img = cv2.imread(img_path)
    if img is None: return []
    h, w = img.shape[:2]
    step = int(slice_size * (1 - overlap))
    all_boxes = []
    
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            tile = img[y0:min(y0+slice_size,h), x0:min(x0+slice_size,w)]
            if tile.shape[0] < 32 or tile.shape[1] < 32: continue
            res = model(tile, verbose=False, conf=0.25)[0]
            for b in res.boxes:
                if int(b.cls[0]) != 0: continue
                xyxy = b.xyxy[0].tolist()
                all_boxes.append([xyxy[0]+x0, xyxy[1]+y0, xyxy[2]-xyxy[0], xyxy[3]-xyxy[1], float(b.conf[0])])
    
    # Cross-tile NMS
    if not all_boxes: return []
    keep = []
    order = sorted(range(len(all_boxes)), key=lambda i: -all_boxes[i][4])
    used = set()
    for i in order:
        if i in used: continue
        keep.append(all_boxes[i][:4])
        used.add(i)
        for j in order:
            if j in used: continue
            if bbox_iou(all_boxes[i][:4], all_boxes[j][:4]) > 0.5:
                used.add(j)
    return keep

def match_and_score(preds, gts, iou_thresh=0.5):
    bins = {"<50px":[0,0],"50-75px":[0,0],"75-100px":[0,0],">100px":[0,0]}
    tp = 0; fp = 0; gt_matched = [False]*len(gts)
    for p in preds:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gts):
            if gt_matched[j]: continue
            iou = bbox_iou(p, g)
            if iou > best_iou: best_iou = iou; best_j = j
        if best_iou >= iou_thresh: tp += 1; gt_matched[best_j] = True
        else: fp += 1
    for j, g in enumerate(gts):
        b = size_bin(g[2], g[3]); bins[b][1] += 1
        if gt_matched[j]: bins[b][0] += 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn, bins

def main():
    print("="*60)
    print("EXPERIMENT 1: YOLO + SAHI on Okutama Test Frames")
    print("="*60)
    
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    print(f"Found {len(label_files)} label files")
    
    sequences = []
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
                xmin = float(parts[1])*SCALE_X; ymin = float(parts[2])*SCALE_Y
                xmax = float(parts[3])*SCALE_X; ymax = float(parts[4])*SCALE_Y
                fi = int(parts[5]); w = xmax-xmin; h = ymax-ymin
                if w > 0 and h > 0: frame_dict[fi].append([xmin,ymin,w,h])
        if frame_dict: sequences.append({"name":name,"dir":frame_dir,"frames":frame_dict})
    
    random.seed(42)
    if len(sequences) > 10: sequences = random.sample(sequences, 10)
    print(f"Using {len(sequences)} sequences: {[s['name'] for s in sequences]}")
    
    model = YOLO(WEIGHTS)
    print("Model loaded.")
    
    agg = {"std":{"tp":0,"fp":0,"fn":0,"bins":{"<50px":[0,0],"50-75px":[0,0],"75-100px":[0,0],">100px":[0,0]},"times":[]},
           "sahi":{"tp":0,"fp":0,"fn":0,"bins":{"<50px":[0,0],"50-75px":[0,0],"75-100px":[0,0],">100px":[0,0]},"times":[]}}
    total_frames = 0; examples_saved = 0
    
    for seq in sequences:
        imgs = sorted(glob.glob(os.path.join(seq["dir"], "*.jpg")))
        if not imgs: print(f"  {seq['name']}: no JPGs"); continue
        sampled = imgs[::6][:30]  # max 30 per sequence
        print(f"  {seq['name']}: {len(sampled)} frames", flush=True)
        
        for img_path in sampled:
            try: f_idx = int(os.path.basename(img_path).replace(".jpg",""))
            except: continue
            gt_boxes = seq["frames"].get(f_idx, [])
            if not gt_boxes: continue
            total_frames += 1
            
            # Standard
            t0 = time.time()
            res = model(img_path, verbose=False, conf=0.25, imgsz=1280)[0]
            t1 = time.time()
            agg["std"]["times"].append(t1-t0)
            preds_std = []
            for b in res.boxes:
                if int(b.cls[0]) != 0: continue
                xyxy = b.xyxy[0].tolist()
                preds_std.append([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]])
            tp,fp,fn,br = match_and_score(preds_std, gt_boxes)
            agg["std"]["tp"]+=tp; agg["std"]["fp"]+=fp; agg["std"]["fn"]+=fn
            for k in br: agg["std"]["bins"][k][0]+=br[k][0]; agg["std"]["bins"][k][1]+=br[k][1]
            
            # SAHI
            t2 = time.time()
            preds_sahi = sahi_inference(model, img_path, 640, 0.2)
            t3 = time.time()
            agg["sahi"]["times"].append(t3-t2)
            tp2,fp2,fn2,br2 = match_and_score(preds_sahi, gt_boxes)
            agg["sahi"]["tp"]+=tp2; agg["sahi"]["fp"]+=fp2; agg["sahi"]["fn"]+=fn2
            for k in br2: agg["sahi"]["bins"][k][0]+=br2[k][0]; agg["sahi"]["bins"][k][1]+=br2[k][1]
            
            # Example frames
            if examples_saved < 10 and len(gt_boxes) > 2:
                img_cv = cv2.imread(img_path)
                for g in gt_boxes: cv2.rectangle(img_cv,(int(g[0]),int(g[1])),(int(g[0]+g[2]),int(g[1]+g[3])),(0,255,0),2)
                for p in preds_sahi: cv2.rectangle(img_cv,(int(p[0]),int(p[1])),(int(p[0]+p[2]),int(p[1]+p[3])),(0,0,255),2)
                cv2.imwrite(os.path.join(EXAMPLES_DIR, f"example_{examples_saved}.jpg"), img_cv)
                examples_saved += 1
    
    # Report
    results = {}
    for cond in ["std","sahi"]:
        a = agg[cond]; tp=a["tp"]; fp=a["fp"]; fn=a["fn"]
        p=tp/(tp+fp) if (tp+fp)>0 else 0; r=tp/(tp+fn) if (tp+fn)>0 else 0
        f1=2*p*r/(p+r) if (p+r)>0 else 0
        br = {k:(v[0]/v[1] if v[1]>0 else 0) for k,v in a["bins"].items()}
        label = "Standard YOLO" if cond=="std" else "SAHI YOLO"
        print(f"\n--- {label} ---")
        print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} Time={np.mean(a['times'])*1000:.0f}ms/frame")
        for k in ["<50px","50-75px","75-100px",">100px"]:
            print(f"  Recall {k:10s}: {br[k]*100:5.1f}% (n={a['bins'][k][1]})")
        results[label] = {"precision":p,"recall":r,"f1":f1,"time_ms":np.mean(a['times'])*1000,
                          "bins":br,"bin_counts":{k:v[1] for k,v in a["bins"].items()}}
    results["total_frames"] = total_frames
    
    with open(os.path.join(FULL_DIR,"yolo_sahi_results.json"),"w") as f: json.dump(results,f,indent=2)
    
    # Plot
    labels = ["<50px","50-75px","75-100px",">100px"]
    s_v = [results["Standard YOLO"]["bins"][l] for l in labels]
    sh_v = [results["SAHI YOLO"]["bins"][l] for l in labels]
    x = np.arange(4); w = 0.35
    fig,ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, s_v, w, label='Standard', color='steelblue')
    ax.bar(x+w/2, sh_v, w, label='SAHI', color='darkorange')
    ax.set_ylabel('Recall'); ax.set_title('Person Detection Recall by Scale'); ax.set_xticks(x)
    ax.set_xticklabels(labels); ax.legend(); plt.ylim(0,1.15); plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR,"yolo_sahi_recall_by_size.png"),dpi=150)
    plt.close()
    print(f"\nEXPERIMENT 1 COMPLETE. Saved to {FULL_DIR}/")

if __name__ == "__main__":
    main()
