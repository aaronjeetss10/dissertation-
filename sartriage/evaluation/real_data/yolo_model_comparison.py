"""
YOLO Model Comparison on Okutama: HERIDAL-finetuned vs COCO-pretrained
"""
import os, sys, json, math, time, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

OUT_DIR = "evaluation/real_data/full"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_BASE, "Labels", "MultiActionLabels", "3840x2160")
SCALE_X = 1280.0/3840.0; SCALE_Y = 720.0/2160.0
SEQUENCES = ["1.1.8","1.2.3","2.2.3"]

def log(msg):
    with open('/tmp/yolo_compare_progress.txt','a') as f: f.write(msg+'\n')

def iou(b1, b2):
    """IoU between [x,y,w,h] boxes."""
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[0]+b1[2],b2[0]+b2[2]), min(b1[1]+b1[3],b2[1]+b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/(union+1e-8)

def load_gt_for_frames(seq, frame_ids):
    """Load GT person boxes for specific frames."""
    gt_per_frame = defaultdict(list)
    with open(os.path.join(LABELS_DIR, f"{seq}.txt")) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 11: continue
            fi = int(p[5])
            if fi not in frame_ids: continue
            xmin = float(p[1])*SCALE_X; ymin = float(p[2])*SCALE_Y
            xmax = float(p[3])*SCALE_X; ymax = float(p[4])*SCALE_Y
            w = xmax-xmin; h = ymax-ymin
            size = math.sqrt(max(w,1)*max(h,1))
            gt_per_frame[fi].append({"bbox":[xmin,ymin,w,h], "size":size})
    return gt_per_frame

def evaluate_model(model, model_name, frame_paths, gt_per_frame, conf_thresh=0.1):
    """Run model on frames and compute detection metrics."""
    all_tp = 0; all_fp = 0; all_fn = 0
    size_bins = {"<50":{"tp":0,"fn":0,"total":0}, "50-75":{"tp":0,"fn":0,"total":0},
                 "75-100":{"tp":0,"fn":0,"total":0}, ">100":{"tp":0,"fn":0,"total":0}}
    times = []
    
    for fi, img_path in frame_paths.items():
        t0 = time.time()
        results = model(img_path, imgsz=1280, conf=conf_thresh, verbose=False)
        dt = time.time()-t0; times.append(dt)
        
        # Extract person detections
        dets = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    dets.append([x1,y1,x2-x1,y2-y1,float(box.conf[0])])
        
        gt_boxes = gt_per_frame.get(fi, [])
        matched_gt = set(); matched_det = set()
        
        # Match dets to GT (greedy, highest IoU first)
        pairs = []
        for di, det in enumerate(dets):
            for gi, gt in enumerate(gt_boxes):
                v = iou(det[:4], gt["bbox"])
                if v >= 0.5: pairs.append((v, di, gi))
        pairs.sort(reverse=True)
        for v, di, gi in pairs:
            if di in matched_det or gi in matched_gt: continue
            matched_det.add(di); matched_gt.add(gi)
        
        tp = len(matched_gt); fp = len(dets)-tp; fn = len(gt_boxes)-tp
        all_tp += tp; all_fp += fp; all_fn += fn
        
        # Size breakdown
        for gi, gt in enumerate(gt_boxes):
            s = gt["size"]
            if s < 50: b = "<50"
            elif s < 75: b = "50-75"
            elif s < 100: b = "75-100"
            else: b = ">100"
            size_bins[b]["total"] += 1
            if gi in matched_gt: size_bins[b]["tp"] += 1
            else: size_bins[b]["fn"] += 1
    
    prec = all_tp/(all_tp+all_fp+1e-8)
    rec = all_tp/(all_tp+all_fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    
    size_recall = {}
    for b, v in size_bins.items():
        size_recall[b] = {"recall": v["tp"]/(v["total"]+1e-8), "total": v["total"], "tp": v["tp"]}
    
    return {
        "model": model_name, "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "n_frames": len(frame_paths), "ms_per_frame": float(np.mean(times)*1000),
        "size_recall": size_recall,
    }

def main():
    with open('/tmp/yolo_compare_progress.txt','w') as f: f.write('YOLO Model Comparison\n')
    from ultralytics import YOLO
    
    # Collect 300 test frames across sequences
    all_frames = {}; all_gt = {}
    for seq in SEQUENCES:
        frame_dir = None
        for m in glob.glob(os.path.join(TEST_BASE,"**",seq), recursive=True):
            if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
        if not frame_dir: continue
        
        # Get all frame IDs from GT
        gt_frames = set()
        with open(os.path.join(LABELS_DIR, f"{seq}.txt")) as f:
            for line in f:
                p = line.strip().split()
                if len(p)>=11: gt_frames.add(int(p[5]))
        
        sampled = sorted(gt_frames)[::6][:100]  # ~100 per sequence
        gt_per_frame = load_gt_for_frames(seq, set(sampled))
        
        for fi in sampled:
            img_path = os.path.join(frame_dir, f"{fi}.jpg")
            if os.path.exists(img_path):
                all_frames[f"{seq}_{fi}"] = img_path
                all_gt[f"{seq}_{fi}"] = gt_per_frame.get(fi, [])
    
    # Reorganise for evaluation
    frame_paths = {k: v for k, v in all_frames.items()}
    gt_map = {k: v for k, v in all_gt.items()}
    log(f'Total frames: {len(frame_paths)}')
    total_gt = sum(len(v) for v in gt_map.values())
    log(f'Total GT boxes: {total_gt}')
    
    # Size distribution
    size_dist = {"<50":0, "50-75":0, "75-100":0, ">100":0}
    for boxes in gt_map.values():
        for b in boxes:
            s = b["size"]
            if s<50: size_dist["<50"]+=1
            elif s<75: size_dist["50-75"]+=1
            elif s<100: size_dist["75-100"]+=1
            else: size_dist[">100"]+=1
    log(f'GT size distribution: {size_dist}')
    
    # ── Model 1: HERIDAL fine-tuned ──
    log('\n=== Model 1: HERIDAL fine-tuned YOLO11n ===')
    heridal = YOLO("evaluation/results/heridal_finetune/weights/best.pt")
    r1 = evaluate_model(heridal, "HERIDAL-finetuned", frame_paths, gt_map, conf_thresh=0.1)
    log(f'  P={r1["precision"]:.3f} R={r1["recall"]:.3f} F1={r1["f1"]:.3f} ({r1["ms_per_frame"]:.0f}ms/frame)')
    for b, v in r1["size_recall"].items():
        log(f'  {b}: recall={v["recall"]:.3f} ({v["tp"]}/{v["total"]})')
    del heridal
    
    # ── Model 2: COCO YOLO11n ──
    log('\n=== Model 2: COCO-pretrained YOLO11n ===')
    coco11 = YOLO("yolo11n.pt")
    r2 = evaluate_model(coco11, "COCO-YOLO11n", frame_paths, gt_map, conf_thresh=0.1)
    log(f'  P={r2["precision"]:.3f} R={r2["recall"]:.3f} F1={r2["f1"]:.3f} ({r2["ms_per_frame"]:.0f}ms/frame)')
    for b, v in r2["size_recall"].items():
        log(f'  {b}: recall={v["recall"]:.3f} ({v["tp"]}/{v["total"]})')
    del coco11
    
    # ── Model 3: COCO YOLOv8n ──
    log('\n=== Model 3: COCO-pretrained YOLOv8n ===')
    coco8 = YOLO("yolov8n.pt")
    r3 = evaluate_model(coco8, "COCO-YOLOv8n", frame_paths, gt_map, conf_thresh=0.1)
    log(f'  P={r3["precision"]:.3f} R={r3["recall"]:.3f} F1={r3["f1"]:.3f} ({r3["ms_per_frame"]:.0f}ms/frame)')
    for b, v in r3["size_recall"].items():
        log(f'  {b}: recall={v["recall"]:.3f} ({v["tp"]}/{v["total"]})')
    del coco8
    
    # ── Also try higher conf thresholds for COCO models ──
    log('\n=== COCO YOLO11n at conf=0.25 ===')
    coco11_h = YOLO("yolo11n.pt")
    r4 = evaluate_model(coco11_h, "COCO-YOLO11n-c25", frame_paths, gt_map, conf_thresh=0.25)
    log(f'  P={r4["precision"]:.3f} R={r4["recall"]:.3f} F1={r4["f1"]:.3f}')
    for b, v in r4["size_recall"].items():
        log(f'  {b}: recall={v["recall"]:.3f} ({v["tp"]}/{v["total"]})')
    del coco11_h
    
    results = {"HERIDAL": r1, "COCO-YOLO11n": r2, "COCO-YOLOv8n": r3, "COCO-YOLO11n-c25": r4,
               "n_frames": len(frame_paths), "total_gt": total_gt, "gt_size_dist": size_dist}
    
    with open(os.path.join(OUT_DIR, "yolo_model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # ── Figure ──
    models = [r1, r2, r3, r4]
    names = [r["model"] for r in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6']
    
    # P/R/F1 bars
    ax = axes[0]
    x = np.arange(len(names)); w = 0.25
    ax.bar(x-w, [r["precision"] for r in models], w, label='Precision', color='#3498db', edgecolor='black')
    ax.bar(x, [r["recall"] for r in models], w, label='Recall', color='#2ecc71', edgecolor='black')
    ax.bar(x+w, [r["f1"] for r in models], w, label='F1', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([n.replace('-','\n') for n in names], fontsize=9)
    ax.set_ylabel('Score'); ax.set_title('Detection Metrics', fontsize=12, fontweight='bold')
    ax.legend(); ax.set_ylim(0,1)
    for i, r in enumerate(models):
        ax.text(i, r["f1"]+0.02, f'{r["f1"]:.2f}', ha='center', fontsize=8)
    
    # Recall by size
    ax = axes[1]
    size_bins = ["<50","50-75","75-100",">100"]
    for mi, r in enumerate(models):
        recs = [r["size_recall"][b]["recall"] for b in size_bins]
        ax.plot(size_bins, recs, 'o-', color=colors[mi], label=r["model"], linewidth=2, markersize=8)
    ax.set_xlabel('Person Size (px)'); ax.set_ylabel('Recall')
    ax.set_title('Recall by Person Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8); ax.set_ylim(0,1); ax.grid(True, alpha=0.3)
    
    # GT count by size
    ax = axes[2]
    counts = [size_dist[b] for b in size_bins]
    ax.bar(size_bins, counts, color='#95a5a6', edgecolor='black')
    ax.set_xlabel('Person Size (px)'); ax.set_ylabel('Count')
    ax.set_title('GT Person Size Distribution', fontsize=12, fontweight='bold')
    for i, c in enumerate(counts): ax.text(i, c+5, str(c), ha='center', fontsize=10)
    
    plt.suptitle('YOLO Model Comparison on Okutama-Action', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "yolo_model_comparison.png"), dpi=200); plt.close()
    
    # Summary
    log('\n' + '='*70)
    log('YOLO MODEL COMPARISON SUMMARY')
    log('='*70)
    log(f'\n  {"Model":<22} {"Prec":>6} {"Recall":>7} {"F1":>6} {"ms/fr":>6} {"<50 R":>6} {"50-75 R":>8} {"75-100 R":>9} {">100 R":>7}')
    log(f'  {"-"*80}')
    for r in models:
        sr = r["size_recall"]
        log(f'  {r["model"]:<22} {r["precision"]:>6.3f} {r["recall"]:>7.3f} {r["f1"]:>6.3f} {r["ms_per_frame"]:>6.0f} '
            f'{sr["<50"]["recall"]:>6.3f} {sr["50-75"]["recall"]:>8.3f} {sr["75-100"]["recall"]:>9.3f} {sr[">100"]["recall"]:>7.3f}')
    
    best = max(models, key=lambda r: r["f1"])
    log(f'\n  Best model by F1: {best["model"]} (F1={best["f1"]:.3f})')
    log(f'  Best for small (<50px): {max(models, key=lambda r: r["size_recall"]["<50"]["recall"])["model"]}')
    log('\nDone.')

if __name__ == "__main__":
    main()
