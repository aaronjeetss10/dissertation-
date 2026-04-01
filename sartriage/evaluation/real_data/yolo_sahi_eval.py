import os
import sys
import glob
import time
import json
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from decimal import Decimal

# YOLO & SAHI
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

FULL_DIR = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/real_data/full"
FRAMES_OUT_DIR = os.path.join(FULL_DIR, "yolo_example_frames")
os.makedirs(FRAMES_OUT_DIR, exist_ok=True)

# Pointing to the downloaded dataset
BASE_PATH = "/Users/aaronsandhu/Downloads/TestSetFrames"

# Check if path exists
if not os.path.exists(BASE_PATH):
    print(f"Error: Path {BASE_PATH} not found.")
    sys.exit(1)

def bbox_iou(box1, box2):
    # box format: [x,y,w,h]
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0: return 0.0
    return inter_area / union_area

def format_bin(sz):
    if sz < 50: return "<50px"
    elif sz < 75: return "50-75px"
    elif sz < 100: return "75-100px"
    else: return ">100px"

def evaluate_predictions(preds_list, gts_list, iou_thresh=0.5):
    """
    preds_list: list of lists of predictions (per frame) -> [[x,y,w,h], ...]
    gts_list: list of lists of ground truth -> [[x,y,w,h], ...]
    Returns dict representing precision, recall, f1
    """
    tp = 0
    fp = 0
    total_gts = sum(len(gts) for gts in gts_list)
    total_preds = sum(len(preds) for preds in preds_list)
    
    if total_gts == 0: return {"P": 0, "R": 0, "F1": 0}
    
    matched_gt = 0
    
    # Calculate size thresholds recall
    bins = {"<50px": {"tp":0, "total":0}, "50-75px": {"tp":0, "total":0}, "75-100px": {"tp":0, "total":0}, ">100px": {"tp":0, "total":0}}
    
    for i in range(len(preds_list)):
        preds = preds_list[i]
        gts = gts_list[i]
        
        # for each gt, classify size bin
        gt_matched = [False] * len(gts)
        pred_matched = [False] * len(preds)
        
        for gi, gt in enumerate(gts):
            b_name = format_bin(math.sqrt(gt[2]*gt[3]))
            bins[b_name]["total"] += 1
            
            # find best matching pred
            best_iou = 0
            best_pi = -1
            for pi, p in enumerate(preds):
                if not pred_matched[pi]:
                    iou = bbox_iou(gt, p)
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi
            
            if best_iou >= iou_thresh:
                gt_matched[gi] = True
                pred_matched[best_pi] = True
                matched_gt += 1
                tp += 1
                bins[b_name]["tp"] += 1
                
        # remaining preds are fps
        fp += sum(1 for pm in pred_matched if not pm)
    
    recall = tp / total_gts if total_gts > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    bin_recalls = {}
    for k, v in bins.items():
        bin_recalls[k] = v["tp"] / v["total"] if v["total"] > 0 else 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map_05": recall * precision, # naive proxy, real mAP needs sorting config
        "bins": bin_recalls,
        "total_gts": total_gts
    }


def main():
    print("Scanning TestSetFrames directory...")
    sequences = []
    
    # Check all txt files in Labels
    txt_files = glob.glob(os.path.join(BASE_PATH, "Labels", "**", "*.txt"), recursive=True)
    
    for label_f in txt_files:
        name = os.path.basename(label_f).replace(".txt", "")
        # Find corresponding dir recursively
        expected_matches = glob.glob(os.path.join(BASE_PATH, "**", name), recursive=True)
        seq_dir = None
        for m in expected_matches:
            if os.path.isdir(m) and not m.endswith("Labels") and "Extracted-Frames-1280" in m:
                seq_dir = m
                break
                
        if seq_dir:
            # Parse label file
            frame_dict = defaultdict(list)
            try:
                with open(label_f, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 11:
                            xmin = float(parts[1]) / 3.0
                            ymin = float(parts[2]) / 3.0
                            xmax = float(parts[3]) / 3.0
                            ymax = float(parts[4]) / 3.0
                            frame = int(parts[5])
                            w = xmax - xmin
                            h = ymax - ymin
                            frame_dict[frame].append([xmin, ymin, w, h])
                if len(frame_dict) > 0:
                    sequences.append({"name": name, "dir": seq_dir, "frames": frame_dict})
            except Exception as e:
                pass
                
    random.seed(42)
    # Pick 10
    if len(sequences) > 10:
        sequences = random.sample(sequences, 10)
    elif len(sequences) == 0:
        print("Could not find any sequences matching labels. Make sure directory structure is correct.")
        sys.exit(1)
        
    print(f"Selected {len(sequences)} test sequences for inference evaluation.")
    
    # Load Models
    weights_path = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/results/heridal_finetune/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt" # Fallback if finetuned missing
        
    model_yolo = YOLO(weights_path)
    
    # Load SAHI
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=weights_path,
        confidence_threshold=0.25,
        device="cpu", # safe fallback
    )
    
    all_gts = []
    all_preds_std = []
    all_preds_sahi = []
    
    time_std = []
    time_sahi = []
    
    example_frames_saved = 0
    total_imgs = 0
    
    print("Beginning Inference...")
    for seq in sequences:
        # get all sorted frames inside dir
        imgs = sorted(glob.glob(os.path.join(seq["dir"], "*.jpg")))
        if not imgs:
            imgs = sorted(glob.glob(os.path.join(seq["dir"], "frames", "*.jpg")))
        
        # sample every 6th
        sampled_imgs = imgs[::6]
        
        for img_path in sampled_imgs:
            # parse frame num from filename (e.g. "00000.jpg")
            try:
                base = os.path.basename(img_path).replace(".jpg", "")
                f_idx = int(base)
            except:
                continue
                
            gt_boxes = seq["frames"].get(f_idx, [])
            all_gts.append(gt_boxes)
            
            # 1. Standard
            t0 = time.time()
            res_std = model_yolo(img_path, verbose=False)[0]
            boxes_std = []
            for b in res_std.boxes:
                # map to person explicitly or just fallback all
                if int(b.cls[0]) == 0 or True: # Assuming class 0 is person
                    xyxy = b.xyxy[0].tolist()
                    boxes_std.append([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]])
            t1 = time.time()
            time_std.append(t1 - t0)
            all_preds_std.append(boxes_std)
            
            # 2. SAHI inference
            t2 = time.time()
            res_sahi = get_sliced_prediction(
                img_path, sahi_model, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0
            )
            boxes_sahi = []
            for ob in res_sahi.object_prediction_list:
                bbox = ob.bbox
                boxes_sahi.append([bbox.minx, bbox.miny, bbox.maxx - bbox.minx, bbox.maxy - bbox.miny])
            t3 = time.time()
            time_sahi.append(t3 - t2)
            all_preds_sahi.append(boxes_sahi)
            
            total_imgs += 1
            
            # Save 10 images with SAHI boxes visually overlaid for proof
            if example_frames_saved < 10 and len(gt_boxes) > 0 and len(boxes_sahi) > 0:
                img_cv = cv2.imread(img_path)
                for g in gt_boxes:
                    cv2.rectangle(img_cv, (int(g[0]), int(g[1])), (int(g[0]+g[2]), int(g[1]+g[3])), (0,255,0), 2)
                for s in boxes_sahi:
                    cv2.rectangle(img_cv, (int(s[0]), int(s[1])), (int(s[0]+s[2]), int(s[1]+s[3])), (0,0,255), 2)
                
                out_p = os.path.join(FRAMES_OUT_DIR, f"example_{example_frames_saved}.jpg")
                cv2.imwrite(out_p, img_cv)
                example_frames_saved += 1
                
    print(f"\nProcessed {total_imgs} sampled frames.")
    
    # Evaluate
    met_std = evaluate_predictions(all_preds_std, all_gts)
    met_sahi = evaluate_predictions(all_preds_sahi, all_gts)
    
    print("\n--- RESULTS: Standard YOLO (1280px) ---")
    print(f"Precision: {met_std['precision']:.3f} | Recall: {met_std['recall']:.3f} | F1: {met_std['f1']:.3f}")
    print(f"Avg Time : {np.mean(time_std):.3f}s / frame")
    print(f"Recall by Size:")
    for k, v in met_std["bins"].items(): print(f"  {k:10s}: {v*100:.1f}%")
        
    print("\n--- RESULTS: SAHI YOLO (640px Tiles, 0.2 Overlap) ---")
    print(f"Precision: {met_sahi['precision']:.3f} | Recall: {met_sahi['recall']:.3f} | F1: {met_sahi['f1']:.3f}")
    print(f"Avg Time : {np.mean(time_sahi):.3f}s / frame")
    print(f"Recall by Size:")
    for k, v in met_sahi["bins"].items(): print(f"  {k:10s}: {v*100:.1f}%")

    out_j = os.path.join(FULL_DIR, "yolo_sahi_results.json")
    with open(out_j, "w") as f:
        json.dump({
            "Standard": {"precision": met_std['precision'], "recall": met_std['recall'], "f1": met_std['f1'], "time": np.mean(time_std), "bins": met_std["bins"]},
            "SAHI": {"precision": met_sahi['precision'], "recall": met_sahi['recall'], "f1": met_sahi['f1'], "time": np.mean(time_sahi), "bins": met_sahi["bins"]}
        }, f, indent=2)
        
    # Plot recalling
    labels = ["<50px", "50-75px", "75-100px", ">100px"]
    s_vals = [met_std["bins"][l] for l in labels]
    sahi_vals = [met_sahi["bins"][l] for l in labels]
    
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - w/2, s_vals, w, label='Standard YOLO', color='blue', alpha=0.7)
    ax.bar(x + w/2, sahi_vals, w, label='SAHI YOLO', color='orange', alpha=0.7)
    ax.set_ylabel('Recall')
    ax.set_title('Person Recall by Scale (YOLO vs SAHI)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for i in range(len(labels)):
        ax.text(i - w/2, s_vals[i] + 0.02, f"{s_vals[i]*100:.1f}%", ha='center', fontsize=9)
        ax.text(i + w/2, sahi_vals[i] + 0.02, f"{sahi_vals[i]*100:.1f}%", ha='center', fontsize=9)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(FULL_DIR, "yolo_sahi_recall_by_size.png"))
    plt.close()
    
    print("\nSaved SAHI pipeline artifacts natively.")

if __name__ == "__main__":
    from collections import defaultdict
    main()
