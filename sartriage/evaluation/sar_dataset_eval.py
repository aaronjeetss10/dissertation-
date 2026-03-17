"""
evaluation/sar_dataset_eval.py
================================
Evaluate SARTriage on domain-specific SAR datasets:

1. HERIDAL — SAR person detection in wilderness aerial imagery
   Tests: YOLO11 + SAHI detection on real SAR conditions
   
2. Okutama-Action — Aerial drone action recognition
   Tests: TMS trajectory classification against GT action labels
   
This validates the entire pipeline on datasets that actually represent
the target deployment scenario (drone SAR), not just generic benchmarks.
"""

from __future__ import annotations

import json, sys, time, math, re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
HERIDAL_DIR = Path(__file__).parent / "datasets" / "heridal"
OKUTAMA_DIR = Path(__file__).parent / "datasets" / "okutama"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# HERIDAL Evaluation — SAR Person Detection
# ══════════════════════════════════════════════════════════════════════════

def load_heridal_gt(split="test"):
    """Load HERIDAL ground truth in normalised YOLO format."""
    img_dir = HERIDAL_DIR / split / "images"
    lbl_dir = HERIDAL_DIR / split / "labels"

    if not img_dir.exists():
        print(f"  ⚠ HERIDAL {split} not found at {img_dir}")
        return []

    data = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        bboxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, w, h = int(parts[0]), *[float(x) for x in parts[1:5]]
                        bboxes.append({"cx": cx, "cy": cy, "w": w, "h": h})

        data.append({"image": str(img_path), "bboxes": bboxes})

    return data


def run_heridal_detection():
    """Evaluate YOLO11 + SAHI on HERIDAL SAR dataset."""
    import matplotlib.pyplot as plt

    print("  Loading HERIDAL test set...")
    test_data = load_heridal_gt("test")
    if not test_data:
        print("  ⚠ HERIDAL test data not found, trying valid...")
        test_data = load_heridal_gt("valid")
    if not test_data:
        return

    print(f"  Loaded {len(test_data)} images with {sum(len(d['bboxes']) for d in test_data)} annotated persons")

    from ultralytics import YOLO

    configs = [
        ("YOLO11n (raw)", "yolo11n.pt", False, False),
        ("YOLO11n + SAR Preproc", "yolo11n.pt", False, True),
        ("YOLO11n + SAHI", "yolo11n.pt", True, False),
        ("YOLO11n + SAHI + SAR", "yolo11n.pt", True, True),
    ]

    all_results = {}

    for config_name, model_path, use_sahi, use_preproc in configs:
        print(f"  Running {config_name}...")
        model = YOLO(model_path)

        total_tp, total_fp, total_fn = 0, 0, 0
        total_gt = 0
        person_sizes = []
        confidences = []
        inference_times = []

        for item in test_data:
            img = cv2.imread(item["image"])
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_bboxes = item["bboxes"]
            total_gt += len(gt_bboxes)

            # SAR preprocessing
            proc_img = img.copy()
            if use_preproc:
                proc_img = _sar_preprocess(proc_img)

            t0 = time.time()

            if use_sahi:
                detections = _sahi_detect(model, proc_img, slice_size=320, conf=0.15)
            else:
                preds = model(proc_img, verbose=False, conf=0.15)
                detections = []
                for r in preds:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "conf": float(box.conf[0]),
                            })

            t1 = time.time()
            inference_times.append(t1 - t0)

            # Match detections to GT
            gt_matched = [False] * len(gt_bboxes)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det_cx = (x1 + x2) / 2 / w
                det_cy = (y1 + y2) / 2 / h
                det_w = (x2 - x1) / w
                det_h = (y2 - y1) / h

                best_iou, best_idx = 0, -1
                for gi, gt in enumerate(gt_bboxes):
                    if gt_matched[gi]:
                        continue
                    iou = _bbox_iou_norm(
                        det_cx, det_cy, det_w, det_h,
                        gt["cx"], gt["cy"], gt["w"], gt["h"]
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi

                if best_iou > 0.3 and best_idx >= 0:  # Lower IoU for tiny objects
                    gt_matched[best_idx] = True
                    total_tp += 1
                    confidences.append(det["conf"])
                    person_sizes.append(max(det["bbox"][2] - det["bbox"][0],
                                           det["bbox"][3] - det["bbox"][1]))
                else:
                    total_fp += 1

            total_fn += sum(1 for m in gt_matched if not m)

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        result = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "total_gt": total_gt,
            "mean_conf": round(float(np.mean(confidences)) if confidences else 0, 4),
            "mean_person_size_px": round(float(np.mean(person_sizes)) if person_sizes else 0, 1),
            "mean_inference_ms": round(float(np.mean(inference_times)) * 1000, 1),
            "images_tested": len(test_data),
        }
        all_results[config_name] = result
        print(f"    P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
              f"(TP={total_tp}, FP={total_fp}, FN={total_fn})")

    # Save
    with open(RESULTS_DIR / "heridal_detection.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = list(all_results.keys())
    precs = [all_results[n]["precision"] for n in names]
    recs = [all_results[n]["recall"] for n in names]
    f1s = [all_results[n]["f1"] for n in names]

    # Panel 1: P/R/F1 bars
    ax = axes[0]
    x = np.arange(len(names))
    w_bar = 0.25
    ax.bar(x - w_bar, precs, w_bar, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x, recs, w_bar, label="Recall", color="#e74c3c", alpha=0.85)
    ax.bar(x + w_bar, f1s, w_bar, label="F1", color="#2ecc71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("HERIDAL SAR Detection\nPrecision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Panel 2: Improvement waterfall
    ax = axes[1]
    baseline_f1 = f1s[0]
    improvements = [f1 - baseline_f1 for f1 in f1s]
    colors = ["#95a5a6"] + ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements[1:]]
    ax.bar(range(len(names)), f1s, color=colors, alpha=0.85)
    for i, (f1, name) in enumerate(zip(f1s, names)):
        ax.text(i, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Improvement over Baseline\non Real SAR Data", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Panel 3: Stats
    ax = axes[2]
    stats_text = [
        f"Dataset: HERIDAL (SAR-specific)",
        f"Images tested: {len(test_data)}",
        f"Total GT persons: {total_gt}",
        f"Person size: wilderness/mountain",
        f"",
        f"Best config: {names[np.argmax(f1s)]}",
        f"Best F1: {max(f1s):.3f}",
        f"Best Recall: {max(recs):.3f}",
        f"",
        f"SAHI improvement: +{(f1s[2]-f1s[0])*100:.1f}pp F1",
        f"SAR preproc: +{(f1s[3]-f1s[2])*100:.1f}pp F1",
    ]
    ax.text(0.1, 0.95, "\n".join(stats_text), transform=ax.transAxes,
           fontsize=11, verticalalignment="top", family="monospace",
           bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))
    ax.set_title("HERIDAL Results Summary", fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.suptitle("SARTriage Detection on HERIDAL\n"
                 "Real SAR Aerial Person Detection in Wilderness",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heridal_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ heridal_detection.png")


def _sar_preprocess(img):
    """SAR-specific image preprocessing."""
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma correction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean() / 255.0
    if mean_brightness < 0.4:
        gamma = 0.7
    elif mean_brightness > 0.7:
        gamma = 1.3
    else:
        gamma = 1.0

    if gamma != 1.0:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    # Sharpening
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return img


def _sahi_detect(model, img, slice_size=320, overlap=0.2, conf=0.15):
    """Tiled inference using SAHI approach."""
    h, w = img.shape[:2]
    stride = int(slice_size * (1 - overlap))
    detections = []

    # Full image pass
    preds = model(img, verbose=False, conf=conf)
    for r in preds:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(box.conf[0]),
                })

    # Tiled passes
    for y in range(0, h - slice_size // 2, stride):
        for x in range(0, w - slice_size // 2, stride):
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            tile = img[y:y2, x:x2]

            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue

            preds = model(tile, verbose=False, conf=conf)
            for r in preds:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            "bbox": [float(tx1 + x), float(ty1 + y),
                                    float(tx2 + x), float(ty2 + y)],
                            "conf": float(box.conf[0]),
                        })

    # NMS
    if detections:
        detections = _nms(detections, iou_thresh=0.5)

    return detections


def _nms(detections, iou_thresh=0.5):
    """Simple NMS on detection list."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []

    for det in detections:
        is_dup = False
        for kept in keep:
            iou = _bbox_iou_abs(det["bbox"], kept["bbox"])
            if iou > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            keep.append(det)

    return keep


def _bbox_iou_abs(b1, b2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(area1 + area2 - inter, 1e-6)


def _bbox_iou_norm(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    """IoU between two normalised center-format boxes."""
    x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
    x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
    x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
    x2_max, y2_max = cx2 + w2/2, cy2 + h2/2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y

    area1 = w1 * h1
    area2 = w2 * h2
    return inter / max(area1 + area2 - inter, 1e-6)


# ══════════════════════════════════════════════════════════════════════════
# Okutama-Action Evaluation — Aerial Action Recognition + TMS
# ══════════════════════════════════════════════════════════════════════════

def parse_okutama_labels(label_path):
    """Parse Okutama-Action label file into tracks with actions."""
    tracks = defaultdict(list)

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract all quoted strings first
            quoted = re.findall(r'"([^"]+)"', line)

            # Parse numeric fields (everything before first quote)
            num_part = line[:line.index('"')].strip().split()
            if len(num_part) < 9:
                continue

            track_id = int(num_part[0])
            x1, y1, x2, y2 = int(num_part[1]), int(num_part[2]), int(num_part[3]), int(num_part[4])
            frame = int(num_part[5])
            lost = int(num_part[6])
            occluded = int(num_part[7])

            if lost == 1:
                continue

            # First quoted string is "Person", rest are actions
            actions = quoted[1:] if len(quoted) > 1 else []

            tracks[track_id].append({
                "frame": frame,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "actions": actions,
                "occluded": occluded,
            })

    return dict(tracks)


def run_okutama_tms():
    """Evaluate TMS on Okutama-Action aerial drone footage."""
    import matplotlib.pyplot as plt
    from streams.tms_classifier import TrajectoryFeatures, TMS_RULES

    label_path = OKUTAMA_DIR / "1.1.1.txt"
    video_path = OKUTAMA_DIR / "1.1.1.mov"

    if not label_path.exists():
        print("  ⚠ Okutama labels not found")
        return

    print("  Parsing Okutama-Action labels...")
    tracks = parse_okutama_labels(label_path)
    print(f"  Found {len(tracks)} tracks")

    # Get video dimensions
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    print(f"  Video: {vid_w}x{vid_h} @ {fps:.0f}fps")

    # Map Okutama actions to our SAR categories
    okutama_to_sar = {
        "Walking": "running",       # walking = slow running
        "Running": "running",
        "Lying": "lying_down",
        "Sitting": "lying_down",     # seated = low posture
        "Standing": "waving",        # standing = stationary (approximation)
        "Carrying": "running",       # carrying = walking with object
        "Pushing/Pulling": "crawling",  # low-effort movement
        "Reading": "waving",         # stationary activity
        "Drinking": "waving",        # stationary activity
        "Calling": "waving",         # stationary activity
        "Hand Shaking": "waving",    # stationary gesture
    }

    # Classify each track using TMS
    results = {"tracks": [], "confusion": defaultdict(lambda: defaultdict(int))}
    frame_dims = (vid_h, vid_w)

    action_counts = defaultdict(int)
    tms_correct = 0
    tms_total = 0

    for tid, track_data in tracks.items():
        if len(track_data) < 8:
            continue

        # Extract trajectory data
        centroids = []
        timestamps = []
        aspects = []
        bbox_sizes = []
        gt_actions = defaultdict(int)

        for entry in track_data:
            cx = (entry["x1"] + entry["x2"]) / 2
            cy = (entry["y1"] + entry["y2"]) / 2
            bw = entry["x2"] - entry["x1"]
            bh = entry["y2"] - entry["y1"]
            aspect = bh / max(bw, 1)

            centroids.append((cx, cy))
            timestamps.append(entry["frame"] / fps)
            aspects.append(aspect)
            bbox_sizes.append(max(bw, bh))

            for act in entry["actions"]:
                gt_actions[act] += 1

        # Get dominant GT action
        if not gt_actions:
            continue
        gt_action = max(gt_actions, key=gt_actions.get)
        sar_label = okutama_to_sar.get(gt_action, "unknown")
        action_counts[gt_action] += 1

        # Run TMS
        tf = TrajectoryFeatures(centroids, timestamps, aspects, frame_dims, bbox_sizes)
        best_label, best_score = "unknown", 0.0
        rule_scores = {}
        for rule in TMS_RULES:
            score = rule.score(tf.features)
            rule_scores[rule.label] = round(score, 4)
            if score > best_score:
                best_score = score
                best_label = rule.label

        # Check if TMS prediction matches mapped SAR label
        is_correct = best_label == sar_label
        if is_correct:
            tms_correct += 1
        tms_total += 1

        results["confusion"][gt_action][best_label] = \
            results["confusion"][gt_action].get(best_label, 0) + 1

        results["tracks"].append({
            "track_id": tid,
            "length": len(track_data),
            "gt_action": gt_action,
            "gt_sar_label": sar_label,
            "tms_label": best_label,
            "tms_conf": round(best_score, 4),
            "correct": is_correct,
            "mean_size_px": round(float(np.mean(bbox_sizes)), 1),
        })

    overall_acc = tms_correct / max(tms_total, 1)
    results["overall_accuracy"] = round(overall_acc, 4)
    results["total_tracks"] = tms_total
    results["action_distribution"] = dict(action_counts)

    print(f"  TMS accuracy on Okutama: {overall_acc:.1%} ({tms_correct}/{tms_total})")
    print(f"  Action distribution: {dict(action_counts)}")

    # Save results
    serializable = {
        "overall_accuracy": results["overall_accuracy"],
        "total_tracks": results["total_tracks"],
        "action_distribution": results["action_distribution"],
        "confusion": {k: dict(v) for k, v in results["confusion"].items()},
        "tracks": results["tracks"][:50],  # save first 50 for brevity
    }
    with open(RESULTS_DIR / "okutama_tms.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: GT action distribution
    ax = axes[0]
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    act_names = [a[0] for a in sorted_actions]
    act_counts = [a[1] for a in sorted_actions]
    colors_bar = ["#e74c3c" if okutama_to_sar.get(a, "") in ["running", "falling"] else
                  "#f39c12" if okutama_to_sar.get(a, "") in ["lying_down", "crawling"] else
                  "#3498db" for a in act_names]
    bars = ax.barh(act_names, act_counts, color=colors_bar, alpha=0.85)
    for bar, count in zip(bars, act_counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               str(count), va="center", fontsize=10)
    ax.set_xlabel("Number of Tracks", fontsize=11)
    ax.set_title("Okutama-Action Distribution\n(Ground Truth)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Confusion matrix
    ax = axes[1]
    okutama_actions = [a for a in act_names if action_counts.get(a, 0) >= 3]
    tms_labels = sorted(set(t["tms_label"] for t in results["tracks"]))

    conf_matrix = np.zeros((len(okutama_actions), len(tms_labels)))
    for oa_idx, oa in enumerate(okutama_actions):
        confusion_row = results["confusion"].get(oa, {})
        for tl_idx, tl in enumerate(tms_labels):
            conf_matrix[oa_idx, tl_idx] = confusion_row.get(tl, 0)

    # Normalise rows
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_matrix_norm = conf_matrix / row_sums

    im = ax.imshow(conf_matrix_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(tms_labels)))
    ax.set_xticklabels(tms_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(okutama_actions)))
    ax.set_yticklabels(okutama_actions, fontsize=9)
    ax.set_xlabel("TMS Prediction", fontsize=11)
    ax.set_ylabel("Okutama GT Action", fontsize=11)
    ax.set_title("TMS vs Okutama GT\nCross-Domain Confusion", fontsize=13, fontweight="bold")

    for i in range(len(okutama_actions)):
        for j in range(len(tms_labels)):
            val = conf_matrix_norm[i, j]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                       fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Per-size accuracy
    ax = axes[2]
    size_bins = [(0, 30, "Tiny (<30px)"), (30, 60, "Small (30-60px)"),
                 (60, 120, "Medium (60-120px)"), (120, 9999, "Large (>120px)")]
    bin_accs = []
    bin_names = []
    bin_counts = []

    for lo, hi, name in size_bins:
        matching = [t for t in results["tracks"] if lo <= t["mean_size_px"] < hi]
        if matching:
            acc = sum(1 for t in matching if t["correct"]) / len(matching)
            bin_accs.append(acc)
            bin_names.append(name)
            bin_counts.append(len(matching))

    if bin_accs:
        bars = ax.bar(bin_names, bin_accs, color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"],
                     alpha=0.85)
        for bar, acc, count in zip(bars, bin_accs, bin_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{acc:.0%}\n(n={count})", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("TMS Accuracy", fontsize=11)
        ax.set_title("TMS Accuracy by Person Size\n(Real Aerial Data)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, f"Overall: {overall_acc:.0%}", transform=ax.transAxes,
               ha="center", fontsize=16, fontweight="bold")
        ax.set_title("TMS on Okutama", fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"SARTriage TMS on Okutama-Action (Real Drone Footage)\n"
                 f"Overall Accuracy: {overall_acc:.1%} across {tms_total} tracks",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "okutama_tms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ okutama_tms.png")


# ══════════════════════════════════════════════════════════════════════════
# Experiment 3: HERIDAL Size Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════

def run_heridal_size_analysis():
    """Analyse person size distribution in HERIDAL to validate small-object claims."""
    import matplotlib.pyplot as plt

    print("  Analysing HERIDAL person size distribution...")

    all_sizes = []
    for split in ["train", "valid", "test"]:
        data = load_heridal_gt(split)
        for item in data:
            img = cv2.imread(item["image"])
            if img is None:
                continue
            h, w = img.shape[:2]
            for bbox in item["bboxes"]:
                px_w = bbox["w"] * w
                px_h = bbox["h"] * h
                all_sizes.append(max(px_w, px_h))

    if not all_sizes:
        print("  ⚠ No size data")
        return

    all_sizes = np.array(all_sizes)

    stats = {
        "total_annotations": len(all_sizes),
        "mean_px": round(float(np.mean(all_sizes)), 1),
        "median_px": round(float(np.median(all_sizes)), 1),
        "min_px": round(float(np.min(all_sizes)), 1),
        "max_px": round(float(np.max(all_sizes)), 1),
        "pct_under_20px": round(float(np.mean(all_sizes < 20)) * 100, 1),
        "pct_under_30px": round(float(np.mean(all_sizes < 30)) * 100, 1),
        "pct_under_50px": round(float(np.mean(all_sizes < 50)) * 100, 1),
    }

    with open(RESULTS_DIR / "heridal_size_analysis.json", "w") as f:
        json.dump(stats, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(all_sizes, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
    ax1.axvline(x=20, color="#3498db", linestyle="--", linewidth=2, label=f"20px ({stats['pct_under_20px']:.0f}% below)")
    ax1.axvline(x=30, color="#2ecc71", linestyle="--", linewidth=2, label=f"30px ({stats['pct_under_30px']:.0f}% below)")
    ax1.axvline(x=50, color="#f39c12", linestyle="--", linewidth=2, label=f"50px ({stats['pct_under_50px']:.0f}% below)")
    ax1.set_xlabel("Person Size (pixels)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("HERIDAL Person Size Distribution\n(Real SAR Wilderness Imagery)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # CDF
    sorted_sizes = np.sort(all_sizes)
    cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    ax2.plot(sorted_sizes, cdf, color="#e74c3c", linewidth=2)
    ax2.fill_between(sorted_sizes, cdf, alpha=0.15, color="#e74c3c")
    ax2.axhline(y=0.5, color="#7f8c8d", linestyle=":", alpha=0.5)
    ax2.axvline(x=float(np.median(all_sizes)), color="#7f8c8d", linestyle=":", alpha=0.5)
    ax2.text(float(np.median(all_sizes)), 0.05,
            f"Median: {np.median(all_sizes):.0f}px", fontsize=10, color="#7f8c8d")
    ax2.set_xlabel("Person Size (pixels)", fontsize=12)
    ax2.set_ylabel("Cumulative Proportion", fontsize=12)
    ax2.set_title("Cumulative Distribution\n(How small are SAR targets?)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"HERIDAL: {stats['pct_under_30px']:.0f}% of persons are <30px\n"
                 f"This validates the need for TMS (trajectory-based classification)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heridal_size_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ heridal_size_analysis.png (median={np.median(all_sizes):.0f}px, "
          f"{stats['pct_under_30px']:.0f}% under 30px)")


# ══════════════════════════════════════════════════════════════════════════

def main():
    setup()
    print("=" * 60)
    print("  SARTriage — SAR Dataset Evaluation")
    print("=" * 60)

    print("\n🔬 Experiment 1: HERIDAL SAR Person Detection")
    run_heridal_detection()

    print("\n🔬 Experiment 2: Okutama-Action TMS Evaluation")
    run_okutama_tms()

    print("\n🔬 Experiment 3: HERIDAL Person Size Analysis")
    run_heridal_size_analysis()

    print(f"\n{'='*60}")
    print(f"  ✓ All SAR dataset evaluations complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
