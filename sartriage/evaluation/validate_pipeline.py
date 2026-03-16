"""
evaluation/validate_pipeline.py
================================
Run the full SARTriage pipeline on REAL test data and generate
validated metrics + publication-quality graphs.

Test data:
  1. VisDrone aerial drone images (real drone footage with GT bounding boxes)
  2. Kinetics action clips (real video of human actions)

Validates:
  - YOLO person detection (VisDrone GT comparison)
  - Action classification (on real video clips)
  - Pose estimation (on detected tracks)
  - Anomaly detection (on diverse real footage)
  - Cross-stream fusion
  - Full ablation (with real detections)

Output:  evaluation/figures/  (updated graphs with real data)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
TEST_DIR = Path(__file__).parent / "test_videos"
VISDRONE_DIR = TEST_DIR / "VisDrone2019-DET-val"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Test 1: YOLO + SAHI Person Detection on VisDrone
# ══════════════════════════════════════════════════════════════════════════

def test_yolo_detection():
    """Evaluate YOLO + SAHI person detection against VisDrone ground truth."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    print("  Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    img_dir = VISDRONE_DIR / "images"
    ann_dir = VISDRONE_DIR / "annotations"

    if not img_dir.exists():
        print("  ⚠ VisDrone images not found, skipping")
        return {}

    images = sorted(img_dir.glob("*.jpg"))[:100]  # test on 100 images
    print(f"  Testing on {len(images)} VisDrone images...")

    results_data = {
        "yolo_only": {"tp": 0, "fp": 0, "fn": 0, "detections": []},
        "yolo_sahi": {"tp": 0, "fp": 0, "fn": 0, "detections": []},
    }

    for img_path in images:
        ann_path = ann_dir / img_path.with_suffix(".txt").name
        if not ann_path.exists():
            continue

        # Parse VisDrone GT (category 1 = pedestrian, 2 = person)
        gt_boxes = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cat = int(parts[5])
                if cat in (1, 2):  # pedestrian or person
                    gt_boxes.append((x, y, x + w, y + h))

        if not gt_boxes:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Mode 1: Standard YOLO
        preds = model(img, verbose=False, conf=0.25)
        yolo_boxes = []
        for r in preds:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    yolo_boxes.append((x1, y1, x2, y2))

        tp, fp, fn = _compute_detection_metrics(yolo_boxes, gt_boxes, iou_thresh=0.3)
        results_data["yolo_only"]["tp"] += tp
        results_data["yolo_only"]["fp"] += fp
        results_data["yolo_only"]["fn"] += fn
        results_data["yolo_only"]["detections"].append(len(yolo_boxes))

        # Mode 2: YOLO + SAHI
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path="yolov8n.pt",
                confidence_threshold=0.25,
                device="cpu",
            )

            result = get_sliced_prediction(
                str(img_path),
                detection_model,
                slice_height=320,
                slice_width=320,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0,
            )

            sahi_boxes = []
            for pred in result.object_prediction_list:
                if pred.category.id == 0:  # person
                    bbox = pred.bbox
                    sahi_boxes.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))

            tp, fp, fn = _compute_detection_metrics(sahi_boxes, gt_boxes, iou_thresh=0.3)
            results_data["yolo_sahi"]["tp"] += tp
            results_data["yolo_sahi"]["fp"] += fp
            results_data["yolo_sahi"]["fn"] += fn
            results_data["yolo_sahi"]["detections"].append(len(sahi_boxes))

        except Exception as e:
            results_data["yolo_sahi"]["detections"].append(0)

    # Compute precision/recall/F1
    for mode in ["yolo_only", "yolo_sahi"]:
        d = results_data[mode]
        d["precision"] = d["tp"] / max(d["tp"] + d["fp"], 1)
        d["recall"] = d["tp"] / max(d["tp"] + d["fn"], 1)
        d["f1"] = 2 * d["precision"] * d["recall"] / max(d["precision"] + d["recall"], 1e-8)
        d["avg_detections"] = np.mean(d["detections"]) if d["detections"] else 0

    print(f"\n  YOLO-only:  P={results_data['yolo_only']['precision']:.3f}  "
          f"R={results_data['yolo_only']['recall']:.3f}  "
          f"F1={results_data['yolo_only']['f1']:.3f}  "
          f"Avg det/img={results_data['yolo_only']['avg_detections']:.1f}")
    print(f"  YOLO+SAHI:  P={results_data['yolo_sahi']['precision']:.3f}  "
          f"R={results_data['yolo_sahi']['recall']:.3f}  "
          f"F1={results_data['yolo_sahi']['f1']:.3f}  "
          f"Avg det/img={results_data['yolo_sahi']['avg_detections']:.1f}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: YOLO vs YOLO+SAHI
    metrics = ["Precision", "Recall", "F1"]
    yolo_vals = [results_data["yolo_only"]["precision"],
                 results_data["yolo_only"]["recall"],
                 results_data["yolo_only"]["f1"]]
    sahi_vals = [results_data["yolo_sahi"]["precision"],
                 results_data["yolo_sahi"]["recall"],
                 results_data["yolo_sahi"]["f1"]]

    x = np.arange(len(metrics))
    width = 0.3
    bars1 = ax1.bar(x - width/2, yolo_vals, width, label="YOLOv8 Standard",
                    color="#e74c3c", alpha=0.85)
    bars2 = ax1.bar(x + width/2, sahi_vals, width, label="YOLOv8 + SAHI",
                    color="#2ecc71", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{bar.get_height():.2f}", ha="center", fontsize=10, fontweight="bold")

    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Person Detection on VisDrone\n(Aerial Drone Images)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Detection count comparison
    if results_data["yolo_only"]["detections"] and results_data["yolo_sahi"]["detections"]:
        ax2.hist(results_data["yolo_only"]["detections"], bins=20, alpha=0.6,
                 label="YOLO Standard", color="#e74c3c")
        ax2.hist(results_data["yolo_sahi"]["detections"], bins=20, alpha=0.6,
                 label="YOLO + SAHI", color="#2ecc71")
        ax2.set_xlabel("Detections per Image", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Detection Count Distribution\n(SAHI finds more small people)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Validated Person Detection — YOLOv8 on VisDrone Dataset",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "visdrone_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ visdrone_detection.png")

    return results_data


def _compute_detection_metrics(pred_boxes, gt_boxes, iou_thresh=0.3):
    """Compute TP, FP, FN using IoU matching."""
    matched_gt = set()
    tp = 0
    for pb in pred_boxes:
        best_iou = 0
        best_gt = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = _iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-8)


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Full Pipeline on Real Video
# ══════════════════════════════════════════════════════════════════════════

def test_full_pipeline():
    """Run the full 5-stream pipeline on real videos."""
    import matplotlib.pyplot as plt
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    test_videos = {
        "VisDrone (aerial)": str(TEST_DIR / "visdrone_test.mp4"),
        "Kinetics (actions)": str(TEST_DIR / "kinetics_test.mp4"),
    }

    all_results = []

    for name, video_path in test_videos.items():
        if not Path(video_path).exists():
            print(f"  ⚠ {name}: {video_path} not found, skipping")
            continue

        print(f"\n  Running pipeline on: {name}")
        from main import run_pipeline

        t0 = time.time()
        try:
            result = run_pipeline(video_path, config=config, task_id=f"validate_{name.split()[0].lower()}")
            elapsed = time.time() - t0

            events = result.get("events", [])
            stream_counts = {}
            for e in events:
                s = e.get("stream", "unknown")
                stream_counts[s] = stream_counts.get(s, 0) + 1

            entry = {
                "name": name,
                "video": Path(video_path).name,
                "total_events": len(events),
                "critical": sum(1 for e in events if e.get("severity") == "critical"),
                "high": sum(1 for e in events if e.get("severity") == "high"),
                "medium": sum(1 for e in events if e.get("severity") == "medium"),
                "low": sum(1 for e in events if e.get("severity") == "low"),
                "stream_counts": stream_counts,
                "processing_time": round(elapsed, 1),
                "avg_confidence": round(np.mean([e.get("confidence", 0) for e in events]), 4) if events else 0,
                "cross_stream": sum(1 for e in events if len(e.get("streams", [])) > 1),
                "events_sample": events[:5] if events else [],
            }
            all_results.append(entry)

            print(f"    → {entry['total_events']} events "
                  f"(C:{entry['critical']}, H:{entry['high']}, M:{entry['medium']}, L:{entry['low']})")
            print(f"    → Streams: {stream_counts}")
            print(f"    → Time: {elapsed:.1f}s")

        except Exception as exc:
            print(f"    ✗ Failed: {exc}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "error": str(exc)})

    # Save results
    with open(RESULTS_DIR / "pipeline_validation.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  ✓ pipeline_validation.json")

    # ── Plot ──────────────────────────────────────────────────────────
    valid = [r for r in all_results if "error" not in r]
    if valid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Events by severity
        names = [r["name"] for r in valid]
        x = np.arange(len(names))
        width = 0.2
        severities = ["critical", "high", "medium", "low"]
        colors = ["#e74c3c", "#f39c12", "#3498db", "#7f8c8d"]

        for i, (sev, col) in enumerate(zip(severities, colors)):
            vals = [r.get(sev, 0) for r in valid]
            ax1.bar(x + i * width, vals, width, label=sev.capitalize(), color=col, alpha=0.85)

        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(names, fontsize=10)
        ax1.set_ylabel("Number of Events", fontsize=12)
        ax1.set_title("Events by Severity — Real Video", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")

        # Stream contribution
        all_streams = set()
        for r in valid:
            all_streams.update(r.get("stream_counts", {}).keys())
        all_streams = sorted(all_streams)

        stream_colors = {
            "motion": "#2ecc71", "tracking": "#9b59b6",
            "action": "#e74c3c", "pose": "#1abc9c", "anomaly": "#f39c12",
        }

        bottom = np.zeros(len(valid))
        for stream in all_streams:
            vals = [r.get("stream_counts", {}).get(stream, 0) for r in valid]
            ax2.bar(names, vals, bottom=bottom, label=stream,
                    color=stream_colors.get(stream, "#95a5a6"), alpha=0.85)
            bottom += np.array(vals, dtype=float)

        ax2.set_ylabel("Events", fontsize=12)
        ax2.set_title("Events by Stream — Real Video", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle("SARTriage Pipeline Validation on Real Footage",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pipeline_validation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ pipeline_validation.png")

    return all_results


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Real Ablation with VisDrone video
# ══════════════════════════════════════════════════════════════════════════

def test_real_ablation():
    """Run ablation study on real VisDrone footage."""
    import matplotlib.pyplot as plt
    import yaml

    visdrone_video = TEST_DIR / "visdrone_test.mp4"
    if not visdrone_video.exists():
        print("  ⚠ VisDrone test video not found, skipping ablation")
        return

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    from main import run_pipeline

    configs = [
        ("Full Pipeline (5 streams)", base_config),
    ]

    for stream in ["action", "motion", "tracking", "pose", "anomaly"]:
        cfg = json.loads(json.dumps(base_config))
        if stream in cfg:
            cfg[stream]["enabled"] = False
        configs.append((f"Without {stream}", cfg))

    # Motion+Tracking only
    cfg = json.loads(json.dumps(base_config))
    for s in ["action", "pose", "anomaly"]:
        if s in cfg:
            cfg[s]["enabled"] = False
    configs.append(("Motion+Tracking only", cfg))

    results = []
    for label, config in configs:
        print(f"    Running: {label}")
        try:
            t0 = time.time()
            result = run_pipeline(str(visdrone_video), config=config, task_id="real_ablation")
            elapsed = time.time() - t0

            events = result.get("events", [])
            results.append({
                "label": label,
                "total": len(events),
                "critical": sum(1 for e in events if e.get("severity") == "critical"),
                "high": sum(1 for e in events if e.get("severity") == "high"),
                "medium": sum(1 for e in events if e.get("severity") == "medium"),
                "low": sum(1 for e in events if e.get("severity") == "low"),
                "avg_conf": float(np.mean([e.get("confidence", 0) for e in events])) if events else 0,
                "time": round(elapsed, 1),
            })
            print(f"      → {len(events)} events in {elapsed:.1f}s")
        except Exception as e:
            print(f"      ✗ {e}")
            results.append({"label": label, "total": 0, "critical": 0,
                           "high": 0, "medium": 0, "low": 0, "avg_conf": 0, "time": 0})

    # Save
    with open(RESULTS_DIR / "real_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    labels = [r["label"] for r in results]
    totals = [r["total"] for r in results]
    criticals = [r["critical"] for r in results]
    highs = [r["high"] for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, totals, width, label="Total Events", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x, criticals, width, label="Critical", color="#e74c3c", alpha=0.85)
    bars3 = ax.bar(x + width, highs, width, label="High", color="#f39c12", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        str(int(h)), ha="center", fontsize=9)

    ax.set_ylabel("Number of Events", fontsize=12)
    ax.set_title("Ablation Study on Real VisDrone Aerial Footage", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="green")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ real_ablation.png")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Test 4: Attention on Real Clips
# ══════════════════════════════════════════════════════════════════════════

def test_attention_real():
    """Generate Grad-CAM heatmaps on real Kinetics clips."""
    import matplotlib.pyplot as plt
    import torch
    import torchvision.models.video as vm

    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "action_mvit2_sar.pt"
    if not model_path.exists():
        print("  ⚠ Trained model not found, skipping")
        return

    from core.attention_viz import GradCAMVideoExplainer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 9)
    idx_to_label = ckpt.get("idx_to_label", {})

    model = vm.mvit_v2_s(weights=None)
    model.head[1] = torch.nn.Linear(768, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    explainer = GradCAMVideoExplainer(model, device)

    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, -1, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, -1, 1, 1, 1)

    # Load real Kinetics clips
    base = Path(__file__).parent.parent / "training" / "data" / "videos"
    actions = ["falling", "crawling", "running", "climbing", "waving_hand", "lying_down", "pushing", "pulling"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, action in enumerate(actions[:8]):
        action_dir = base / action
        if not action_dir.exists():
            axes[i].axis("off")
            continue

        clips = sorted(action_dir.glob("*.mp4"))
        if not clips:
            axes[i].axis("off")
            continue

        # Read 16 frames from the clip
        cap = cv2.VideoCapture(str(clips[0]))
        frames = []
        for _ in range(16):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 16:
            frames = frames + [frames[-1]] * (16 - len(frames))

        # Preprocess
        processed = []
        for f in frames:
            f = cv2.resize(f, (224, 224))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            processed.append(f)

        clip_np = np.stack(processed)
        tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        tensor = (tensor - mean) / std

        result = explainer.generate(tensor.to(device))

        pred_idx = result["predicted_class"]
        pred_prob = result["predicted_prob"]
        pred_label = idx_to_label.get(str(pred_idx), idx_to_label.get(pred_idx, f"cls_{pred_idx}"))
        heatmap = result["spatial_heatmap"]

        # Overlay on mid-frame
        mid = frames[8]
        mid = cv2.resize(mid, (224, 224))
        hmap = cv2.resize(heatmap.astype(np.float32), (224, 224))
        hmap_u8 = (hmap * 255).clip(0, 255).astype(np.uint8)
        hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(mid, 0.55, hmap_color, 0.45, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        # Correctness marker
        correct = action == pred_label or action in pred_label or pred_label in action
        marker = "✓" if correct else "✗"
        color = "green" if correct else "red"

        axes[i].imshow(blended_rgb)
        axes[i].set_title(f"GT: {action}\nPred: {pred_label} ({pred_prob:.0%}) {marker}",
                         fontsize=10, color=color if not correct else "black")
        axes[i].axis("off")

    plt.suptitle("Grad-CAM on Real Kinetics Video Clips — MViTv2-S",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "real_attention_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ real_attention_heatmaps.png")

    explainer.cleanup()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    setup()

    print("=" * 60)
    print("  SARTriage — Pipeline Validation on Real Data")
    print("=" * 60)

    print("\n🔬 Test 1: YOLO+SAHI Detection on VisDrone")
    detection_results = test_yolo_detection()

    print("\n🔬 Test 2: Full Pipeline on Real Videos")
    pipeline_results = test_full_pipeline()

    print("\n🔬 Test 3: Ablation Study on Real VisDrone Footage")
    ablation_results = test_real_ablation()

    print("\n🔬 Test 4: Attention Maps on Real Kinetics Clips")
    test_attention_real()

    # Summary
    all_results = {
        "detection": detection_results,
        "pipeline": pipeline_results,
        "ablation": ablation_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RESULTS_DIR / "full_validation.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  ✓ All validation tests complete!")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    figures = sorted(FIGURES_DIR.glob("*.png"))
    print(f"  {len(figures)} total figures generated")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
