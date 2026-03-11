"""
training/train_action_classifier.py
====================================
Fine-tune MViTv2-S (SOTA, 81% top-1 on K400) on SAR-relevant action classes.

This script:
  1. Downloads the Kinetics-400 annotation CSVs from DeepMind.
  2. Filters for SAR-relevant classes.
  3. Downloads the actual YouTube clips using yt-dlp.
  4. Extracts 16-frame sub-clips at the annotated timestamps.
  5. Fine-tunes a pre-trained R3D-18 (frozen backbone, trainable FC).
  6. Saves checkpoints after EVERY epoch for overnight resumption.

Usage
-----
    cd sartriage
    source venv/bin/activate
    python training/train_action_classifier.py

To resume from a checkpoint::

    python training/train_action_classifier.py --resume

The script will automatically find the latest checkpoint and continue
from there. Safe to Ctrl+C and restart at any time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.models.video as video_models


# ── Config ───────────────────────────────────────────────────────────────────

# Mapping: our SAR label → list of Kinetics-400 class names
SAR_TO_KINETICS: Dict[str, List[str]] = {
    "falling":      ["faceplanting"],
    "crawling":     ["crawling baby"],
    "lying_down":   ["stretching arm", "stretching leg"],
    "running":      ["jogging", "running on treadmill"],
    "waving_hand":  ["clapping", "pumping fist"],
    "climbing":     ["climbing a rope", "climbing ladder", "climbing tree",
                     "rock climbing", "ice climbing"],
    "stumbling":    ["stumbling"],  # falls back to synthetic if not in K400
    "pushing":      ["pushing car", "pushing cart", "pushing wheelchair"],
    "pulling":      ["pull ups"],
}

# Flat list of our labels
SAR_LABELS: List[str] = list(SAR_TO_KINETICS.keys())
NUM_CLASSES = len(SAR_LABELS)
LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(SAR_LABELS)}
IDX_TO_LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(SAR_LABELS)}

# Reverse mapping: kinetics class name → our SAR label
KINETICS_TO_SAR: Dict[str, str] = {}
for sar_label, kinetics_names in SAR_TO_KINETICS.items():
    for kn in kinetics_names:
        KINETICS_TO_SAR[kn] = sar_label

# Training hyper-parameters
CLIP_FRAMES = 16
CLIP_SIZE = 224               # MViTv2 expects 224×224 input (was 112 for R3D-18)
BATCH_SIZE = 4            # smaller batch for overnight stability
EPOCHS = 30               # more epochs for real data
LEARNING_RATE = 1e-3
MAX_CLIPS_PER_CLASS = 150  # cap per SAR class (combine kinetics sub-classes)
VAL_SPLIT = 0.2

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
CLIPS_DIR = DATA_DIR / "clips"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = BASE_DIR.parent / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "action_mvit2_sar.pt"
TRAINING_LOG_PATH = MODELS_DIR / "training_log.json"

# Kinetics-400 annotation URLs (from cvdfoundation/kinetics-dataset)
KINETICS_URLS = {
    "train": "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
    "validate": "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
}


# ── 1. Download Annotations ─────────────────────────────────────────────────

def download_annotations() -> Dict[str, Path]:
    """Download Kinetics-400 annotation CSVs."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in KINETICS_URLS.items():
        dest = ANNOTATIONS_DIR / f"{split}.csv"
        if dest.exists():
            print(f"  ✓ {split}.csv already exists")
        else:
            print(f"  ↓ Downloading {split}.csv...")
            import urllib.request
            urllib.request.urlretrieve(url, str(dest))
            print(f"  ✓ Saved {dest}")
        paths[split] = dest
    return paths


# ── 2. Filter Annotations for SAR Classes ────────────────────────────────

def filter_annotations(csv_path: Path, split: str) -> List[Dict]:
    """Read a Kinetics CSV and return rows matching SAR-relevant classes."""
    filtered = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kinetics_label = row.get("label", "").strip()
            if kinetics_label in KINETICS_TO_SAR:
                filtered.append({
                    "youtube_id": row.get("youtube_id", "").strip(),
                    "time_start": int(row.get("time_start", 0)),
                    "time_end": int(row.get("time_end", 0)),
                    "kinetics_label": kinetics_label,
                    "sar_label": KINETICS_TO_SAR[kinetics_label],
                    "split": split,
                })
    return filtered


def balance_and_cap(
    annotations: List[Dict],
    max_per_class: int = MAX_CLIPS_PER_CLASS,
) -> List[Dict]:
    """Balance classes and cap at max_per_class clips each."""
    by_class: Dict[str, List[Dict]] = {lbl: [] for lbl in SAR_LABELS}
    for ann in annotations:
        by_class[ann["sar_label"]].append(ann)

    balanced = []
    for lbl, anns in by_class.items():
        random.seed(42)
        random.shuffle(anns)
        capped = anns[:max_per_class]
        balanced.extend(capped)
        print(f"    {lbl:<15}: {len(capped):>4} clips (from {len(anns)} available)")

    return balanced


# ── 3. Download Videos ───────────────────────────────────────────────────

def download_video(youtube_id: str, time_start: int, time_end: int, output_path: Path) -> bool:
    """Download a YouTube clip using yt-dlp."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True  # already downloaded

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "yt-dlp",
            "--quiet", "--no-warnings",
            "-f", "worst[ext=mp4]/worst",  # smallest quality to save space/time
            "--download-sections", f"*{time_start}-{time_end}",
            "--force-keyframes-at-cuts",
            "-o", str(output_path),
            f"https://www.youtube.com/watch?v={youtube_id}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return output_path.exists() and output_path.stat().st_size > 1000
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def download_all_videos(annotations: List[Dict], progress_file: Path) -> List[Dict]:
    """Download all videos, tracking progress for resumption."""
    # Load existing progress
    downloaded = set()
    if progress_file.exists():
        with open(progress_file, "r") as f:
            downloaded = set(json.load(f))

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    successful = []
    failed = 0
    skipped = 0

    for i, ann in enumerate(annotations):
        vid_id = ann["youtube_id"]
        filename = f"{vid_id}_{ann['time_start']}_{ann['time_end']}.mp4"
        output_path = VIDEOS_DIR / ann["sar_label"] / filename

        if filename in downloaded:
            if output_path.exists():
                ann["video_path"] = str(output_path)
                successful.append(ann)
                skipped += 1
            continue

        # Show progress
        print(f"\r  [{i+1}/{len(annotations)}] Downloading {vid_id} "
              f"({ann['sar_label']}) — {len(successful)} ok, {failed} failed, "
              f"{skipped} skipped", end="", flush=True)

        ok = download_video(vid_id, ann["time_start"], ann["time_end"], output_path)

        if ok:
            ann["video_path"] = str(output_path)
            successful.append(ann)
        else:
            failed += 1

        # Save progress incrementally
        downloaded.add(filename)
        with open(progress_file, "w") as f:
            json.dump(list(downloaded), f)

    print(f"\n  ✓ Downloaded: {len(successful)}, Failed/unavailable: {failed}, Skipped: {skipped}")
    return successful


# ── 4. Video Clip Dataset ────────────────────────────────────────────────

class KineticsClipDataset(Dataset):
    """Load real video clips from downloaded Kinetics-400 files."""

    def __init__(
        self,
        annotations: List[Dict],
        clip_frames: int = CLIP_FRAMES,
        clip_size: int = CLIP_SIZE,
    ):
        self.annotations = annotations
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(-1, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(-1, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ann = self.annotations[idx]
        video_path = ann["video_path"]
        label_idx = LABEL_TO_IDX[ann["sar_label"]]

        try:
            clip = self._load_clip(video_path)
        except Exception:
            # Fallback: return a zero clip (will be a minority)
            clip = torch.zeros(3, self.clip_frames, self.clip_size, self.clip_size)
            return clip, label_idx

        return clip, label_idx

    def _load_clip(self, video_path: str) -> torch.Tensor:
        """Extract clip_frames evenly-spaced frames from the video."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise RuntimeError(f"Empty video {video_path}")

        # Sample clip_frames evenly
        indices = np.linspace(0, total_frames - 1, self.clip_frames, dtype=int)
        frames = []

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.clip_size, self.clip_size, 3), dtype=np.uint8)

            # Resize to clip_size × clip_size
            frame = cv2.resize(frame, (self.clip_size, self.clip_size))
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Stack: (T, H, W, 3) → (3, T, H, W) float32 normalised
        clip = np.stack(frames, axis=0)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        clip = (clip - self.mean) / self.std

        return clip


# ── 5. Synthetic Fallback Dataset ────────────────────────────────────────

class SyntheticSARClipDataset(Dataset):
    """Generate synthetic clips as fallback if Kinetics downloads fail.

    Uses the SAME interface as KineticsClipDataset so the training
    loop doesn't need to change.
    """

    def __init__(self, clips_per_class: int = 80, clip_frames: int = CLIP_FRAMES,
                 clip_size: int = CLIP_SIZE):
        self.clips_per_class = clips_per_class
        self.clip_frames = clip_frames
        self.clip_size = clip_size
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(-1, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(-1, 1, 1, 1)
        self.samples: List[Tuple[int, int]] = []
        for cls_idx in range(NUM_CLASSES):
            for v in range(clips_per_class):
                self.samples.append((cls_idx, cls_idx * 10000 + v))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cls_idx, seed = self.samples[idx]
        rng = np.random.RandomState(seed)
        T, H, W = self.clip_frames, self.clip_size, self.clip_size

        clip = np.zeros((T, H, W, 3), dtype=np.uint8)
        bg = rng.randint(10, 40, size=3).astype(np.uint8)
        clip[:] = bg
        noise = rng.randint(0, 20, size=(T, H, W, 3), dtype=np.uint8)
        clip = np.clip(clip.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        color = rng.randint(150, 255, size=3).astype(np.uint8)
        ph, pw = H // 4, W // 4
        label = SAR_LABELS[cls_idx]

        for t in range(T):
            frac = t / T
            if label == "falling":
                y = int(frac * (H - ph)); x = W//2 - pw//2
            elif label == "crawling":
                y = int(H * 0.7); x = int(frac * (W - pw) * 0.5)
            elif label == "lying_down":
                y = int(H * 0.75); x = W//6; pw = W*2//3
            elif label == "running":
                y = H//2 - ph//2; x = int(frac * (W - pw))
            elif label == "waving_hand":
                y = H//6; x = W//2 + int(np.sin(t * np.pi/3) * W//4) - pw//2
            elif label == "climbing":
                y = int((1-frac) * (H - ph)); x = W//2 - pw//2
            elif label == "stumbling":
                y = H//2 + rng.randint(-15,15); x = W//2 + rng.randint(-15,15)
            elif label == "pushing":
                s = 0.5 + frac*0.5; pph=int(ph*s); ppw=int(pw*s)
                y = H//2 - pph//2; x = W//2 - ppw//2; ph, pw = pph, ppw
            elif label == "pulling":
                s = 1 - frac*0.5; pph=int(ph*s); ppw=int(pw*s)
                y = H//2 - pph//2; x = W//2 - ppw//2; ph, pw = pph, ppw
            else:
                y, x = H//2, W//2

            y = np.clip(y, 0, H - max(ph, 1))
            x = np.clip(x, 0, W - max(pw, 1))
            clip[t, y:y+ph, x:x+pw] = color

        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor, cls_idx


# ── 6. Model ────────────────────────────────────────────────────────────

def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Load pre-trained MViTv2-S and replace classifier head.

    MViTv2-S (Multiscale Vision Transformer v2 - Small) achieves
    81.0% top-1 on Kinetics-400, making it SOTA for video
    action recognition.  The head is:
        Sequential(Dropout(0.5), Linear(768, 400))
    We replace the Linear layer with one mapping to num_classes.
    """
    print("  Loading pre-trained MViTv2-S (Kinetics-400 weights)...")
    weights = video_models.MViT_V2_S_Weights.KINETICS400_V1
    model = video_models.mvit_v2_s(weights=weights)

    if freeze_backbone:
        print("  Freezing backbone (transfer learning)...")
        for param in model.parameters():
            param.requires_grad = False

    # Head structure: Sequential(Dropout(0.5), Linear(768, 400))
    in_features = model.head[1].in_features  # 768
    model.head[1] = nn.Linear(in_features, num_classes)

    for param in model.head.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)")
    print(f"  Head: {model.head}")
    return model


# ── 7. Training Loop ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * clips.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# ── 8. Checkpoint Management ────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    val_acc: float,
    val_loss: float,
    is_best: bool,
    training_log: List[Dict],
) -> None:
    """Save checkpoint with full state for resumption."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "num_classes": NUM_CLASSES,
        "labels": SAR_LABELS,
        "label_to_idx": LABEL_TO_IDX,
        "idx_to_label": IDX_TO_LABEL,
        "clip_frames": CLIP_FRAMES,
        "clip_size": CLIP_SIZE,
        "training_log": training_log,
    }

    # Save epoch checkpoint (always)
    epoch_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, epoch_path)
    print(f"    💾 Checkpoint saved: {epoch_path.name}")

    # Save latest (for easy resumption)
    latest_path = CHECKPOINT_DIR / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save best model
    if is_best:
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"    ⭐ New best model saved (val_acc={val_acc:.1%})")

    # Save training log as JSON (human-readable)
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)


def load_checkpoint(model, optimizer, scheduler, device) -> Tuple[int, float, List[Dict]]:
    """Load the latest checkpoint if it exists."""
    latest_path = CHECKPOINT_DIR / "checkpoint_latest.pt"
    if not latest_path.exists():
        return 0, 0.0, []

    print(f"  📂 Resuming from {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_acc = checkpoint.get("val_acc", 0.0)
    training_log = checkpoint.get("training_log", [])

    print(f"  ✓ Resumed from epoch {start_epoch} (best val_acc={best_val_acc:.1%})")
    return start_epoch, best_val_acc, training_log


# ── 9. Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SARTriage — Train Action Classifier")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of Kinetics-400 (for testing)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-clips", type=int, default=MAX_CLIPS_PER_CLASS,
                        help="Max clips per SAR class")
    args = parser.parse_args()

    print("=" * 60)
    print("  SARTriage — Action Classifier Training")
    print("  Model:  MViTv2-S (SOTA, 81% top-1 on K400)")
    print(f"  Input:   {CLIP_FRAMES} frames × {CLIP_SIZE}×{CLIP_SIZE}")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(SAR_LABELS)})")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Data:    {'Synthetic' if args.synthetic else 'Kinetics-400 (real)'}")
    print(f"  Resume:  {'Yes' if args.resume else 'No'}")
    print("=" * 60)

    # ── Device ───────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✓ Using Apple MPS (Metal) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"\n⚠ Using CPU (training will be slower)")

    # ── Dataset ──────────────────────────────────────────────────────
    if args.synthetic:
        print("\nGenerating synthetic training data...")
        full_dataset = SyntheticSARClipDataset(clips_per_class=80)
    else:
        print("\n── Step 1: Download Kinetics-400 annotations ──")
        csv_paths = download_annotations()

        print("\n── Step 2: Filter for SAR-relevant classes ──")
        train_anns = filter_annotations(csv_paths["train"], "train")
        val_anns = filter_annotations(csv_paths["validate"], "validate")
        print(f"  Found {len(train_anns)} training + {len(val_anns)} validation clips")

        print("\n  Balancing training set:")
        train_anns = balance_and_cap(train_anns, args.max_clips)

        print("\n  Balancing validation set:")
        val_anns = balance_and_cap(val_anns, max(args.max_clips // 5, 20))

        print(f"\n── Step 3: Download video clips ──")
        print("  (This may take a while on first run. Progress is saved.)")
        print("  (Safe to Ctrl+C and restart — downloads will resume.)\n")

        # Check for yt-dlp
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("  ⚠ yt-dlp not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"],
                           capture_output=True)

        progress_file = DATA_DIR / "download_progress.json"
        print("  Downloading training videos:")
        train_anns = download_all_videos(train_anns, progress_file)

        print("  Downloading validation videos:")
        val_progress = DATA_DIR / "download_progress_val.json"
        val_anns = download_all_videos(val_anns, val_progress)

        if len(train_anns) < 10:
            print("\n  ⚠ Too few videos downloaded. Falling back to synthetic data.")
            print("  (Many Kinetics-400 YouTube videos have been removed.)")
            print("  Tip: Add your own labelled SAR clips to training/data/videos/<class>/\n")
            args.synthetic = True
            full_dataset = SyntheticSARClipDataset(clips_per_class=80)

    # Create datasets
    if args.synthetic:
        total_size = len(full_dataset)
        val_size = int(total_size * VAL_SPLIT)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        # Check for opencv
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("  Installing opencv-python-headless...")
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"],
                           capture_output=True)

        train_dataset = KineticsClipDataset(train_anns)
        val_dataset = KineticsClipDataset(val_anns)

    print(f"\n  Training clips:   {len(train_dataset)}")
    print(f"  Validation clips: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ── Model ────────────────────────────────────────────────────────
    print()
    model = build_model(NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    training_log: List[Dict] = []

    if args.resume:
        start_epoch, best_val_acc, training_log = load_checkpoint(
            model, optimizer, scheduler, device
        )

    # ── Training ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Epoch':>6} │ {'Train Loss':>11} │ {'Train Acc':>10} │ "
          f"{'Val Loss':>10} │ {'Val Acc':>9} │ {'LR':>8} │ {'Time':>6}")
    print(f"{'─'*65}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        print(
            f"  {epoch:>4} │ {train_loss:>11.4f} │ {train_acc:>9.1%} │ "
            f"{val_loss:>10.4f} │ {val_acc:>8.1%} │ {lr:>8.6f} │ {elapsed:>5.1f}s"
        )

        # Log entry
        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(lr, 8),
            "elapsed_s": round(elapsed, 1),
            "is_best": is_best,
        }
        training_log.append(log_entry)

        # Save checkpoint after EVERY epoch
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            val_acc, val_loss, is_best, training_log,
        )

    print(f"{'─'*65}")
    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.1%}")
    print(f"  Best model: {BEST_MODEL_PATH}")
    print(f"  Training log: {TRAINING_LOG_PATH}")
    print(f"  All checkpoints: {CHECKPOINT_DIR}/")

    # ── Per-class accuracy ───────────────────────────────────────────
    print(f"\n  Per-class validation accuracy:")
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            _, predicted = outputs.max(1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    for cls_idx in range(NUM_CLASSES):
        if class_total[cls_idx] > 0:
            acc = class_correct[cls_idx] / class_total[cls_idx]
            print(f"    {SAR_LABELS[cls_idx]:<15}: {acc:.1%} "
                  f"({class_correct[cls_idx]}/{class_total[cls_idx]})")
        else:
            print(f"    {SAR_LABELS[cls_idx]:<15}: N/A (no val samples)")

    print(f"\nTo resume training, run:")
    print(f"  python training/train_action_classifier.py --resume --epochs {args.epochs + 10}")


if __name__ == "__main__":
    main()
