"""
streams/action_classifier.py
=============================
Stream 1 — Action Classification using a trained R3D-18 model.

Analyses sliding-window clips of N frames for SAR-relevant human
actions (falling, crawling, waving, etc.) using a fine-tuned R3D-18
backbone with Kinetics-400 transfer learning.

The model is loaded from the checkpoint saved by the training script
(``training/train_action_classifier.py``).  If no trained model is
found, falls back to stub mode with random predictions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent

log = logging.getLogger("sartriage.action")


class ActionClassifierStream(BaseStream):
    """Stream 1: clip-level action classification using R3D-18."""

    @property
    def name(self) -> str:
        return "action"

    def setup(self) -> None:
        """Load the trained R3D-18 model from checkpoint."""
        self._clip_length = self.global_config.get("video", {}).get("clip_length", 16)
        self._clip_stride = self.global_config.get("video", {}).get("clip_stride", 8)
        self._clip_size = self.config.get("clip_size", 112)
        self._confidence_gate = self.config.get("confidence_gate", 0.40)
        self._z_threshold = self.config.get("z_score_threshold", 2.0)
        self._sar_labels = self.config.get("sar_action_labels", [])
        self._top_k = self.config.get("top_k", 3)

        # Normalisation constants (Kinetics-400 pre-training stats)
        self._mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, -1, 1, 1, 1)
        self._std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, -1, 1, 1, 1)

        # Load trained model
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cpu")
        self._stub_mode = False

        model_path = self.config.get("model_path", "")
        if model_path:
            full_path = Path(__file__).parent.parent / model_path
            if full_path.exists():
                self._load_model(full_path)
            else:
                log.warning("Model file not found at %s — using stub mode", full_path)
                self._stub_mode = True
        else:
            log.warning("No model_path in config — using stub mode")
            self._stub_mode = True

        super().setup()

    def _load_model(self, path: Path) -> None:
        """Load the R3D-18 model from a checkpoint file."""
        import torchvision.models.video as video_models

        # Select device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        log.info("Loading action classifier from %s (device=%s)", path.name, self._device)

        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        # Reconstruct the model architecture
        num_classes = checkpoint.get("num_classes", len(self._sar_labels))
        self._idx_to_label = checkpoint.get("idx_to_label", {})

        model = video_models.r3d_18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()

        self._model = model
        val_acc = checkpoint.get("val_acc", 0)
        epoch = checkpoint.get("epoch", "?")
        log.info("Model loaded — epoch %s, val_acc=%.1f%%", epoch, val_acc * 100)

    def _preprocess_clip(self, packets: Sequence[FramePacket]) -> torch.Tensor:
        """Convert a sequence of FramePackets to a model-ready tensor.

        Returns shape: (1, 3, T, H, W) normalised float32.
        """
        import cv2

        frames = []
        for pkt in packets:
            img = pkt.image
            if img is None:
                img = np.zeros((self._clip_size, self._clip_size, 3), dtype=np.uint8)

            # Resize to clip_size × clip_size
            if img.shape[0] != self._clip_size or img.shape[1] != self._clip_size:
                img = cv2.resize(img, (self._clip_size, self._clip_size))

            # Ensure RGB (if BGR from OpenCV)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frames.append(img)

        # Stack: (T, H, W, 3) → (1, 3, T, H, W)
        clip = np.stack(frames, axis=0)
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

        # Normalise
        tensor = (tensor - self._mean) / self._std

        return tensor.to(self._device)

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Run action classification over sliding-window clips.

        For each clip window:
          1. Pre-process frames → (1, 3, T, H, W) tensor
          2. Forward pass through R3D-18
          3. Softmax → top-k class predictions
          4. If max confidence > gate AND z-score > threshold → emit event
        """
        if not packets:
            return []

        if self._stub_mode:
            return self._detect_stub(packets)

        events: List[SAREvent] = []
        fps = self.global_config.get("video", {}).get("target_fps", 5)

        # Collect per-clip max scores for z-score computation
        clip_scores: List[float] = []
        clip_data: List[Dict[str, Any]] = []

        n_clips = max(1, (len(packets) - self._clip_length) // self._clip_stride + 1)
        log.info("Running action classifier on %d clips (device=%s)", n_clips, self._device)

        with torch.no_grad():
            for i in range(0, len(packets) - self._clip_length + 1, self._clip_stride):
                clip_packets = packets[i: i + self._clip_length]

                # Pre-process and run model
                tensor = self._preprocess_clip(clip_packets)
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)  # shape: (num_classes,)

                # Get top-k predictions
                topk_probs, topk_indices = probs.topk(self._top_k)
                max_prob = topk_probs[0].item()
                max_idx = topk_indices[0].item()

                clip_scores.append(max_prob)
                clip_data.append({
                    "start_idx": i,
                    "end_idx": i + self._clip_length - 1,
                    "score": max_prob,
                    "class_idx": max_idx,
                    "top_k": [
                        {
                            "label": self._get_label(idx.item()),
                            "prob": prob.item(),
                        }
                        for prob, idx in zip(topk_probs, topk_indices)
                    ],
                    "start_pkt": clip_packets[0],
                    "end_pkt": clip_packets[-1],
                })

        # Emit events for clips exceeding the confidence gate & z-threshold
        for idx, cd in enumerate(clip_data):
            if cd["score"] < self._confidence_gate:
                continue
            z = self.compute_z_score(cd["score"], clip_scores)
            if z < self._z_threshold:
                continue

            label = self._get_label(cd["class_idx"])
            severity = self.severity_from_z(z)

            events.append(SAREvent(
                stream_name=self.name,
                start_frame=cd["start_pkt"].index,
                end_frame=cd["end_pkt"].index,
                start_time=cd["start_pkt"].timestamp,
                end_time=cd["end_pkt"].timestamp,
                confidence=round(cd["score"], 4),
                z_score=round(z, 4),
                label=f"Action: {label}",
                severity=severity,
                metadata={
                    "action_label": label,
                    "clip_index": idx,
                    "top_k": cd["top_k"],
                },
            ))

        log.info("Action classifier: %d events (from %d clips)", len(events), len(clip_data))
        return events

    def _get_label(self, class_idx: int) -> str:
        """Map class index to human-readable label."""
        # Try the checkpoint's index-to-label mapping first
        if self._idx_to_label and str(class_idx) in self._idx_to_label:
            return self._idx_to_label[str(class_idx)]
        if self._idx_to_label and class_idx in self._idx_to_label:
            return self._idx_to_label[class_idx]
        # Fall back to config labels
        if 0 <= class_idx < len(self._sar_labels):
            return self._sar_labels[class_idx]
        return f"class_{class_idx}"

    def _detect_stub(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Fallback stub when no trained model is available."""
        import random

        events: List[SAREvent] = []
        fps = self.global_config.get("video", {}).get("target_fps", 5)
        n_clips = max(1, (len(packets) - self._clip_length) // self._clip_stride + 1)

        clip_scores: List[float] = []
        clip_data: List[Dict[str, Any]] = []

        for i in range(0, len(packets) - self._clip_length + 1, self._clip_stride):
            clip = packets[i: i + self._clip_length]
            score = random.gauss(0.25, 0.12)
            if random.random() < 0.08:
                score = random.uniform(0.55, 0.95)
            score = max(0.0, min(1.0, score))
            clip_scores.append(score)
            clip_data.append({
                "score": score,
                "start_pkt": clip[0],
                "end_pkt": clip[-1],
            })

        for idx, cd in enumerate(clip_data):
            if cd["score"] < self._confidence_gate:
                continue
            z = self.compute_z_score(cd["score"], clip_scores)
            if z < self._z_threshold:
                continue
            label = random.choice(self._sar_labels) if self._sar_labels else "unknown"
            severity = self.severity_from_z(z)
            events.append(SAREvent(
                stream_name=self.name,
                start_frame=cd["start_pkt"].index,
                end_frame=cd["end_pkt"].index,
                start_time=cd["start_pkt"].timestamp,
                end_time=cd["end_pkt"].timestamp,
                confidence=round(cd["score"], 4),
                z_score=round(z, 4),
                label=f"Action: {label} (stub)",
                severity=severity,
                metadata={"stub": True, "clip_index": idx},
            ))

        return events

    def teardown(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().teardown()
