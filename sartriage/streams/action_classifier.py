"""
streams/action_classifier.py
=============================
Stream 1 — **Person-Centric** Action Classification (R3D-18).

Key improvement for SAR drone footage:
  Instead of classifying the whole frame (where people are tiny),
  this stream crops around EACH detected person from the YOLO
  front-end and classifies the crop.  This means a 20-pixel-tall
  person in a 720p frame gets upscaled to fill the 112×112 input,
  making posture/action far more discernible.

Pipeline:
  1. Sliding-window over frame packets (16-frame clips)
  2. For each clip, gather all YOLO person detections
  3. For each tracked person, crop a spatio-temporal tube
  4. Run per-person crop through R3D-18 → softmax
  5. Emit SAR events for high-confidence, high-z-score actions

Falls back to full-frame mode if no detections exist, and to
stub mode if no trained model is found.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent

log = logging.getLogger("sartriage.action")

# Expansion factor for person crops (add context around the bbox)
CROP_EXPAND = 0.35  # 35% padding around each side


class ActionClassifierStream(BaseStream):
    """Stream 1: person-centric clip-level action classification."""

    @property
    def name(self) -> str:
        return "action"

    def setup(self) -> None:
        """Load the trained R3D-18 model from checkpoint."""
        self._clip_length = self.global_config.get("video", {}).get("clip_length", 16)
        self._clip_stride = self.global_config.get("video", {}).get("clip_stride", 8)
        self._clip_size = self.config.get("clip_size", 112)
        self._confidence_gate = self.config.get("confidence_gate", 0.30)
        self._z_threshold = self.config.get("z_score_threshold", 2.0)
        self._sar_labels = self.config.get("sar_action_labels", [])
        self._top_k = self.config.get("top_k", 3)
        self._person_centric = self.config.get("person_centric", True)
        self._min_person_px = self.config.get("min_person_pixels", 12)

        # Normalisation (Kinetics-400 pre-training stats)
        self._mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, -1, 1, 1, 1)
        self._std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, -1, 1, 1, 1)

        # Load model
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cpu")
        self._stub_mode = False

        model_path = self.config.get("model_path", "")
        if model_path:
            full_path = Path(__file__).parent.parent / model_path
            if full_path.exists():
                self._load_model(full_path)
            else:
                log.warning("Model not found at %s — stub mode", full_path)
                self._stub_mode = True
        else:
            log.warning("No model_path — stub mode")
            self._stub_mode = True

        super().setup()

    def _load_model(self, path: Path) -> None:
        """Load the R3D-18 model from a checkpoint file."""
        import torchvision.models.video as video_models

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        log.info("Loading action classifier from %s (device=%s)", path.name, self._device)
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        num_classes = checkpoint.get("num_classes", len(self._sar_labels))
        self._idx_to_label = checkpoint.get("idx_to_label", {})
        saved_clip_size = checkpoint.get("clip_size", 112)

        # Auto-detect model architecture from checkpoint
        if saved_clip_size >= 224:
            # MViTv2-S (SOTA)
            log.info("Detected MViTv2-S checkpoint (clip_size=%d)", saved_clip_size)
            model = video_models.mvit_v2_s(weights=None)
            model.head[1] = nn.Linear(model.head[1].in_features, num_classes)
            self._clip_size = 224
        else:
            # R3D-18 (legacy)
            log.info("Detected R3D-18 checkpoint (clip_size=%d)", saved_clip_size)
            model = video_models.r3d_18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes),
            )
            self._clip_size = 112

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()
        self._model = model

        val_acc = checkpoint.get("val_acc", 0)
        epoch = checkpoint.get("epoch", "?")
        log.info("Model loaded — epoch %s, val_acc=%.1f%%", epoch, val_acc * 100)

    # ── Person-Centric Cropping ───────────────────────────────────────────

    def _get_person_crops(
        self,
        packets: Sequence[FramePacket],
    ) -> Dict[int, List[Tuple[np.ndarray, Dict]]]:
        """Group detections by approximate track ID and extract crops.

        Returns a dict mapping pseudo-track-ID → list of (crop, detection)
        across the clip frames.  If no detections exist, returns empty.
        """
        import cv2

        # Group detections by spatial proximity (simple IoU-based linking)
        tracks: Dict[int, List[Tuple[int, List[int], float]]] = {}
        next_id = 0

        for t, pkt in enumerate(packets):
            if not pkt.detections or pkt.image is None:
                continue
            for det in pkt.detections:
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                bw, bh = x2 - x1, y2 - y1
                if bw < self._min_person_px or bh < self._min_person_px:
                    continue

                # Try to match to existing track by centre proximity
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                matched = False
                for tid, frames_data in tracks.items():
                    if not frames_data:
                        continue
                    last_t, last_bbox, _ = frames_data[-1]
                    if t - last_t > 3:
                        continue
                    lx1, ly1, lx2, ly2 = last_bbox
                    lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                    dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                    max_dist = max(bw, bh) * 1.5
                    if dist < max_dist:
                        tracks[tid].append((t, [x1, y1, x2, y2], det.get("confidence", 0)))
                        matched = True
                        break

                if not matched:
                    tracks[next_id] = [(t, [x1, y1, x2, y2], det.get("confidence", 0))]
                    next_id += 1

        # Now extract crops for tracks that span at least 4 frames
        result: Dict[int, List[Tuple[np.ndarray, Dict]]] = {}

        for tid, frames_data in tracks.items():
            if len(frames_data) < 4:
                continue

            crops = []
            for t_idx, bbox, conf in frames_data:
                pkt = packets[t_idx]
                if pkt.image is None:
                    continue
                h, w = pkt.image.shape[:2]
                x1, y1, x2, y2 = bbox
                bw, bh = x2 - x1, y2 - y1

                # Expand crop with context padding
                pad_x = int(bw * CROP_EXPAND)
                pad_y = int(bh * CROP_EXPAND)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)

                crop = pkt.image[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # Resize to clip_size × clip_size
                crop = cv2.resize(crop, (self._clip_size, self._clip_size))
                if len(crop.shape) == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                crops.append((crop, {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "frame_idx": t_idx,
                }))

            if len(crops) >= 4:
                result[tid] = crops

        return result

    def _crops_to_tensor(self, crops: List[Tuple[np.ndarray, Dict]]) -> torch.Tensor:
        """Convert a list of person crops to a model-ready tensor.

        Samples clip_length frames evenly from the available crops.
        Returns shape: (1, 3, T, H, W)
        """
        n = len(crops)
        indices = np.linspace(0, n - 1, self._clip_length, dtype=int)

        frames = []
        for idx in indices:
            frames.append(crops[idx][0])

        clip = np.stack(frames, axis=0)
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        tensor = (tensor - self._mean) / self._std
        return tensor.to(self._device)

    def _preprocess_fullframe_clip(self, packets: Sequence[FramePacket]) -> torch.Tensor:
        """Fallback: full-frame preprocessing when no detections exist."""
        import cv2

        frames = []
        for pkt in packets:
            img = pkt.image
            if img is None:
                img = np.zeros((self._clip_size, self._clip_size, 3), dtype=np.uint8)
            if img.shape[0] != self._clip_size or img.shape[1] != self._clip_size:
                img = cv2.resize(img, (self._clip_size, self._clip_size))
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        clip = np.stack(frames, axis=0)
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        tensor = (tensor - self._mean) / self._std
        return tensor.to(self._device)

    # ── Main Detection ────────────────────────────────────────────────────

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Run person-centric action classification.

        Strategy:
          1. Slide a window over the frame sequence
          2. For each window, extract per-person spatio-temporal crops
          3. Classify each person's crop independently
          4. Emit events for SAR-relevant actions above thresholds
        """
        if not packets:
            return []
        if self._stub_mode:
            return self._detect_stub(packets)

        events: List[SAREvent] = []
        clip_scores: List[float] = []
        clip_data: List[Dict[str, Any]] = []

        n_clips = max(1, (len(packets) - self._clip_length) // self._clip_stride + 1)
        mode = "person-centric" if self._person_centric else "full-frame"
        log.info("Action classifier: %d clips, mode=%s, device=%s", n_clips, mode, self._device)

        with torch.no_grad():
            for i in range(0, len(packets) - self._clip_length + 1, self._clip_stride):
                clip_packets = packets[i: i + self._clip_length]

                if self._person_centric:
                    # ── Person-centric: classify each detected person ──
                    person_crops = self._get_person_crops(clip_packets)

                    if person_crops:
                        for track_id, crops in person_crops.items():
                            tensor = self._crops_to_tensor(crops)
                            logits = self._model(tensor)
                            probs = F.softmax(logits, dim=1).squeeze(0)

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
                                    {"label": self._get_label(idx.item()), "prob": prob.item()}
                                    for prob, idx in zip(topk_probs, topk_indices)
                                ],
                                "start_pkt": clip_packets[0],
                                "end_pkt": clip_packets[-1],
                                "track_id": track_id,
                                "person_bbox": crops[len(crops) // 2][1].get("bbox"),
                                "mode": "person_crop",
                            })
                    else:
                        # No detections → full-frame fallback
                        tensor = self._preprocess_fullframe_clip(clip_packets)
                        logits = self._model(tensor)
                        probs = F.softmax(logits, dim=1).squeeze(0)

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
                                {"label": self._get_label(idx.item()), "prob": prob.item()}
                                for prob, idx in zip(topk_probs, topk_indices)
                            ],
                            "start_pkt": clip_packets[0],
                            "end_pkt": clip_packets[-1],
                            "mode": "full_frame",
                        })
                else:
                    # Full-frame mode (legacy)
                    tensor = self._preprocess_fullframe_clip(clip_packets)
                    logits = self._model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze(0)

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
                            {"label": self._get_label(idx.item()), "prob": prob.item()}
                            for prob, idx in zip(topk_probs, topk_indices)
                        ],
                        "start_pkt": clip_packets[0],
                        "end_pkt": clip_packets[-1],
                        "mode": "full_frame",
                    })

        # ── SAR-critical actions: emit on confidence alone ────────────
        # These actions are the primary SAR targets — a person falling,
        # lying motionless, or crawling is always worth flagging.
        SAR_CRITICAL_ACTIONS = {"falling", "lying_down", "crawling", "stumbling"}
        SAR_HIGH_ACTIONS = {"waving_hand", "running"}  # distress signals
        CRITICAL_CONFIDENCE = 0.40  # lower gate for SAR-critical
        HIGH_CONFIDENCE = 0.50  # gate for high-priority actions

        # Log the predictions for debugging
        if clip_data:
            action_counts: Dict[str, int] = {}
            for cd in clip_data:
                lbl = self._get_label(cd["class_idx"])
                action_counts[lbl] = action_counts.get(lbl, 0) + 1
            log.info("Action predictions: %s", action_counts)

        for idx, cd in enumerate(clip_data):
            label = self._get_label(cd["class_idx"])
            score = cd["score"]
            z = self.compute_z_score(score, clip_scores)

            emit = False

            if label in SAR_CRITICAL_ACTIONS and score >= CRITICAL_CONFIDENCE:
                # SAR-critical: always emit above confidence gate
                emit = True
                if z < 1.0:
                    z = max(z, 1.5)  # minimum z for severity mapping
            elif label in SAR_HIGH_ACTIONS and score >= HIGH_CONFIDENCE:
                # High-priority: emit with moderate confidence
                emit = True
                if z < 1.0:
                    z = max(z, 1.0)
            elif score >= self._confidence_gate and z >= self._z_threshold:
                # Standard path: both confidence AND z-score must pass
                emit = True

            if not emit:
                continue

            severity = self.severity_from_z(z)
            # Boost severity for SAR-critical detections
            if label in SAR_CRITICAL_ACTIONS and severity.value in ("low", "medium"):
                severity = EventSeverity.HIGH

            # Build metadata
            meta: Dict[str, Any] = {
                "action_label": label,
                "clip_index": idx,
                "top_k": cd["top_k"],
                "mode": cd.get("mode", "unknown"),
                "sar_critical": label in SAR_CRITICAL_ACTIONS,
                "sar_high": label in SAR_HIGH_ACTIONS,
            }
            if "track_id" in cd:
                meta["track_id"] = cd["track_id"]
            if "person_bbox" in cd:
                meta["person_bbox"] = cd["person_bbox"]

            events.append(SAREvent(
                stream_name=self.name,
                start_frame=cd["start_pkt"].index,
                end_frame=cd["end_pkt"].index,
                start_time=cd["start_pkt"].timestamp,
                end_time=cd["end_pkt"].timestamp,
                confidence=round(score, 4),
                z_score=round(z, 4),
                label=f"Action: {label}",
                severity=severity,
                metadata=meta,
            ))

        person_clips = sum(1 for c in clip_data if c.get("mode") == "person_crop")
        full_clips = sum(1 for c in clip_data if c.get("mode") == "full_frame")
        log.info(
            "Action classifier: %d events (%d person-crops, %d full-frame clips)",
            len(events), person_clips, full_clips,
        )
        return events

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_label(self, class_idx: int) -> str:
        if self._idx_to_label and str(class_idx) in self._idx_to_label:
            return self._idx_to_label[str(class_idx)]
        if self._idx_to_label and class_idx in self._idx_to_label:
            return self._idx_to_label[class_idx]
        if 0 <= class_idx < len(self._sar_labels):
            return self._sar_labels[class_idx]
        return f"class_{class_idx}"

    def _detect_stub(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Fallback stub when no trained model is available."""
        import random
        events: List[SAREvent] = []
        clip_scores: List[float] = []
        clip_data: List[Dict[str, Any]] = []

        for i in range(0, len(packets) - self._clip_length + 1, self._clip_stride):
            clip = packets[i: i + self._clip_length]
            score = random.gauss(0.25, 0.12)
            if random.random() < 0.08:
                score = random.uniform(0.55, 0.95)
            score = max(0.0, min(1.0, score))
            clip_scores.append(score)
            clip_data.append({"score": score, "start_pkt": clip[0], "end_pkt": clip[-1]})

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
        if self._model is not None:
            del self._model
            self._model = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        super().teardown()
