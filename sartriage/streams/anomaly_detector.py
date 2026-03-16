"""
streams/anomaly_detector.py
============================
Stream 5 — Self-Supervised Anomaly Detection.

This is the novel contribution: instead of classifying *known* actions,
this stream learns what "normal" drone footage looks like and flags
anything statistically abnormal.

A person lying motionless while everyone else walks, someone crawling
in an unusual location, a sudden cluster of track losses — these are
*novel* emergency behaviours that a supervised classifier trained on
Kinetics-400 would miss.

Method:
  1. Extract per-clip feature vectors from the MViTv2-S backbone
     (frozen, no fine-tuning needed)
  2. Build a Gaussian distribution (mean + covariance) of features
     from all clips in the video (the video IS the "normal" baseline)
  3. Compute Mahalanobis distance for each clip
  4. Flag clips with high Mahalanobis distance as anomalies

This is an *unsupervised* approach — no labels needed. It catches
any behaviour that deviates from the video's own baseline, making
it robust to novel emergency scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent

log = logging.getLogger("sartriage.anomaly")


class AnomalyDetectorStream(BaseStream):
    """Stream 5: self-supervised anomaly detection via feature distribution."""

    @property
    def name(self) -> str:
        return "anomaly"

    def setup(self) -> None:
        """Load the feature extractor (MViTv2-S backbone, frozen)."""
        self._clip_length = self.global_config.get("video", {}).get("clip_length", 16)
        self._clip_stride = self.global_config.get("video", {}).get("clip_stride", 8)
        self._clip_size = self.config.get("clip_size", 224)
        self._mahal_threshold = self.config.get("mahalanobis_threshold", 3.0)
        self._min_clips_for_baseline = self.config.get("min_clips_for_baseline", 10)
        self._confidence_gate = self.config.get("confidence_gate", 0.40)
        self._z_threshold = self.config.get("z_score_threshold", 2.0)

        # Normalisation
        self._mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, -1, 1, 1, 1)
        self._std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, -1, 1, 1, 1)

        self._feature_extractor: Optional[nn.Module] = None
        self._device = torch.device("cpu")
        self._stub_mode = False

        self._load_feature_extractor()
        super().setup()

    def _load_feature_extractor(self) -> None:
        """Load MViTv2-S as a frozen feature extractor."""
        try:
            import torchvision.models.video as video_models

            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")

            # Load pre-trained MViTv2-S (no fine-tuning needed)
            weights = video_models.MViT_V2_S_Weights.KINETICS400_V1
            model = video_models.mvit_v2_s(weights=weights)
            model.eval()

            # Remove the classification head — we only want features
            # MViTv2 head is Sequential(Dropout, Linear)
            # We'll hook into the norm layer before the head
            self._feature_dim = model.head[1].in_features  # 768

            # Replace head with identity to get features
            model.head = nn.Identity()
            model.to(self._device)

            self._feature_extractor = model
            log.info(
                "Anomaly detector: MViTv2-S feature extractor loaded (dim=%d, device=%s)",
                self._feature_dim, self._device,
            )
        except Exception as exc:
            log.warning("Anomaly detector: cannot load MViTv2-S (%s) — stub mode", exc)
            self._stub_mode = True
            self._feature_dim = 768

    def _preprocess_clip(self, packets: Sequence[FramePacket]) -> torch.Tensor:
        """Pre-process a clip for the feature extractor."""
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

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Detect anomalous clips using feature distribution analysis.

        Steps:
          1. Extract features for all clips (sliding window)
          2. Fit Gaussian to the feature distribution (mean + covariance)
          3. Compute Mahalanobis distance for each clip
          4. Flag clips exceeding the threshold
        """
        if not packets:
            return []

        if self._stub_mode:
            return self._detect_stub(packets)

        n_possible = len(packets) - self._clip_length + 1
        if n_possible < self._min_clips_for_baseline:
            log.info("Anomaly detector: too few clips (%d < %d) for baseline",
                     n_possible, self._min_clips_for_baseline)
            return []

        # ── 1. Extract features ──────────────────────────────────────
        features: List[np.ndarray] = []
        clip_meta: List[Dict[str, Any]] = []

        log.info("Anomaly detector: extracting features from %d clips",
                 (n_possible + self._clip_stride - 1) // self._clip_stride)

        with torch.no_grad():
            for i in range(0, n_possible, self._clip_stride):
                clip_packets = packets[i: i + self._clip_length]
                tensor = self._preprocess_clip(clip_packets)

                feat = self._feature_extractor(tensor)  # (1, feature_dim)
                if feat.dim() > 2:
                    feat = feat.mean(dim=list(range(1, feat.dim() - 1)))  # global pool
                feat = feat.squeeze(0).cpu().numpy()  # (feature_dim,)
                features.append(feat)

                clip_meta.append({
                    "start_idx": i,
                    "end_idx": i + self._clip_length - 1,
                    "start_pkt": clip_packets[0],
                    "end_pkt": clip_packets[-1],
                })

        if len(features) < self._min_clips_for_baseline:
            log.info("Anomaly detector: only %d clips extracted, skipping", len(features))
            return []

        # ── 2. Fit Gaussian baseline ─────────────────────────────────
        feat_matrix = np.stack(features, axis=0)  # (N, D)
        mean = feat_matrix.mean(axis=0)  # (D,)
        centered = feat_matrix - mean
        cov = np.cov(centered.T) + np.eye(self._feature_dim) * 1e-6  # regularised

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            log.warning("Anomaly detector: covariance matrix is singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(cov)

        # ── 3. Compute Mahalanobis distances ─────────────────────────
        distances = []
        for feat in features:
            diff = feat - mean
            mahal = float(np.sqrt(diff @ cov_inv @ diff))
            distances.append(mahal)

        distances = np.array(distances)
        dist_mean = distances.mean()
        dist_std = distances.std() + 1e-8

        log.info("Anomaly detector: Mahalanobis stats — mean=%.2f, std=%.2f, max=%.2f",
                 dist_mean, dist_std, distances.max())

        # ── 4. Flag anomalies ────────────────────────────────────────
        events: List[SAREvent] = []

        for idx, (dist, meta) in enumerate(zip(distances, clip_meta)):
            # Z-score of the Mahalanobis distance
            z = (dist - dist_mean) / dist_std

            if z < self._z_threshold:
                continue

            # Anomaly confidence: sigmoid of z-score
            confidence = float(1.0 / (1.0 + np.exp(-0.5 * (z - self._z_threshold))))

            if confidence < self._confidence_gate:
                continue

            severity = self.severity_from_z(z)

            events.append(SAREvent(
                stream_name=self.name,
                start_frame=meta["start_pkt"].index,
                end_frame=meta["end_pkt"].index,
                start_time=meta["start_pkt"].timestamp,
                end_time=meta["end_pkt"].timestamp,
                confidence=round(confidence, 4),
                z_score=round(z, 4),
                label=f"Anomalous behaviour (Mahal={dist:.1f})",
                severity=severity,
                metadata={
                    "mahalanobis_distance": round(float(dist), 2),
                    "baseline_mean": round(float(dist_mean), 2),
                    "baseline_std": round(float(dist_std), 2),
                    "anomaly_z_score": round(float(z), 2),
                    "detection_method": "self_supervised_mahalanobis",
                    "feature_dim": self._feature_dim,
                },
            ))

        log.info("Anomaly detector: %d anomalous clips from %d total", len(events), len(features))
        return events

    def _detect_stub(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Stub when MViTv2-S is unavailable."""
        import random

        events: List[SAREvent] = []
        n_clips = max(1, (len(packets) - self._clip_length) // self._clip_stride + 1)

        # Simulate: ~5% of clips are anomalous
        for i in range(0, n_clips):
            if random.random() > 0.05:
                continue

            clip_start = i * self._clip_stride
            clip_end = min(clip_start + self._clip_length - 1, len(packets) - 1)

            if clip_start >= len(packets) or clip_end >= len(packets):
                continue

            dist = random.uniform(4.0, 8.0)
            z = random.uniform(2.5, 5.0)
            confidence = 1.0 / (1.0 + np.exp(-0.5 * (z - 2.0)))

            events.append(SAREvent(
                stream_name=self.name,
                start_frame=packets[clip_start].index,
                end_frame=packets[clip_end].index,
                start_time=packets[clip_start].timestamp,
                end_time=packets[clip_end].timestamp,
                confidence=round(confidence, 4),
                z_score=round(z, 4),
                label=f"Anomalous behaviour (Mahal={dist:.1f}, stub)",
                severity=self.severity_from_z(z),
                metadata={
                    "stub": True,
                    "mahalanobis_distance": round(dist, 2),
                    "detection_method": "stub",
                },
            ))

        return events

    def teardown(self) -> None:
        if self._feature_extractor is not None:
            del self._feature_extractor
            self._feature_extractor = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        super().teardown()
