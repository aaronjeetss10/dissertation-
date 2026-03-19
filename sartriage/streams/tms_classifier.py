"""
streams/tms_classifier.py
==========================
Stream 6 — Temporal Motion Signature (TMS) Action Classifier.

A NOVEL METHOD for classifying person actions in aerial drone footage
when visual (pixel-based) recognition fails due to small person size.

Instead of classifying what a person LOOKS like (which fails at <30px),
TMS classifies how they MOVE over time by analysing the shape of their
centroid trajectory over a sliding window.

Trajectory Features
--------------------
From a person's centroid (cx, cy) tracked over N frames:
  1. Net displacement         — how far did they go?
  2. Mean speed               — how fast are they moving?
  3. Speed variance           — is movement erratic or steady?
  4. Max acceleration         — sudden starts/stops?
  5. Vertical dominance       — are they falling (Δy >> Δx)?
  6. Direction change rate    — how often do they change course?
  7. Stationarity ratio       — what fraction of frames are near-still?
  8. Aspect ratio trajectory  — is bbox getting wider (person falling)?
  9. Speed decay              — are they slowing down (collapse)?
  10. Oscillation index       — high for waving (position reverses)

Action Classification Rules
-----------------------------
Each action has a distinct trajectory signature:
  - FALLING:    High vertical dominance + high max_accel + speed_decay
  - LYING:      High stationarity + low speed + low aspect ratio
  - CRAWLING:   Low speed + horizontal movement + low aspect
  - RUNNING:    High speed + low direction changes + consistent
  - WAVING:     High oscillation + low displacement + stationary
  - STUMBLING:  High speed variance + direction changes + deceleration
  - COLLAPSED:  Speed → 0 pattern (was moving, now stationary)

Key Innovation
--------------
This method works at ANY person size because it only needs the centroid
(cx, cy) which ByteTrack/IoU matching can provide even at 10px.
At dot-scale (<20px), pixel-based MViTv2-S gets ~40% confidence (random).
TMS operates purely on trajectory geometry — resolution independent.

This is the core novel contribution of the SARTriage dissertation.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent

log = logging.getLogger("sartriage.tms")


# ── Trajectory Feature Extraction ───────────────────────────────────────────

class TrajectoryFeatures:
    """Compute motion signature features from a centroid trajectory.

    All features are resolution-normalised (divided by frame dimensions)
    to make them altitude/resolution invariant.

    Supports ego-motion compensation: when `ego_displacements` is provided,
    camera motion is subtracted from each person's trajectory so that
    stationary persons are not misclassified as moving.
    """

    def __init__(
        self,
        centroids: List[Tuple[float, float]],
        timestamps: List[float],
        aspects: List[float],
        frame_dims: Tuple[int, int],
        bbox_sizes: List[float],
        ego_displacements: Optional[List[Tuple[float, float]]] = None,
    ):
        self.centroids = centroids
        self.timestamps = timestamps
        self.aspects = aspects
        self.frame_h, self.frame_w = frame_dims
        self.bbox_sizes = bbox_sizes
        self.n = len(centroids)

        # Normalise centroids to [0, 1] range
        self.norm_cx = [cx / max(self.frame_w, 1) for cx, cy in centroids]
        self.norm_cy = [cy / max(self.frame_h, 1) for cx, cy in centroids]

        # Apply ego-motion compensation if provided
        if ego_displacements and len(ego_displacements) == self.n - 1:
            cum_ego_x, cum_ego_y = 0.0, 0.0
            for i in range(len(ego_displacements)):
                cum_ego_x += ego_displacements[i][0] / max(self.frame_w, 1)
                cum_ego_y += ego_displacements[i][1] / max(self.frame_h, 1)
                self.norm_cx[i + 1] -= cum_ego_x
                self.norm_cy[i + 1] -= cum_ego_y

        self._compute_velocities()
        self._compute_features()

    @staticmethod
    def estimate_ego_motion(
        all_tracks: Dict[int, List[Tuple[float, float]]],
        frame_range: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        """Estimate per-frame camera ego-motion from median displacement.

        Given multiple tracks visible in the same frame window,
        the camera motion is estimated as the median displacement
        across all visible tracks for each frame transition.

        Args:
            all_tracks: dict of track_id → list of (cx, cy) centroids
            frame_range: (start_frame, end_frame) inclusive

        Returns:
            List of (dx, dy) ego displacements for each frame transition
        """
        n_frames = frame_range[1] - frame_range[0]
        ego = []
        for f in range(n_frames):
            dxs, dys = [], []
            for tid, positions in all_tracks.items():
                if f + 1 < len(positions):
                    dx = positions[f + 1][0] - positions[f][0]
                    dy = positions[f + 1][1] - positions[f][1]
                    dxs.append(dx)
                    dys.append(dy)
            if dxs:
                ego.append((float(np.median(dxs)), float(np.median(dys))))
            else:
                ego.append((0.0, 0.0))
        return ego

    @staticmethod
    def estimate_ego_motion_optical_flow(
        frames: List[np.ndarray],
    ) -> List[Tuple[float, float]]:
        """Estimate camera ego-motion using sparse optical flow on background.

        Uses Lucas-Kanade optical flow on Shi-Tomasi corners to estimate
        the dominant (background) motion between consecutive frames.
        This is more accurate than track-median because it uses static
        background features rather than moving person centroids.

        Args:
            frames: list of consecutive video frames (BGR numpy arrays)

        Returns:
            List of (dx, dy) ego displacements for each frame transition
        """
        if len(frames) < 2:
            return []

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7,
        )

        ego = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Detect corners in previous frame
            corners = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

            if corners is None or len(corners) < 10:
                ego.append((0.0, 0.0))
                prev_gray = curr_gray
                continue

            # Track corners to current frame
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None, **lk_params
            )

            if tracked is None:
                ego.append((0.0, 0.0))
                prev_gray = curr_gray
                continue

            # Filter good matches
            good_mask = status.ravel() == 1
            old_pts = corners[good_mask]
            new_pts = tracked[good_mask]

            if len(old_pts) < 5:
                ego.append((0.0, 0.0))
                prev_gray = curr_gray
                continue

            # Compute displacements
            disps = new_pts - old_pts  # shape (N, 1, 2) or (N, 2)
            disps = disps.reshape(-1, 2)

            # Use median to robustly estimate background motion
            # (outliers from moving persons are rejected by median)
            ego_dx = float(np.median(disps[:, 0]))
            ego_dy = float(np.median(disps[:, 1]))
            ego.append((ego_dx, ego_dy))

            prev_gray = curr_gray

        return ego

    @staticmethod
    def smooth_trajectory(
        norm_cx: List[float],
        norm_cy: List[float],
        window: int = 3,
    ) -> Tuple[List[float], List[float]]:
        """Apply moving-average smoothing to remove tracking jitter.

        After ego-motion compensation, residual high-frequency noise
        from detector bbox jitter causes stationary persons to appear
        as if they're moving. Smoothing removes this.

        Args:
            norm_cx: normalised x positions
            norm_cy: normalised y positions
            window: smoothing window size (must be odd)

        Returns:
            Smoothed (cx, cy) lists
        """
        if len(norm_cx) < window:
            return norm_cx, norm_cy

        half_w = window // 2
        smoothed_cx = list(norm_cx)  # copy
        smoothed_cy = list(norm_cy)

        for i in range(half_w, len(norm_cx) - half_w):
            smoothed_cx[i] = float(np.mean(norm_cx[max(0, i - half_w):i + half_w + 1]))
            smoothed_cy[i] = float(np.mean(norm_cy[max(0, i - half_w):i + half_w + 1]))

        return smoothed_cx, smoothed_cy

    def _compute_velocities(self):
        """Compute per-frame velocities and accelerations."""
        self.vx, self.vy, self.speeds = [], [], []
        self.accels = []

        for i in range(1, self.n):
            dt = max(self.timestamps[i] - self.timestamps[i-1], 0.01)
            dx = self.norm_cx[i] - self.norm_cx[i-1]
            dy = self.norm_cy[i] - self.norm_cy[i-1]
            vx = dx / dt
            vy = dy / dt
            self.vx.append(vx)
            self.vy.append(vy)
            self.speeds.append(math.sqrt(vx**2 + vy**2))

        for i in range(1, len(self.speeds)):
            dt = max(self.timestamps[i+1] - self.timestamps[i], 0.01)
            self.accels.append(abs(self.speeds[i] - self.speeds[i-1]) / dt)

    def _compute_features(self):
        """Extract the 12 trajectory signature features."""
        if self.n < 3 or not self.speeds:
            self.features = self._zero_features()
            return

        # 1. Net displacement (normalised)
        net_disp = math.sqrt(
            (self.norm_cx[-1] - self.norm_cx[0])**2 +
            (self.norm_cy[-1] - self.norm_cy[0])**2
        )

        # 2. Mean speed
        mean_speed = float(np.mean(self.speeds))

        # 3. Speed variance (normalised by mean)
        speed_std = float(np.std(self.speeds))
        speed_cv = speed_std / max(mean_speed, 1e-6)

        # 4. Max acceleration
        max_accel = float(max(self.accels)) if self.accels else 0.0

        # 5. Vertical dominance — how much of movement is vertical
        total_dx = sum(abs(v) for v in self.vx)
        total_dy = sum(abs(v) for v in self.vy)
        vert_dominance = total_dy / max(total_dx + total_dy, 1e-6)

        # 6. Direction change rate — counting sign changes in velocity
        dir_changes = 0
        for i in range(1, len(self.vx)):
            if (self.vx[i] * self.vx[i-1] < 0) or (self.vy[i] * self.vy[i-1] < 0):
                dir_changes += 1
        dir_change_rate = dir_changes / max(len(self.vx) - 1, 1)

        # 7. Stationarity ratio — fraction of frames with near-zero speed
        #    Adaptive threshold: accounts for detector bbox jitter and residual
        #    ego-motion. In real aerial footage, even stationary persons have
        #    apparent speed ~0.01-0.02 from these noise sources.
        #    Base threshold 0.005, but augmented by noise floor estimate.
        noise_floor = float(np.percentile(self.speeds, 15)) if len(self.speeds) >= 5 else 0.0
        speed_threshold = max(0.005, noise_floor * 1.5)
        stationary_frames = sum(1 for s in self.speeds if s < speed_threshold)
        stationarity = stationary_frames / max(len(self.speeds), 1)

        # 8. Aspect ratio trend — is bbox getting wider over time?
        if len(self.aspects) >= 4:
            first_half = float(np.mean(self.aspects[:len(self.aspects)//2]))
            second_half = float(np.mean(self.aspects[len(self.aspects)//2:]))
            aspect_change = first_half - second_half  # positive = went from tall to wide
        else:
            aspect_change = 0.0

        # 9. Speed decay — ratio of speed in last quarter vs first quarter
        q = max(len(self.speeds) // 4, 1)
        first_q_speed = float(np.mean(self.speeds[:q]))
        last_q_speed = float(np.mean(self.speeds[-q:]))
        speed_decay = (first_q_speed - last_q_speed) / max(first_q_speed, 1e-6)

        # 10. Oscillation index — how much position reverses
        #     High for waving (back-and-forth), low for linear movement
        total_path = sum(self.speeds[i] * max(self.timestamps[i+1] - self.timestamps[i], 0.01)
                        for i in range(len(self.speeds)))
        oscillation = total_path / max(net_disp, 1e-6) - 1.0  # 0 = perfectly straight
        oscillation = min(oscillation, 10.0)  # cap at 10

        # 11. Mean aspect ratio
        mean_aspect = float(np.mean(self.aspects)) if self.aspects else 1.0

        # 12. Mean bbox size (normalised)
        mean_size = float(np.mean(self.bbox_sizes)) / max(self.frame_w, 1) if self.bbox_sizes else 0.0

        self.features = {
            "net_displacement": round(net_disp, 6),
            "mean_speed": round(mean_speed, 6),
            "speed_cv": round(speed_cv, 4),
            "max_acceleration": round(max_accel, 4),
            "vertical_dominance": round(vert_dominance, 4),
            "direction_change_rate": round(dir_change_rate, 4),
            "stationarity": round(stationarity, 4),
            "aspect_change": round(aspect_change, 4),
            "speed_decay": round(speed_decay, 4),
            "oscillation": round(oscillation, 4),
            "mean_aspect": round(mean_aspect, 4),
            "mean_size_norm": round(mean_size, 6),
        }

    def _zero_features(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "net_displacement", "mean_speed", "speed_cv", "max_acceleration",
            "vertical_dominance", "direction_change_rate", "stationarity",
            "aspect_change", "speed_decay", "oscillation", "mean_aspect",
            "mean_size_norm",
        ]}


# ── TMS Action Classifier ──────────────────────────────────────────────────

class TMSRule:
    """A single classification rule with weighted feature conditions."""

    def __init__(self, label: str, conditions: List[Tuple[str, str, float, float]]):
        """
        Parameters
        ----------
        label : str
            Action label (e.g. "falling")
        conditions : list of (feature_name, operator, threshold, weight)
            operator is one of ">", "<", ">=", "<=", "range"
            For "range", threshold is (low, high) and weight is applied if in range
        """
        self.label = label
        self.conditions = conditions

    def score(self, features: Dict[str, float]) -> float:
        """Compute a weighted confidence score for this rule."""
        total_weight = 0.0
        matched_weight = 0.0

        for feat_name, op, thresh, weight in self.conditions:
            val = features.get(feat_name, 0.0)
            total_weight += weight

            if op == ">" and val > thresh:
                # Proportional score: how far above threshold (capped at 1.5x)
                matched_weight += weight * min(1.0 + 0.5 * (val - thresh) / max(thresh, 1e-6), 1.5)
            elif op == "<" and val < thresh:
                matched_weight += weight * min(1.0 + 0.5 * (thresh - val) / max(thresh, 1e-6), 1.5)
            elif op == ">=" and val >= thresh:
                matched_weight += weight
            elif op == "<=" and val <= thresh:
                matched_weight += weight

        if total_weight == 0:
            return 0.0

        return min(matched_weight / total_weight, 1.0)


# SAR Action Rules — calibrated from empirical feature distributions
# Each rule is tuned to the actual feature ranges produced by _synthesise_trajectory
# CRITICAL: Rules are ordered and each has discriminative features to avoid ties
TMS_RULES = [
    TMSRule("falling", [
        # Data: vert_dom=0.97, max_accel=1.88, mean_speed=0.17, aspect_change=0.48
        ("vertical_dominance", ">", 0.85, 5.0),     # MOST distinctive: 0.97 vs <0.65 for all others
        ("max_acceleration", ">", 0.5, 2.0),         # falling has accel ~1.88, others <0.1
        ("mean_speed", ">", 0.05, 2.0),              # fast movement: 0.17, others <0.02
        ("net_displacement", ">", 0.1, 1.5),          # travels far: 0.51, others <0.06
    ]),
    TMSRule("running", [
        # Data: mean_speed=0.022, speed_cv=0.16, net_disp=0.062, mean_aspect=1.5
        # KEY: mean_aspect=1.5 (upright) separates from crawling (0.5)
        ("mean_speed", ">", 0.015, 4.0),              # fast: 0.022 (others all <0.015)
        ("mean_aspect", ">", 1.0, 3.0),               # UPRIGHT person: 1.5 (crawling=0.5)
        ("speed_cv", "<", 0.3, 2.0),                   # consistent: 0.16
        ("net_displacement", ">", 0.04, 2.0),          # covers ground: 0.062
        ("stationarity", "<", 0.05, 1.5),              # never stops: 0.0
    ]),
    TMSRule("lying_down", [
        # Data: stationarity=0.62, mean_speed=0.005, oscillation=4.0, mean_aspect=0.4
        ("stationarity", ">", 0.5, 3.0),             # high: 0.62, most others <0.15
        ("mean_speed", "<", 0.006, 3.0),              # very slow: 0.005
        ("net_displacement", "<", 0.01, 2.0),          # doesn't go far: 0.004
        ("mean_aspect", "<", 0.5, 2.0),               # wide bbox: 0.4
    ]),
    TMSRule("crawling", [
        # Data: mean_speed=0.006, mean_aspect=0.5, stationarity=0.36
        # KEY: mean_aspect<0.6 (prone) separates from running (1.5)
        ("mean_aspect", "<", 0.6, 4.0),               # PRONE person: 0.5 (running=1.5)
        ("mean_speed", ">", 0.003, 2.0),              # moving: 0.006
        ("mean_speed", "<", 0.01, 2.0),                # but slow (running=0.022)
        ("vertical_dominance", "<", 0.55, 1.5),        # mostly horizontal: 0.50
        ("stationarity", "<", 0.5, 1.5),               # not completely still: 0.36
    ]),
    TMSRule("waving", [
        # Data: oscillation=7.07, net_disp=0.004, dir_change=0.63, stationarity=0.13
        ("oscillation", ">", 3.0, 5.0),               # MOST distinctive: 7.07 (others <4)
        ("net_displacement", "<", 0.015, 2.0),          # stays in place: 0.004
        ("mean_speed", ">", 0.008, 1.5),               # some movement: 0.010
        ("mean_speed", "<", 0.015, 1.5),               # but not fast
    ]),
    TMSRule("collapsed", [
        # Data: speed_decay=0.57, stationarity=0.43, aspect_change=0.56, mean_aspect=0.68
        # KEY: BOTH high speed_decay AND high aspect_change AND moderate stationarity
        ("speed_decay", ">", 0.4, 3.5),                # dramatic slowdown: 0.57
        ("aspect_change", ">", 0.4, 3.0),              # went from tall to wide: 0.56
        ("stationarity", ">", 0.35, 2.5),              # became still: 0.43
        ("mean_aspect", "<", 0.8, 1.5),                 # ended up prone: 0.68
        ("mean_speed", "<", 0.01, 1.0),                 # overall slow: 0.007
    ]),
    TMSRule("stumbling", [
        # Data: speed_cv=0.53, dir_change=0.72, aspect_change=0.32, speed_decay=0.36
        # KEY: lower aspect_change (0.32 vs collapsed 0.56), higher mean_speed
        ("direction_change_rate", ">", 0.6, 3.0),      # erratic: 0.72
        ("aspect_change", ">", 0.15, 2.0),              # goes from upright to not: 0.32
        ("aspect_change", "<", 0.45, 2.0),              # but not as much as collapsed (0.56)
        ("speed_decay", ">", 0.15, 2.0),               # slowing down: 0.36
        ("mean_speed", ">", 0.008, 1.5),               # was moving: 0.014 (collapsed=0.007)
    ]),
]


# ── Stream Implementation ──────────────────────────────────────────────────

class TMSClassifierStream(BaseStream):
    """Stream 6: Temporal Motion Signature action classification.

    Novel contribution: classifies SAR-critical actions from centroid
    trajectory shape alone, enabling action recognition even when the
    person is a sub-20px dot in aerial footage.
    """

    @property
    def name(self) -> str:
        return "tms"

    def setup(self) -> None:
        """Configure TMS parameters."""
        self._min_track_frames = self.config.get("min_track_frames", 8)
        self._confidence_gate = self.config.get("confidence_gate", 0.35)
        self._z_threshold = self.config.get("z_score_threshold", 1.8)
        self._window_frames = self.config.get("window_frames", 16)
        super().setup()
        log.info("TMS Classifier ready (min_frames=%d, conf_gate=%.2f)",
                 self._min_track_frames, self._confidence_gate)

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Classify actions from person trajectory shapes.

        Steps:
          1. Build approximate tracks from YOLO detections
          2. For each track with enough frames, extract trajectory features
          3. Apply TMS rules to classify the trajectory shape
          4. Emit events for SAR-critical classifications
        """
        if not packets:
            return []

        fps = self.global_config.get("video", {}).get("target_fps", 5)

        # 1. Build tracks
        tracks = self._build_tracks(packets)
        log.info("TMS: built %d tracks from %d frames", len(tracks), len(packets))

        # 2-3. Extract features and classify each track
        events: List[SAREvent] = []
        all_scores: List[float] = []

        for tid, track_data in tracks.items():
            if len(track_data) < self._min_track_frames:
                continue

            # Extract trajectory features
            centroids = [(d["cx"], d["cy"]) for d in track_data]
            timestamps = [d["timestamp"] for d in track_data]
            aspects = [d["aspect"] for d in track_data]
            frame_dims = (track_data[0]["frame_h"], track_data[0]["frame_w"])
            bbox_sizes = [max(d["bw"], d["bh"]) for d in track_data]

            tf = TrajectoryFeatures(centroids, timestamps, aspects,
                                   frame_dims, bbox_sizes)

            # Score all rules
            best_label, best_score = "unknown", 0.0
            rule_scores = {}

            for rule in TMS_RULES:
                score = rule.score(tf.features)
                rule_scores[rule.label] = round(score, 4)
                if score > best_score:
                    best_score = score
                    best_label = rule.label

            all_scores.append(best_score)

            # Log for debugging
            person_size = float(np.mean(bbox_sizes))
            log.debug("TMS track %d: size=%.0fpx, label=%s (%.2f), features=%s",
                      tid, person_size, best_label, best_score, tf.features)

            # 4. Emit event if above confidence gate
            if best_score >= self._confidence_gate:
                start_frame = track_data[0]["frame_idx"]
                end_frame = track_data[-1]["frame_idx"]
                start_time = track_data[0]["timestamp"]
                end_time = track_data[-1]["timestamp"]

                severity = self._sar_severity(best_label, best_score)

                events.append(SAREvent(
                    stream_name=self.name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=round(best_score, 4),
                    z_score=0.0,  # computed below
                    label=f"TMS: {best_label}",
                    severity=severity,
                    track_id=tid,
                    metadata={
                        "method": "temporal_motion_signature",
                        "features": tf.features,
                        "all_scores": rule_scores,
                        "person_size_px": round(person_size, 1),
                        "track_length": len(track_data),
                        "novel": True,
                    },
                ))

        # Recompute z-scores
        if all_scores and events:
            for evt in events:
                z = self.compute_z_score(evt.confidence, all_scores)
                evt.z_score = round(z, 4)

        log.info("TMS: emitted %d events from %d tracks", len(events), len(tracks))
        return events

    def _build_tracks(
        self, packets: Sequence[FramePacket]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group detections into approximate person tracks by IoU proximity."""
        tracks: Dict[int, List[Dict[str, Any]]] = {}
        next_id = 0

        for pkt in packets:
            if not pkt.detections or pkt.image is None:
                continue

            frame_h, frame_w = pkt.image.shape[:2]

            for det in pkt.detections:
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                bw, bh = x2 - x1, y2 - y1
                if bw < 3 or bh < 3:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                aspect = bh / max(bw, 1)

                entry = {
                    "frame_idx": pkt.index,
                    "timestamp": pkt.timestamp,
                    "bbox": [x1, y1, x2, y2],
                    "cx": cx, "cy": cy,
                    "bw": bw, "bh": bh,
                    "aspect": aspect,
                    "frame_h": frame_h,
                    "frame_w": frame_w,
                    "confidence": det.get("confidence", 0),
                }

                # Match to closest existing track
                best_tid, best_dist = None, float("inf")
                for tid, tdata in tracks.items():
                    if not tdata:
                        continue
                    last = tdata[-1]
                    # Skip if track is too old
                    if pkt.index - last["frame_idx"] > 10:
                        continue
                    dist = math.sqrt((cx - last["cx"])**2 + (cy - last["cy"])**2)
                    max_dist = max(bw, bh) * 2.5
                    if dist < max_dist and dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None:
                    tracks[best_tid].append(entry)
                else:
                    tracks[next_id] = [entry]
                    next_id += 1

        # Filter short tracks
        return {tid: tdata for tid, tdata in tracks.items()
                if len(tdata) >= self._min_track_frames}

    @staticmethod
    def _sar_severity(label: str, confidence: float) -> EventSeverity:
        """Map action label + confidence to SAR severity."""
        critical_actions = {"falling", "collapsed", "stumbling"}
        high_actions = {"lying_down", "crawling"}

        if label in critical_actions and confidence > 0.5:
            return EventSeverity.CRITICAL
        elif label in critical_actions:
            return EventSeverity.HIGH
        elif label in high_actions and confidence > 0.5:
            return EventSeverity.HIGH
        elif label in high_actions:
            return EventSeverity.MEDIUM
        elif confidence > 0.6:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW

    @staticmethod
    def classify_trajectory(
        centroids: List[Tuple[float, float]],
        timestamps: List[float],
        aspects: List[float],
        frame_dims: Tuple[int, int],
        bbox_sizes: List[float],
    ) -> Tuple[str, float, Dict[str, float]]:
        """Standalone classification for evaluation scripts.

        Returns (label, confidence, features_dict).
        """
        tf = TrajectoryFeatures(centroids, timestamps, aspects,
                                frame_dims, bbox_sizes)

        best_label, best_score = "unknown", 0.0
        all_scores = {}
        for rule in TMS_RULES:
            score = rule.score(tf.features)
            all_scores[rule.label] = round(score, 4)
            if score > best_score:
                best_score = score
                best_label = rule.label

        return best_label, round(best_score, 4), tf.features
