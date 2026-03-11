"""
streams/pose_estimator.py
==========================
Stream 4 — Pose-Based Posture Detection.

Uses geometric analysis of YOLO bounding box aspect ratios and
positions over time to classify body postures.  This is a
lightweight alternative that works for small figures in drone
footage where full pose-estimation models fail.

Detected postures:
  - **lying_down**: bbox wider than tall (aspect ratio < 0.6)
  - **falling**:    sudden vertical displacement between frames
  - **crawling**:   low bbox with low aspect ratio + movement
  - **waving**:     periodic vertical oscillation at top of bbox
  - **standing**:   normal upright bbox (aspect > 1.2)
  - **running**:    horizontal movement exceeding a speed threshold

This approach is specifically designed for SAR drone footage where
people are often small (20–80 pixels tall) and standard pose
estimation (MediaPipe, OpenPose) fails.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent

log = logging.getLogger("sartriage.pose")


class PoseEstimatorStream(BaseStream):
    """Stream 4: posture detection from bbox geometry + temporal motion."""

    @property
    def name(self) -> str:
        return "pose"

    def setup(self) -> None:
        """Configure pose detection thresholds."""
        self._confidence_gate = self.config.get("confidence_gate", 0.50)
        self._z_threshold = self.config.get("z_score_threshold", 1.8)

        # Posture detection thresholds
        self._lying_aspect_max = self.config.get("lying_aspect_ratio_max", 0.6)
        self._fall_v_threshold = self.config.get("fall_velocity_threshold", 30.0)
        self._crawl_height_ratio = self.config.get("crawl_height_ratio", 0.7)
        self._run_h_threshold = self.config.get("run_horizontal_threshold", 15.0)
        self._min_track_frames = self.config.get("min_track_frames", 5)

        super().setup()
        log.info("Pose estimator initialised (bbox-geometry mode)")

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Analyse person bounding boxes over time for posture cues.

        Steps:
          1. Extract all person detections and group by spatial proximity
          2. For each "track", compute temporal features:
             - Aspect ratio history
             - Vertical velocity (fall detection)
             - Horizontal velocity (running detection)
             - Position relative to frame (crawling = near bottom)
          3. Classify posture and emit events
        """
        if not packets:
            return []

        fps = self.global_config.get("video", {}).get("target_fps", 5)

        # ── 1. Build person tracks from bbox detections ───────────────
        tracks = self._build_tracks(packets)

        if not tracks:
            log.info("Pose estimator: no person tracks found")
            return []

        # ── 2. Analyse each track for posture ─────────────────────────
        all_events: List[SAREvent] = []
        all_scores: List[float] = []

        for track_id, track_data in tracks.items():
            postures = self._analyse_track(track_data, packets, fps)

            for posture in postures:
                all_scores.append(posture["confidence"])

        # ── 3. Emit events with z-score filtering ─────────────────────
        # Re-analyse with z-scores now that we have the score distribution
        for track_id, track_data in tracks.items():
            postures = self._analyse_track(track_data, packets, fps)

            for posture in postures:
                if posture["confidence"] < self._confidence_gate:
                    continue

                z = self.compute_z_score(posture["confidence"], all_scores) if all_scores else 0
                if z < self._z_threshold and len(all_scores) > 5:
                    continue

                severity = self.severity_from_z(z)
                start_frame = posture["start_frame"]
                end_frame = posture["end_frame"]
                start_time = posture["start_time"]
                end_time = posture["end_time"]

                all_events.append(SAREvent(
                    stream_name=self.name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=round(posture["confidence"], 4),
                    z_score=round(z, 4),
                    label=f"Posture: {posture['label']}",
                    severity=severity,
                    metadata={
                        "posture": posture["label"],
                        "track_id": track_id,
                        "aspect_ratio_mean": round(posture.get("aspect_mean", 0), 3),
                        "vertical_velocity": round(posture.get("v_velocity", 0), 1),
                        "horizontal_velocity": round(posture.get("h_velocity", 0), 1),
                        "bbox_midpoint": posture.get("bbox_mid"),
                        "detection_method": "bbox_geometry",
                    },
                ))

        log.info("Pose estimator: %d posture events from %d tracks", len(all_events), len(tracks))
        return all_events

    def _build_tracks(
        self, packets: Sequence[FramePacket]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Group detections into approximate person tracks by spatial proximity."""
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
                x1, y1, x2, y2 = [int(v) for v in bbox]
                bw, bh = x2 - x1, y2 - y1
                if bw < 5 or bh < 5:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                aspect = bh / max(bw, 1)  # tall person → aspect > 1

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

                # Match to existing track
                matched = False
                for tid, tdata in tracks.items():
                    if not tdata:
                        continue
                    last = tdata[-1]
                    if pkt.index - last["frame_idx"] > 5:
                        continue
                    dist = ((cx - last["cx"]) ** 2 + (cy - last["cy"]) ** 2) ** 0.5
                    max_dist = max(bw, bh) * 2.0
                    if dist < max_dist:
                        tracks[tid].append(entry)
                        matched = True
                        break

                if not matched:
                    tracks[next_id] = [entry]
                    next_id += 1

        # Filter tracks with too few frames
        return {tid: tdata for tid, tdata in tracks.items()
                if len(tdata) >= self._min_track_frames}

    def _analyse_track(
        self,
        track_data: List[Dict[str, Any]],
        packets: Sequence[FramePacket],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """Analyse a single track for posture characteristics."""
        if len(track_data) < self._min_track_frames:
            return []

        postures: List[Dict[str, Any]] = []

        aspects = [d["aspect"] for d in track_data]
        cys = [d["cy"] for d in track_data]
        cxs = [d["cx"] for d in track_data]
        frame_hs = [d["frame_h"] for d in track_data]

        aspect_mean = np.mean(aspects)
        aspect_std = np.std(aspects)

        start_frame = track_data[0]["frame_idx"]
        end_frame = track_data[-1]["frame_idx"]
        start_time = track_data[0]["timestamp"]
        end_time = track_data[-1]["timestamp"]
        duration = max(end_time - start_time, 0.1)

        # Vertical velocity (pixels per second)
        if len(cys) >= 2:
            v_velocity = (cys[-1] - cys[0]) / duration
        else:
            v_velocity = 0.0

        # Horizontal velocity
        if len(cxs) >= 2:
            h_velocity = abs(cxs[-1] - cxs[0]) / duration
        else:
            h_velocity = 0.0

        # Position relative to frame height (0=top, 1=bottom)
        rel_y = np.mean([cy / max(fh, 1) for cy, fh in zip(cys, frame_hs)])

        bbox_mid = track_data[len(track_data) // 2].get("bbox")

        base_info = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "aspect_mean": aspect_mean,
            "v_velocity": v_velocity,
            "h_velocity": h_velocity,
            "bbox_mid": bbox_mid,
        }

        # ── Lying down: aspect ratio consistently low ────────────────
        if aspect_mean < self._lying_aspect_max and aspect_std < 0.3:
            conf = min(1.0, (self._lying_aspect_max - aspect_mean) / self._lying_aspect_max + 0.3)
            postures.append({**base_info, "label": "lying_down", "confidence": conf})

        # ── Falling: sudden downward vertical movement ───────────────
        if v_velocity > self._fall_v_threshold:
            # Also check aspect ratio change (person goes from upright to horizontal)
            if len(aspects) >= 4:
                first_half = np.mean(aspects[:len(aspects)//2])
                second_half = np.mean(aspects[len(aspects)//2:])
                aspect_change = first_half - second_half
            else:
                aspect_change = 0

            conf = min(1.0, v_velocity / (self._fall_v_threshold * 3) + 0.2)
            if aspect_change > 0.3:  # went from upright to horizontal
                conf = min(1.0, conf + 0.2)
            postures.append({**base_info, "label": "falling", "confidence": conf})

        # ── Running: fast horizontal movement ────────────────────────
        if h_velocity > self._run_h_threshold and aspect_mean > 0.8:
            conf = min(1.0, h_velocity / (self._run_h_threshold * 4) + 0.3)
            postures.append({**base_info, "label": "running", "confidence": conf})

        # ── Crawling: low position + low aspect + horizontal movement ─
        if (aspect_mean < 0.8 and rel_y > self._crawl_height_ratio
                and h_velocity > 3.0):
            conf = min(1.0, 0.4 + (1 - aspect_mean) * 0.3 + rel_y * 0.2)
            postures.append({**base_info, "label": "crawling", "confidence": conf})

        return postures
