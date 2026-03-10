"""
streams/motion_detector.py
==========================
Stream 2 — Ego-compensated Optical Flow motion detector.

Computes dense Farneback optical flow, subtracts the dominant (camera)
motion via an affine RANSAC step, then flags frames where residual
magnitude is anomalously high.  This stub simulates the computation
and emits plausible dummy events.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Sequence

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent


class MotionDetectorStream(BaseStream):
    """Stream 2: frame-level ego-compensated motion anomaly detection."""

    @property
    def name(self) -> str:
        return "motion"

    def setup(self) -> None:
        """Pre-compute Farneback params (stubbed)."""
        self._magnitude_pct = self.config.get("magnitude_percentile", 95)
        self._z_threshold = self.config.get("z_score_threshold", 2.5)
        self._ego_enabled = self.config.get("ego_compensation", {}).get("enabled", True)
        super().setup()

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Compute optical flow between consecutive frames and flag spikes.

        Real implementation:
          1. cv2.calcOpticalFlowFarneback between frame[i] and frame[i+1]
          2. Estimate global affine via RANSAC → subtract camera motion
          3. Compute 95th-percentile of residual magnitudes
          4. Z-score against running distribution → emit event if anomalous

        This stub simulates timing and produces random events.
        """
        if len(packets) < 2:
            return []

        fps = self.global_config.get("video", {}).get("target_fps", 5)

        # Simulate Farneback computation (~15 ms per frame pair)
        time.sleep(len(packets) * 0.015)

        # Generate per-frame magnitude scores
        magnitudes: List[float] = []
        for _ in range(len(packets) - 1):
            mag = random.gauss(12.0, 5.0)  # baseline camera motion ~12 px
            if random.random() < 0.06:      # ~6 % spike
                mag = random.uniform(35.0, 80.0)
            magnitudes.append(max(0.0, mag))

        events: List[SAREvent] = []

        # Scan for anomalous frames and merge consecutive spikes
        in_event = False
        evt_start = 0

        for i, mag in enumerate(magnitudes):
            z = self.compute_z_score(mag, magnitudes)
            if z >= self._z_threshold:
                if not in_event:
                    in_event = True
                    evt_start = i
            else:
                if in_event:
                    # Close event
                    events.append(self._make_event(
                        packets, magnitudes, evt_start, i, fps
                    ))
                    in_event = False

        # Close dangling event
        if in_event:
            events.append(self._make_event(
                packets, magnitudes, evt_start, len(magnitudes) - 1, fps
            ))

        return events

    def _make_event(
        self,
        packets: Sequence[FramePacket],
        magnitudes: List[float],
        start: int,
        end: int,
        fps: float,
    ) -> SAREvent:
        peak_mag = max(magnitudes[start : end + 1])
        z = self.compute_z_score(peak_mag, magnitudes)
        severity = self.severity_from_z(z)

        return SAREvent(
            stream_name=self.name,
            start_frame=packets[start].index,
            end_frame=packets[end].index,
            start_time=packets[start].timestamp,
            end_time=packets[end].timestamp,
            confidence=round(min(peak_mag / 80.0, 1.0), 4),
            z_score=round(z, 4),
            label="Sudden motion spike" if z >= 3.0 else "Motion anomaly",
            severity=severity,
            metadata={
                "peak_magnitude": round(peak_mag, 2),
                "ego_compensated": self._ego_enabled,
            },
        )
