"""
streams/tracking_events.py
==========================
Stream 3 — ByteTrack Track-Gain / Track-Loss event detection.

Monitors the set of active track IDs across frames.  A "gain" event
fires when a new track persists for ≥ N frames; a "loss" event fires
when a track disappears for ≥ M frames.  Both situations are
operationally relevant in SAR (new person entering scene, person
disappearing behind terrain, etc.).

This stub simulates ByteTrack output and emits dummy events.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set

from .base_stream import BaseStream, EventSeverity, FramePacket, SAREvent


class TrackingEventsStream(BaseStream):
    """Stream 3: track gain / loss event detection via ByteTrack."""

    @property
    def name(self) -> str:
        return "tracking"

    def setup(self) -> None:
        """Initialise ByteTrack tracker (stubbed)."""
        bt_cfg = self.config.get("bytetrack", {})
        self._track_thresh = bt_cfg.get("track_thresh", 0.5)
        self._track_buffer = bt_cfg.get("track_buffer", 30)
        self._match_thresh = bt_cfg.get("match_thresh", 0.8)

        evt_cfg = self.config.get("events", {})
        self._gain_min_frames = evt_cfg.get("gain_min_frames", 3)
        self._loss_grace = evt_cfg.get("loss_grace_period", 15)
        self._z_threshold = self.config.get("z_score_threshold", 2.0)
        super().setup()

    def detect(self, packets: Sequence[FramePacket]) -> List[SAREvent]:
        """Detect track gain/loss events across the frame sequence.

        Real implementation:
          1. Feed each frame's YOLO detections into ByteTrack
          2. Compare active track set between frames
          3. Emit GAIN when a new track survives ≥ gain_min_frames
          4. Emit LOSS when a track is absent ≥ loss_grace_period frames

        This stub simulates tracking dynamics.
        """
        if not packets:
            return []

        fps = self.global_config.get("video", {}).get("target_fps", 5)

        # Simulate ByteTrack update time (~5 ms per frame)
        time.sleep(len(packets) * 0.005)

        # --- Simulate track-ID evolution ---
        active_tracks: Set[int] = set()
        next_id = 1
        gain_counts: List[int] = []   # gains per frame
        loss_counts: List[int] = []   # losses per frame
        events: List[SAREvent] = []

        for pkt in packets:
            gains = 0
            losses = 0

            # Random chance a new track appears
            if random.random() < 0.04:
                active_tracks.add(next_id)
                gains += 1
                # Emit gain event
                events.append(SAREvent(
                    stream_name=self.name,
                    start_frame=pkt.index,
                    end_frame=min(pkt.index + self._gain_min_frames, packets[-1].index),
                    start_time=pkt.timestamp,
                    end_time=pkt.timestamp + self._gain_min_frames / fps,
                    confidence=round(random.uniform(0.6, 0.95), 4),
                    z_score=0.0,  # will be recomputed below
                    label=f"Track gain (person #{next_id})",
                    severity=EventSeverity.MEDIUM,
                    track_id=next_id,
                    metadata={"event_type": "gain"},
                ))
                next_id += 1

            # Random chance an existing track is lost
            if active_tracks and random.random() < 0.03:
                lost_id = random.choice(list(active_tracks))
                active_tracks.discard(lost_id)
                losses += 1
                events.append(SAREvent(
                    stream_name=self.name,
                    start_frame=pkt.index,
                    end_frame=min(pkt.index + self._loss_grace, packets[-1].index),
                    start_time=pkt.timestamp,
                    end_time=pkt.timestamp + self._loss_grace / fps,
                    confidence=round(random.uniform(0.50, 0.85), 4),
                    z_score=0.0,
                    label=f"Track loss (person #{lost_id})",
                    severity=EventSeverity.HIGH,
                    track_id=lost_id,
                    metadata={"event_type": "loss"},
                ))

            gain_counts.append(gains)
            loss_counts.append(losses)

        # Recompute z-scores based on population statistics
        all_counts = [g + l for g, l in zip(gain_counts, loss_counts)]
        for evt in events:
            frame_idx = evt.start_frame - packets[0].index
            if 0 <= frame_idx < len(all_counts):
                z = self.compute_z_score(float(all_counts[frame_idx]), [float(c) for c in all_counts])
                evt.z_score = round(z, 4)
                evt.severity = self.severity_from_z(z)

        return events
