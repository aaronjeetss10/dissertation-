"""
core/frame_annotator.py
========================
Draws detection overlays (bounding boxes, labels, stream badges)
on keyframes for event detail views.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from streams.base_stream import FramePacket

log = logging.getLogger("sartriage.annotator")

# Colour palette for overlays (BGR for OpenCV)
SEVERITY_COLORS = {
    "critical": (0, 0, 255),     # red
    "high":     (0, 140, 255),   # orange
    "medium":   (0, 200, 255),   # yellow
    "low":      (200, 200, 0),   # cyan
}

STREAM_COLORS = {
    "action":   (180, 105, 255),  # pink/magenta
    "motion":   (0, 255, 200),    # teal
    "tracking": (255, 180, 0),    # blue/cyan
    "pose":     (100, 255, 100),  # green
}

BOX_COLOR = (0, 255, 100)  # green for YOLO boxes


def save_event_frames(
    event_dicts: List[Dict[str, Any]],
    packets: List[FramePacket],
    output_dir: Path,
    max_events: int = 100,
) -> None:
    """Save annotated keyframes for the top events.

    For each event, finds the mid-point frame, draws all YOLO bounding
    boxes visible at that time, adds the event label and stream badge,
    and saves as a JPEG.

    Parameters
    ----------
    event_dicts : list
        Serialised event dicts from ``RankedEvent.to_dict()``.
    packets : list[FramePacket]
        The full decoded frame sequence (with detections attached).
    output_dir : Path
        Directory to save annotated frames into.
    max_events : int
        Cap on how many event frames to save.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a quick lookup: frame index → packet
    pkt_lookup = {p.index: p for p in packets}

    saved = 0
    for evt in event_dicts[:max_events]:
        start_frame = evt.get("start_frame", 0)
        end_frame = evt.get("end_frame", start_frame)
        mid_frame = (start_frame + end_frame) // 2

        # Find the best frame (prefer mid, fall back to start)
        pkt = pkt_lookup.get(mid_frame) or pkt_lookup.get(start_frame)
        if pkt is None or pkt.image is None:
            continue

        # Clone the frame
        frame = pkt.image.copy()
        h, w = frame.shape[:2]

        # ── Draw YOLO bounding boxes ──────────────────────────────────
        if pkt.detections:
            for det in pkt.detections:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    conf = det.get("confidence", 0)

                    # Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

                    # Confidence label
                    label = f"person {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), BOX_COLOR, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Draw event info overlay ───────────────────────────────────
        severity = evt.get("severity", "medium")
        sev_color = SEVERITY_COLORS.get(severity, (200, 200, 200))
        stream = evt.get("stream", "unknown")
        stream_color = STREAM_COLORS.get(stream, (200, 200, 200))

        # Semi-transparent banner at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Severity badge
        badge_text = severity.upper()
        cv2.rectangle(frame, (10, 8), (10 + len(badge_text) * 14 + 16, 35), sev_color, -1)
        cv2.putText(frame, badge_text, (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Event label
        label_text = evt.get("label", "Unknown event")
        cv2.putText(frame, label_text, (10 + len(badge_text) * 14 + 30, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Stream badge + timestamp
        t_start = evt.get("start_time", 0)
        t_end = evt.get("end_time", 0)
        conf = evt.get("confidence", 0)
        info_text = f"[{stream}]  {t_start:.1f}s - {t_end:.1f}s  |  Conf: {conf:.0%}"
        cv2.putText(frame, info_text, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, stream_color, 1, cv2.LINE_AA)

        # ── Draw stream-specific overlays ─────────────────────────────

        if stream == "motion":
            # Draw motion indicator arrows (simulated direction)
            for i in range(5):
                cx = np.random.randint(w // 4, 3 * w // 4)
                cy = np.random.randint(h // 4, 3 * h // 4)
                dx = np.random.randint(-30, 30)
                dy = np.random.randint(-30, 30)
                cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy),
                                (0, 255, 255), 2, tipLength=0.3)

        elif stream == "tracking":
            # Draw track ID near center of detections
            if pkt.detections:
                for i, det in enumerate(pkt.detections):
                    bbox = det.get("bbox", [])
                    if len(bbox) == 4:
                        cx = (int(bbox[0]) + int(bbox[2])) // 2
                        cy = int(bbox[3]) + 15
                        track_id = det.get("track_id", i + 1)
                        cv2.putText(frame, f"ID:{track_id}", (cx - 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 180, 0), 1, cv2.LINE_AA)

        elif stream == "pose":
            # Highlight the person bbox with posture annotation
            meta = evt.get("metadata", {})
            person_bbox = meta.get("bbox_midpoint")
            posture = meta.get("posture", "unknown")
            aspect = meta.get("aspect_ratio_mean", 0)
            v_vel = meta.get("vertical_velocity", 0)
            h_vel = meta.get("horizontal_velocity", 0)

            if person_bbox and len(person_bbox) == 4:
                px1, py1, px2, py2 = [int(v) for v in person_bbox]
                # Draw highlighted bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 255, 100), 3)
                # Posture label
                plabel = f"{posture} (AR:{aspect:.2f})"
                cv2.putText(frame, plabel, (px1, py1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
                # Velocity indicators
                if abs(v_vel) > 5:
                    arrow_len = min(int(abs(v_vel)), 50)
                    direction = 1 if v_vel > 0 else -1
                    acx = (px1 + px2) // 2
                    acy = (py1 + py2) // 2
                    cv2.arrowedLine(frame, (acx, acy), (acx, acy + arrow_len * direction),
                                    (0, 0, 255), 2, tipLength=0.3)
                if abs(h_vel) > 5:
                    arrow_len = min(int(abs(h_vel)), 50)
                    acx = (px1 + px2) // 2
                    acy = (py1 + py2) // 2
                    cv2.arrowedLine(frame, (acx, acy), (acx + arrow_len, acy),
                                    (255, 255, 0), 2, tipLength=0.3)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), sev_color, 3)

        # ── Save ──────────────────────────────────────────────────────
        event_id = evt.get("event_id", f"evt_{saved}")
        filename = f"{event_id}.jpg"
        cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved += 1

    log.info("Saved %d annotated keyframes to %s", saved, output_dir)
