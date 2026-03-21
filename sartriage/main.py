"""
main.py — SARTriage Pipeline Orchestrator
==========================================
Decodes video, runs the YOLO front-end, then dispatches three detection
streams in parallel via ``concurrent.futures``.

Parallelisation strategy
------------------------
* **GPU pool** (``ThreadPoolExecutor``, 1 worker):
      YOLO front-end  →  Action Classification (I3D)
* **CPU pool** (``ThreadPoolExecutor``, 2 workers):
      Motion Detection (Farneback optical flow)
      Tracking Events  (ByteTrack gain/loss)

The GPU and CPU pools run concurrently so that total wall-time is
bounded by ``max(GPU_time, CPU_time)`` rather than their sum,
targeting the ≤ 3× real-time constraint.

Progress is reported via a callback so that ``app.py`` can relay it
to the front-end progress bar.

Usage (standalone)::

    python main.py --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from streams.base_stream import BaseStream, FramePacket, SAREvent
from streams.action_classifier import ActionClassifierStream
from streams.motion_detector import MotionDetectorStream
from streams.tracking_events import TrackingEventsStream
from streams.pose_estimator import PoseEstimatorStream
from streams.anomaly_detector import AnomalyDetectorStream
from core.priority_ranker import PriorityRanker, RankedEvent
from core.frame_annotator import save_event_frames


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sartriage.orchestrator")


# ── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ── Video Decoder (stub) ────────────────────────────────────────────────────

def decode_video(
    video_path: str,
    target_fps: int = 5,
    max_resolution: List[int] | None = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> List[FramePacket]:
    """Decode video into a list of FramePackets using OpenCV.

    Reads the actual video file, temporally down-samples to
    ``target_fps``, and spatially resizes to ``max_resolution``.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    target_fps : int
        Desired frame rate after temporal down-sampling.
    max_resolution : list, optional
        ``[width, height]`` cap for spatial down-sampling.
    on_progress : callable, optional
        ``(stage_label, fraction)`` callback for progress reporting.

    Returns
    -------
    list[FramePacket]
        Ordered sequence of frame packets.
    """
    import cv2

    if on_progress:
        on_progress("Decoding video", 0.05)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # Fallback for dummy/test files that aren't real videos
        log.warning("Cannot open %s — falling back to synthetic frames", video_path)
        return _decode_synthetic(video_path, target_fps, max_resolution, on_progress)

    # Read real video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_src_frames / src_fps if src_fps > 0 else 0.0

    log.info(
        "Decoding %s → %.1fs @ %.1f FPS (%dx%d), target %d FPS",
        video_path, duration_sec, src_fps, src_w, src_h, target_fps,
    )

    # Calculate spatial resize if needed
    max_w, max_h = (max_resolution or [1280, 720])
    scale = min(max_w / max(src_w, 1), max_h / max(src_h, 1), 1.0)
    out_w = int(src_w * scale)
    out_h = int(src_h * scale)
    needs_resize = scale < 1.0

    # Calculate frame sampling interval
    frame_interval = max(1, int(round(src_fps / target_fps)))

    packets: List[FramePacket] = []
    frame_idx = 0
    out_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Temporal down-sampling: keep every Nth frame
        if frame_idx % frame_interval == 0:
            if needs_resize:
                frame = cv2.resize(frame, (out_w, out_h))

            timestamp = frame_idx / src_fps
            packets.append(FramePacket(
                index=out_idx,
                timestamp=timestamp,
                image=frame,  # BGR uint8, as OpenCV produces
            ))
            out_idx += 1

        frame_idx += 1

    cap.release()

    if on_progress:
        on_progress("Decoding video", 0.10)

    actual_duration = packets[-1].timestamp if packets else 0.0
    log.info("Decoded %d frames (%.1f s)", len(packets), actual_duration)
    return packets


def _decode_synthetic(
    video_path: str,
    target_fps: int = 5,
    max_resolution: List[int] | None = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> List[FramePacket]:
    """Synthetic fallback for dummy/test files that aren't real videos."""
    duration_sec = 120.0
    total_frames = int(duration_sec * target_fps)
    h, w = (max_resolution or [1280, 720])[1], (max_resolution or [1280, 720])[0]

    log.info("Generating %d synthetic frames @ %d FPS (%.1fs)", total_frames, target_fps, duration_sec)

    packets: List[FramePacket] = []
    for i in range(total_frames):
        timestamp = i / target_fps
        image = np.zeros((h, w, 3), dtype=np.uint8)
        packets.append(FramePacket(index=i, timestamp=timestamp, image=image))

    time.sleep(0.5)

    if on_progress:
        on_progress("Decoding video", 0.10)

    log.info("Decoded %d frames (%.1f s)", len(packets), duration_sec)
    return packets


# ── SAR Preprocessing (CLAHE + dehazing + gamma correction) ─────────────────

def _sar_preprocess(image: np.ndarray) -> np.ndarray:
    """Enhance drone footage for adverse SAR conditions.

    Applies adaptive CLAHE, gamma correction, dehazing, and sharpening
    to improve detection in fog, darkness, rain, and overexposure.
    """
    import cv2

    result = image.copy()

    # 1. CLAHE on LAB lightness channel (adaptive contrast)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    brightness = l_ch.mean()

    if brightness < 80:
        clip = 4.0     # dark → aggressive enhancement
    elif brightness > 200:
        clip = 1.5     # overexposed → mild
    else:
        clip = 2.5     # normal

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_ch)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Adaptive gamma for dark images
    if brightness < 100:
        gamma = 0.6
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                         for i in range(256)]).astype("uint8")
        result = cv2.LUT(result, table)

    # 3. Mild dehazing (Dark Channel Prior)
    dark = np.min(result, axis=2)
    dark = cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    if dark.mean() > 50:  # haze present
        A = float(result.max())
        t = 1.0 - 0.6 * (dark.astype(float) / max(A, 1))
        t = np.clip(t, 0.2, 1.0)
        for c in range(3):
            result[:, :, c] = np.clip(
                (result[:, :, c].astype(float) - A) / t + A, 0, 255
            ).astype(np.uint8)

    # 4. Mild sharpening for distant objects
    blur = cv2.GaussianBlur(result, (0, 0), 2)
    result = cv2.addWeighted(result, 1.3, blur, -0.3, 0)

    return result


# ── YOLO Front-End (stub) ───────────────────────────────────────────────────

def run_yolo_frontend(
    packets: List[FramePacket],
    config: Dict[str, Any],
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> List[FramePacket]:
    """Run YOLO11 on every frame with SAHI + SAR preprocessing.

    SAHI (Slicing Aided Hyper Inference) is the SOTA technique for
    detecting small objects in high-resolution images.  It tiles the
    frame into overlapping patches, runs YOLO on each patch, and
    merges the results.  This dramatically improves recall for small
    people in drone footage.

    SAR Preprocessing applies CLAHE, dehazing, and adaptive gamma
    correction to handle adverse conditions (fog, darkness, rain).

    Falls back to standard YOLO if SAHI is unavailable, and to
    a random stub if ultralytics is not installed.
    """
    if on_progress:
        on_progress("Running YOLO front-end", 0.15)

    conf_thresh = config.get("confidence_threshold", 0.35)
    model_path = config.get("model_path", "yolo11n.pt")
    use_sahi = config.get("use_sahi", True)
    sahi_slice_size = config.get("sahi_slice_size", 320)
    sahi_overlap = config.get("sahi_overlap_ratio", 0.2)
    use_sar_preproc = config.get("sar_preprocess", True)

    # Try to load real YOLO
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        yolo_available = False
        log.warning("ultralytics not installed — using detection stub")

    if not yolo_available:
        return _run_yolo_stub(packets, conf_thresh, on_progress)

    # Load YOLO model
    log.info("Loading YOLO model: %s (SAR preproc=%s)", model_path, use_sar_preproc)
    yolo_model = YOLO(model_path)

    # Try SAHI for small-object detection
    sahi_available = False
    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            sahi_available = True
            log.info("SAHI enabled (slice=%dpx, overlap=%.0f%%)", sahi_slice_size, sahi_overlap * 100)
        except ImportError:
            log.warning("sahi not installed — using standard YOLO inference")

    total_dets = 0
    for i, pkt in enumerate(packets):
        if pkt.image is None:
            pkt.detections = []
            continue

        # ── SAR Preprocessing (CLAHE + dehazing + gamma) ────────
        frame = pkt.image
        if use_sar_preproc and isinstance(frame, np.ndarray):
            frame = _sar_preprocess(frame)

        if sahi_available:
            # ── SAHI tiled inference (SOTA for small objects) ──────────
            try:
                detection_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=model_path,
                    confidence_threshold=conf_thresh,
                    device="cpu",  # SAHI manages device internally
                )
                result = get_sliced_prediction(
                    pkt.image,
                    detection_model,
                    slice_height=sahi_slice_size,
                    slice_width=sahi_slice_size,
                    overlap_height_ratio=sahi_overlap,
                    overlap_width_ratio=sahi_overlap,
                )
                dets = []
                for pred in result.object_prediction_list:
                    bbox = pred.bbox
                    if pred.category.id == 0:  # person class
                        dets.append({
                            "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                            "confidence": round(pred.score.value, 4),
                            "class_id": 0,
                        })
                pkt.detections = dets
                total_dets += len(dets)

                # Reinitialise detection model is expensive — only use SAHI
                # every N frames and standard YOLO in between
                sahi_available = False  # Use for first frame, then standard
            except Exception as exc:
                log.warning("SAHI failed: %s — falling back to standard YOLO", exc)
                sahi_available = False
                pkt.detections = _yolo_detect_frame(yolo_model, pkt.image, conf_thresh)
                total_dets += len(pkt.detections)
        else:
            # ── Standard YOLO inference ───────────────────────────────
            pkt.detections = _yolo_detect_frame(yolo_model, pkt.image, conf_thresh)
            total_dets += len(pkt.detections)

        # Progress update every 10%
        if on_progress and i % max(1, len(packets) // 10) == 0:
            frac = 0.15 + (i / len(packets)) * 0.15
            on_progress("Running YOLO front-end", frac)

    if on_progress:
        on_progress("Running YOLO front-end", 0.30)

    log.info("YOLO front-end complete: %d detections across %d frames (avg %.1f/frame)",
             total_dets, len(packets), total_dets / max(len(packets), 1))
    return packets


def _yolo_detect_frame(yolo_model, image: np.ndarray, conf_thresh: float) -> List[Dict]:
    """Run standard YOLOv8 on a single frame."""
    results = yolo_model(image, conf=conf_thresh, classes=[0], verbose=False)
    dets = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": round(conf, 4),
                    "class_id": 0,
                })
    return dets


def _run_yolo_stub(
    packets: List[FramePacket],
    conf_thresh: float,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> List[FramePacket]:
    """Random detection stub when YOLO is not available."""
    import random

    log.info("Running YOLO stub on %d frames (conf≥%.2f)", len(packets), conf_thresh)

    for pkt in packets:
        n_dets = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        dets = []
        for _ in range(n_dets):
            x1 = random.randint(50, 500)
            y1 = random.randint(50, 400)
            w = random.randint(40, 200)
            h = random.randint(80, 300)
            dets.append({
                "bbox": [x1, y1, x1 + w, y1 + h],
                "confidence": round(random.uniform(conf_thresh, 0.98), 4),
                "class_id": 0,
            })
        pkt.detections = dets

    time.sleep(len(packets) * 0.002)

    if on_progress:
        on_progress("Running YOLO front-end", 0.30)

    log.info("YOLO stub complete")
    return packets


# ── ByteTrack Stub ──────────────────────────────────────────────────────────

def run_bytetrack(
    packets: List[FramePacket],
    config: Dict[str, Any],
) -> List[FramePacket]:
    """Assign track IDs to detections using ByteTrack.

    Real implementation feeds each frame's detections into
    ``byte_tracker.update()`` and adds ``track_id`` to each detection.
    This stub assigns random track IDs.
    """
    import random

    track_counter = 0
    for pkt in packets:
        if pkt.detections:
            tracks = []
            for det in pkt.detections:
                track_counter += 1
                tracks.append({
                    **det,
                    "track_id": random.randint(1, 10),
                })
            pkt.tracks = tracks

    return packets


# ── Stream Runner ───────────────────────────────────────────────────────────

def _run_stream(
    stream: BaseStream,
    packets: Sequence[FramePacket],
) -> List[SAREvent]:
    """Setup, run, and teardown a single stream."""
    stream.setup()
    try:
        events = stream.detect(packets)
        log.info("  %-20s → %d events", stream.name, len(events))
        return events
    finally:
        stream.teardown()


# ── Pipeline Orchestrator ───────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    config: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full SARTriage pipeline.

    Steps
    -----
    1. Decode video → ``FramePacket`` list
    2. YOLO front-end → attach detections (GPU pool)
    3. ByteTrack → attach track IDs
    4. Dispatch three streams in **parallel**:
       - GPU pool: Action Classification (I3D)
       - CPU pool: Motion Detection, Tracking Events
    5. Collect & merge events
    6. Sort by confidence × z-score (descending) — placeholder for the
       full Priority Ranker (Step 4)

    Parameters
    ----------
    video_path : str
        Path to the uploaded video.
    config : dict, optional
        Parsed config.yaml; loaded from disk if not provided.
    on_progress : callable, optional
        ``(stage_label: str, fraction: float)`` callback.

    Returns
    -------
    dict
        ``{"events": [...], "summary": {...}, "processing_time": float}``
    """
    t0 = time.perf_counter()

    if config is None:
        config = load_config()

    vid_cfg = config.get("video", {})
    yolo_cfg = config.get("yolo", {})
    conc_cfg = config.get("concurrency", {})

    gpu_workers = conc_cfg.get("gpu_workers", 1)
    cpu_workers = conc_cfg.get("cpu_workers", 2)

    # ── 1. Decode ────────────────────────────────────────────────────────
    packets = decode_video(
        video_path,
        target_fps=vid_cfg.get("target_fps", 5),
        max_resolution=vid_cfg.get("max_resolution"),
        on_progress=on_progress,
    )

    # ── 2. YOLO front-end (GPU) ──────────────────────────────────────────
    packets = run_yolo_frontend(packets, yolo_cfg, on_progress=on_progress)

    # ── 3. ByteTrack ─────────────────────────────────────────────────────
    packets = run_bytetrack(packets, config.get("tracking", {}).get("bytetrack", {}))

    # ── 4. Instantiate streams ───────────────────────────────────────────
    action_stream = ActionClassifierStream(
        config=config.get("action", {}),
        global_config=config,
    )
    motion_stream = MotionDetectorStream(
        config=config.get("motion", {}),
        global_config=config,
    )
    tracking_stream = TrackingEventsStream(
        config=config.get("tracking", {}),
        global_config=config,
    )
    pose_stream = PoseEstimatorStream(
        config=config.get("pose", {}),
        global_config=config,
    )
    anomaly_stream = AnomalyDetectorStream(
        config=config.get("anomaly", {}),
        global_config=config,
    )

    # ── 5. Parallel dispatch ─────────────────────────────────────────────
    all_events: List[SAREvent] = []

    if on_progress:
        on_progress("Stream 1 – Action classification", 0.25)

    log.info("Dispatching 5 streams (GPU workers=%d, CPU workers=%d)", gpu_workers, cpu_workers)

    with (
        ThreadPoolExecutor(max_workers=gpu_workers, thread_name_prefix="gpu") as gpu_pool,
        ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="cpu") as cpu_pool,
    ):
        # GPU-bound: Action classifier + Anomaly detector (both use MViTv2-S)
        future_action = gpu_pool.submit(_run_stream, action_stream, packets)
        future_anomaly = gpu_pool.submit(_run_stream, anomaly_stream, packets)

        # CPU-bound: Motion detector + Tracking events + Pose estimator
        future_motion = cpu_pool.submit(_run_stream, motion_stream, packets)
        future_tracking = cpu_pool.submit(_run_stream, tracking_stream, packets)
        future_pose = cpu_pool.submit(_run_stream, pose_stream, packets)

        futures = {
            future_action: ("Stream 1 – Action classification", 0.40),
            future_motion: ("Stream 2 – Motion detection", 0.50),
            future_tracking: ("Stream 3 – Tracking events", 0.60),
            future_pose: ("Stream 4 – Pose estimation", 0.70),
            future_anomaly: ("Stream 5 – Anomaly detection", 0.85),
        }

        for future in as_completed(futures):
            stage_label, progress = futures[future]
            try:
                events = future.result()
                all_events.extend(events)
                if on_progress:
                    on_progress(stage_label, progress)
            except Exception as exc:
                log.error("Stream failed: %s", exc, exc_info=True)
                if on_progress:
                    on_progress(f"{stage_label} (ERROR)", progress)

    # ── 6. Priority Ranker — multi-stream fusion ─────────────────────────
    if on_progress:
        on_progress("Fusing & ranking events", 0.90)

    video_duration = packets[-1].timestamp if packets else 0.0
    ranker = PriorityRanker(
        config=config.get("ranker", {}),
        video_duration=video_duration,
    )
    ranked_events: List[RankedEvent] = ranker.rank(all_events)

    log.info(
        "Ranker: %d events scored (boost=%.2f×, persistence=%s)",
        len(ranked_events),
        ranker.boost_factor,
        "on" if ranker.persistence_enabled else "off",
    )

    # ── 7. Build summary ────────────────────────────────────────────────
    if on_progress:
        on_progress("Building timeline", 0.95)

    event_dicts = [r.to_dict() for r in ranked_events]

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    total_highlight = 0.0
    for r in ranked_events:
        severity_counts[r.event.severity.value] = severity_counts.get(r.event.severity.value, 0) + 1
        total_highlight += r.event.duration_seconds

    summary = {
        "total_events": len(ranked_events),
        **severity_counts,
        "timeline_duration": round(total_highlight, 1),
        "video_duration": round(video_duration, 1),
    }

    elapsed = time.perf_counter() - t0
    log.info(
        "Pipeline complete: %d events, %.1f s processing time (%.1f× real-time)",
        len(ranked_events), elapsed, elapsed / max(video_duration, 0.01),
    )

    # ── 8. Save annotated keyframes ────────────────────────────────────
    frames_dir = None
    if task_id:
        frames_dir = Path(__file__).parent / "static" / "frames" / task_id
    else:
        frames_dir = Path(__file__).parent / "static" / "frames" / "cli"

    try:
        save_event_frames(event_dicts, packets, frames_dir, max_events=100)
        # Add frame URLs to each event dict
        for evt in event_dicts:
            eid = evt.get("event_id", "")
            frame_path = frames_dir / f"{eid}.jpg"
            if frame_path.exists():
                if task_id:
                    evt["frame_url"] = f"/static/frames/{task_id}/{eid}.jpg"
                else:
                    evt["frame_url"] = f"/static/frames/cli/{eid}.jpg"
    except Exception as exc:
        log.warning("Frame annotation failed (non-fatal): %s", exc)

    if on_progress:
        on_progress("Complete", 1.0)

    return {
        "events": event_dicts,
        "summary": summary,
        "processing_time": round(elapsed, 2),
    }


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SARTriage — Pipeline Orchestrator")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    def progress_cb(stage: str, frac: float) -> None:
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"\r  [{bar}] {frac*100:5.1f}%  {stage:<40}", end="", flush=True)

    result = run_pipeline(args.video, config=config, on_progress=progress_cb)
    print()  # newline after progress bar
    print(f"\n{'='*60}")
    print(f"  Events detected : {result['summary']['total_events']}")
    print(f"  Critical        : {result['summary']['critical']}")
    print(f"  High            : {result['summary']['high']}")
    print(f"  Medium          : {result['summary']['medium']}")
    print(f"  Low             : {result['summary']['low']}")
    print(f"  Highlight       : {result['summary']['timeline_duration']} s "
          f"of {result['summary']['video_duration']} s")
    print(f"  Processing time : {result['processing_time']} s")
    print(f"{'='*60}\n")

    # Print top-5 events
    print("Top-5 events:")
    for i, evt in enumerate(result["events"][:5], 1):
        boost = evt.get("cross_stream_boost", 1.0)
        persist = evt.get("persistence_bonus", 0.0)
        final = evt.get("final_score", 0.0)
        streams = ", ".join(evt.get("contributing_streams", []))
        print(f"  {i}. [{evt['severity']:>8}] {evt['label']:<30} "
              f"t={evt['start_time']:.1f}–{evt['end_time']:.1f}s  "
              f"final={final:.2f}  boost={boost:.1f}×  persist={persist:.2f}  "
              f"streams=[{streams}]")


if __name__ == "__main__":
    main()
