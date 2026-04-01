"""
EXPERIMENT 6: Full Pipeline Timing Profile
Times each component of the SARTriage pipeline on a single Okutama sequence.
"""
import os, sys, glob, time, json, math
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

TEST_FRAMES_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"

# TCE v2 state machine (inline)
def tce_v2_score(traj, speed_thresh=40.0):
    if len(traj) < 2: return 0.1
    first_n = traj[:10]
    cxs=[t[0] for t in first_n]; cys=[t[1] for t in first_n]
    ws=[t[2] for t in first_n]; hs=[t[3] for t in first_n]
    if len(cxs) < 2: return 0.1
    speeds = [math.sqrt((cxs[i]-cxs[i-1])**2+(cys[i]-cys[i-1])**2) for i in range(1,len(cxs))]
    stat = sum(1 for s in speeds if s < speed_thresh)/len(speeds) if speeds else 1.0
    ars = [h/(w+1e-8) for w,h in zip(ws,hs)]
    ar = np.mean(ars)
    if stat > 0.8 and ar < 0.6: score = 0.8
    elif stat > 0.8: score = 0.5
    else: score = 0.2
    return score

def main():
    print("="*60)
    print("EXPERIMENT 6: Full Pipeline Timing Profile")
    print("="*60)
    
    import platform
    hw = f"{platform.machine()} / {platform.processor()} / macOS {platform.mac_ver()[0]}"
    print(f"Hardware: {hw}")
    
    # Find one test sequence with frames
    seq_dir = None
    seqs = glob.glob(os.path.join(TEST_FRAMES_BASE, "Drone1", "Morning", "Extracted-Frames-1280x720", "*"))
    for s in seqs:
        if os.path.isdir(s):
            jpgs = glob.glob(os.path.join(s, "*.jpg"))
            if len(jpgs) > 100:
                seq_dir = s; break
    
    if not seq_dir:
        print("ERROR: No sequence with enough frames found.")
        return
    
    seq_name = os.path.basename(seq_dir)
    print(f"Using sequence: {seq_name} from {seq_dir}")
    
    imgs = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))[:30]
    print(f"Processing {len(imgs)} frames (~1s at 30fps)")
    
    # Load YOLO
    from ultralytics import YOLO
    weights = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/results/heridal_finetune/weights/best.pt"
    model = YOLO(weights)
    
    # Warmup
    _ = model(imgs[0], verbose=False, conf=0.25)
    
    # Component 1: YOLO detection
    print("\n1. YOLO Detection...")
    yolo_times = []
    all_detections = []  # per-frame list of [x,y,w,h]
    for img_path in imgs:
        t0 = time.time()
        res = model(img_path, verbose=False, conf=0.25, imgsz=1280)[0]
        t1 = time.time()
        yolo_times.append(t1-t0)
        dets = []
        for b in res.boxes:
            if int(b.cls[0]) != 0: continue
            xyxy = b.xyxy[0].tolist()
            dets.append([xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]])
        all_detections.append(dets)
    
    yolo_mean = np.mean(yolo_times)*1000
    print(f"  Mean: {yolo_mean:.1f}ms/frame")
    
    # Component 2: Track extraction (simple greedy IoU tracker)
    print("\n2. Track Extraction...")
    t0 = time.time()
    tracks = {}
    next_id = 0
    active = {}  # track_id -> last_bbox
    
    for frame_idx, dets in enumerate(all_detections):
        matched = set()
        for det in dets:
            best_id, best_iou = -1, 0.3
            cx, cy = det[0]+det[2]/2, det[1]+det[3]/2
            for tid, last_box in active.items():
                lcx, lcy = last_box[0]+last_box[2]/2, last_box[1]+last_box[3]/2
                dist = math.sqrt((cx-lcx)**2+(cy-lcy)**2)
                if dist < 100 and tid not in matched:
                    if dist < best_iou or best_id == -1:
                        best_iou = dist; best_id = tid
            
            if best_id >= 0:
                tracks[best_id].append([cx, cy, det[2], det[3]])
                active[best_id] = det
                matched.add(best_id)
            else:
                tracks[next_id] = [[cx, cy, det[2], det[3]]]
                active[next_id] = det
                next_id += 1
        
        # Remove stale tracks
        for tid in list(active.keys()):
            if tid not in matched and frame_idx > 0:
                del active[tid]
    
    t1 = time.time()
    track_time = (t1-t0)*1000
    track_per_frame = track_time / len(imgs)
    print(f"  Total: {track_time:.1f}ms | Per frame: {track_per_frame:.1f}ms")
    print(f"  Extracted {len(tracks)} tracks")
    
    # Component 3: TMS-12 extraction
    print("\n3. TMS-12 Feature Extraction...")
    t0 = time.time()
    tms_results = {}
    valid_tracks = {tid: traj for tid, traj in tracks.items() if len(traj) >= 20}
    for tid, traj in valid_tracks.items():
        feats = extract_tms12(traj)
        tms_results[tid] = feats
    t1 = time.time()
    tms_time = (t1-t0)*1000
    tms_per_frame = tms_time / len(imgs)
    print(f"  Total: {tms_time:.1f}ms | Per frame: {tms_per_frame:.1f}ms")
    print(f"  Processed {len(tms_results)} tracks")
    
    # Component 4: TCE scoring
    print("\n4. TCE v2 Scoring...")
    t0 = time.time()
    tce_scores = {}
    for tid, traj in valid_tracks.items():
        tce_scores[tid] = tce_v2_score(traj)
    t1 = time.time()
    tce_time = (t1-t0)*1000
    tce_per_frame = tce_time / len(imgs)
    print(f"  Total: {tce_time:.1f}ms | Per frame: {tce_per_frame:.1f}ms")
    
    # Component 5: Ranking (trivial sort)
    print("\n5. Priority Ranking...")
    t0 = time.time()
    ranked = sorted(tce_scores.items(), key=lambda x: -x[1])
    t1 = time.time()
    rank_time = (t1-t0)*1000
    rank_per_frame = rank_time / len(imgs)
    print(f"  Total: {rank_time:.1f}ms | Per frame: {rank_per_frame:.1f}ms")
    
    # Summary
    total_per_frame = yolo_mean + track_per_frame + tms_per_frame + tce_per_frame + rank_per_frame
    video_duration = len(imgs) / 30.0  # 30fps
    total_pipeline = total_per_frame * len(imgs) / 1000.0
    rt_factor = total_pipeline / video_duration
    
    print(f"\n{'='*50}")
    print(f"TIMING SUMMARY ({len(imgs)} frames, {video_duration:.1f}s video)")
    print(f"{'='*50}")
    print(f"  YOLO Detection  : {yolo_mean:7.1f} ms/frame")
    print(f"  Track Extraction : {track_per_frame:7.1f} ms/frame")
    print(f"  TMS-12 Features  : {tms_per_frame:7.1f} ms/frame")
    print(f"  TCE Scoring      : {tce_per_frame:7.1f} ms/frame")
    print(f"  Ranking          : {rank_per_frame:7.1f} ms/frame")
    print(f"  ────────────────────────────────")
    print(f"  TOTAL            : {total_per_frame:7.1f} ms/frame")
    print(f"  Real-time factor : {rt_factor:.2f}x")
    print(f"  Hardware         : {hw}")
    
    results = {
        "components": {
            "yolo_detection_ms": yolo_mean,
            "track_extraction_ms": track_per_frame,
            "tms12_extraction_ms": tms_per_frame,
            "tce_scoring_ms": tce_per_frame,
            "ranking_ms": rank_per_frame,
            "total_ms": total_per_frame
        },
        "realtime_factor": rt_factor,
        "video_duration_s": video_duration,
        "n_frames": len(imgs),
        "n_tracks": len(tracks),
        "hardware": hw
    }
    with open(os.path.join(FULL_DIR, "timing_profile.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("EXPERIMENT 6 COMPLETE.")

if __name__ == "__main__":
    main()
