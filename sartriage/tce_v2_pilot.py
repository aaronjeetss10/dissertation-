import json
import math
import numpy as np

def assess_initial_state(track_first_n_frames, speed_threshold=20.0):
    if len(track_first_n_frames) < 2:
        return "UNKNOWN", 0.1
        
    cxs = [t[0] for t in track_first_n_frames]
    cys = [t[1] for t in track_first_n_frames]
    ws = [t[2] for t in track_first_n_frames]
    hs = [t[3] for t in track_first_n_frames]
    
    dxs = [cxs[i] - cxs[i-1] for i in range(1, len(track_first_n_frames))]
    dys = [cys[i] - cys[i-1] for i in range(1, len(track_first_n_frames))]
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    
    initial_speed = np.mean(speeds) if speeds else 0.0
    
    # Due to ego-motion (drone panning), absolute pixel speeds sit between 5-30px/frame even when stationary.
    initial_stationarity = sum(1 for s in speeds if s < speed_threshold) / len(speeds) if speeds else 1.0
    
    # Calculate AR as H/W to match pilot instructions (0.6 threshold)
    ars = [h / (w + 1e-8) for w, h in zip(ws, hs)]
    initial_aspect_ratio = np.mean(ars)
    
    # Logic exactly as requested
    if initial_stationarity > 0.8 and initial_aspect_ratio < 0.6:
        return "CRITICAL_STATIC", 0.8
    elif initial_stationarity > 0.8:
        return "SUSTAINED_STILL", 0.5
    elif initial_speed > speed_threshold:
        return "MOVING_FAST", 0.3
    else:
        return "MOVING_SLOW", 0.2

def tce_state_machine_v2(traj, speed_thresh=20.0):
    if len(traj) < 2: return "UNKNOWN", 0.1
    
    # 1. Initial State Assessment
    track_first_n = traj[:10]
    state, score = assess_initial_state(track_first_n, speed_thresh)
    
    cxs = [t[0] for t in traj]
    cys = [t[1] for t in traj]
    dxs = [cxs[i] - cxs[i-1] for i in range(1, len(traj))]
    dys = [cys[i] - cys[i-1] for i in range(1, len(traj))]
    speeds = [math.sqrt(dx**2 + dy**2) for dx, dy in zip(dxs, dys)]
    
    still_count = 0
    fast_count = 0
    
    for i, spd in enumerate(speeds):
        if spd < speed_thresh: cat = "STILL"
        elif spd < speed_thresh * 1.5: cat = "SLOW"
        else: cat = "FAST"
        
        if cat == "STILL":
            still_count += 1
            if still_count > 15:
                if state in ["MOVING_SLOW", "MOVING_FAST"]:
                    state = "COLLAPSED"
                    score = 0.9
                elif state == "STOPPED":
                    state = "SUSTAINED_STILL"
                    score = max(0.5, score)
                elif state == "CRITICAL_STATIC":
                    state = "CRITICAL_STATIC"
                    score = max(0.8, score)
            elif still_count > 5:
                if state not in ["COLLAPSED", "CRITICAL_STATIC", "SUSTAINED_STILL"]:
                    state = "STOPPED"
                    score = 0.3
        elif cat == "FAST":
            fast_count += 1
            still_count = 0
            if fast_count > 10:
                state = "FAST_MOVING"
                score = 0.2
            else:
                if state not in ["CRITICAL_STATIC", "COLLAPSED"]:
                    state = "MOVING_SLOW"
                    score = 0.1
        else:
            still_count = max(0, still_count - 1)
            # Decay score slowly to penalize persistent low-confidence motion
            if state in ["COLLAPSED", "CRITICAL_STATIC", "SUSTAINED_STILL"]:
                pass # NOTE: I REMOVED score *= 0.9. If they are CRITICAL_STATIC, small bounding box drift shouldn't erase their score!
            else:
                state = "MOVING_SLOW"
                score = 0.1

    score = min(0.99, max(0.1, score))
    return state, score

def run_v2():
    with open("evaluation/real_data/okutama_all_tracks.json", "r") as f:
        data = json.load(f)
        
    valid_tracks = []
    
    for tid, t in data["tracks"].items():
        act = t["primary_action"]
        if act in ["Standing", "Sitting"]: mapped = "stationary"
        elif act == "Walking": mapped = "walking"
        elif act == "Lying": mapped = "lying_down"
        else: continue
             
        traj = []
        for (cx,cy), box in zip(t["centroids"], t["bboxes"]):
            traj.append([cx, cy, box[2], box[3]])
            
        valid_tracks.append({
            "id": tid,
            "trajectory": traj,
            "label": mapped
        })

    class_tce_scores = { "lying_down": [], "stationary": [], "walking": [] }
    tce_results = []
    
    correct_lying = 0
    false_escalations = 0
    total_lying = sum(1 for t in valid_tracks if t["label"] == "lying_down")
    total_non_lying = len(valid_tracks) - total_lying
    
    SPEED_THRESH = 40.0 # Bounding boxes in 4K Okutama with drone motion swing wildly.
    
    for trk in valid_tracks:
        st, score = tce_state_machine_v2(trk["trajectory"], speed_thresh=SPEED_THRESH)
        cls = trk["label"]
        class_tce_scores[cls].append(score)
        
        tce_results.append({
             "id": trk["id"],
             "label": cls,
             "final_state": st,
             "final_score": round(score, 4)
        })
        
        if cls == "lying_down" and score >= 0.8:
             correct_lying += 1
        elif cls != "lying_down" and score >= 0.8:
             false_escalations += 1

    print("Mean TCE Priority Score per Class (V2):")
    for cls in ['lying_down', 'stationary', 'walking']:
        scores = class_tce_scores[cls]
        print(f"  {cls:15s}: {np.mean(scores):.4f}  (Min: {np.min(scores):.2f}, Max: {np.max(scores):.2f})")

    print("\nMetrics:")
    print(f"  Correct lying_down escalations: {correct_lying}/{total_lying}")
    print(f"  False escalations on non-lying : {false_escalations}/{total_non_lying}")
    
    if false_escalations > 0:
        print("  WARNING: The fix introduced false positives!")
    else:
        print("  SUCCESS: Zero false escalations or acceptable margin.")
        
    out_file = "evaluation/real_data/pilot/tce_v2_pilot_results.json"
    with open(out_file, "w") as f:
         json.dump({
             "class_means": {c: float(np.mean(s)) for c, s in class_tce_scores.items()},
             "tracks": tce_results
         }, f, indent=2)

if __name__ == "__main__":
    run_v2()
