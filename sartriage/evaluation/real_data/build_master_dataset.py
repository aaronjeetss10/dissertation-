import os
import glob
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime

OKUTAMA_DIR = "evaluation/datasets/okutama/"
OUT_JSON = "evaluation/real_data/okutama_all_tracks.json"
MAPPING_JSON = "evaluation/real_data/class_mapping.json"

def get_size_bin(px):
    if px < 20: return "<20px"
    elif px < 35: return "20-35px"
    elif px < 50: return "35-50px"
    elif px < 75: return "50-75px"
    elif px < 100: return "75-100px"
    elif px < 150: return "100-150px"
    else: return ">150px"

def build_master():
    print("STEP 1: Find and process all annotation files")
    txt_files = glob.glob(os.path.join(OKUTAMA_DIR, "*.txt"))
    print(f"Found {len(txt_files)} annotation files: {[os.path.basename(f) for f in txt_files]}")
    
    master_tracks = {}
    sequences_processed = []
    total_frames = 0
    
    for fpath in txt_files:
        seq_name = os.path.basename(fpath).replace(".txt", "")
        sequences_processed.append(seq_name)
        
        # Temp dict to hold per-track arrays for mapping
        tracks_in_file = defaultdict(lambda: {
            "frames": [], "centroids": [], "bboxes": [], 
            "sizes": [], "action_labels": []
        })
        
        with open(fpath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 11: continue
                
                track_id = parts[0]
                xmin, ymin = float(parts[1]), float(parts[2])
                xmax, ymax = float(parts[3]), float(parts[4])
                frame = int(parts[5])
                label = parts[10].strip('"')
                
                w = max(0.0, xmax - xmin)
                h = max(0.0, ymax - ymin)
                size = math.sqrt(w * h)
                
                tid = f"seq{seq_name}_track{track_id}"
                
                tracks_in_file[tid]["frames"].append(frame)
                tracks_in_file[tid]["centroids"].append([xmin+w/2, ymin+h/2])
                tracks_in_file[tid]["bboxes"].append([xmin, ymin, w, h])
                tracks_in_file[tid]["sizes"].append(size)
                tracks_in_file[tid]["action_labels"].append(label)
                total_frames += 1
                
        for tid, data in tracks_in_file.items():
            labels = data["action_labels"]
            primary_action = max(set(labels), key=labels.count) if labels else "Unknown"
            mean_sz = sum(data["sizes"]) / len(data["sizes"]) if data["sizes"] else 0.0
            
            master_tracks[tid] = {
                "sequence": seq_name,
                "frames": data["frames"],
                "centroids": data["centroids"],
                "bboxes": data["bboxes"],
                "sizes": data["sizes"],
                "action_labels": labels,
                "primary_action": primary_action,
                "mean_size_px": round(mean_sz, 2),
                "track_length_frames": len(data["frames"]),
                "size_bin": get_size_bin(mean_sz)
            }
            
    # STEP 2: JSON Out
    print("STEP 2: Building master trajectory dataset...")
    master_json = {
        "metadata": {
            "total_tracks": len(master_tracks),
            "total_frames": total_frames,
            "sequences_processed": sequences_processed,
            "extraction_date": datetime.now().strftime("%Y-%m-%d")
        },
        "tracks": master_tracks
    }
    
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(master_json, f, indent=2)

    # STEP 3: Stats
    print("\nSTEP 3: Print comprehensive statistics")
    print(f"Total tracks across all sequences: {len(master_tracks)}")
    
    class_dist = defaultdict(int)
    for t in master_tracks.values():
        class_dist[t["primary_action"]] += 1
    
    print("\nClass distribution:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt} tracks")
        
    bins_order = ["<20px", "20-35px", "35-50px", "50-75px", "75-100px", "100-150px", ">150px"]
    size_dist = {b: 0 for b in bins_order}
    for t in master_tracks.values():
        b = t["size_bin"]
        if b in size_dist: size_dist[b] += 1
    
    print("\nPerson size distribution:")
    for b in bins_order:
        cnt = size_dist[b]
        pct = (cnt / max(1, len(master_tracks))) * 100
        print(f"  {b:10s}: {cnt:4d} tracks ({pct:5.1f}%)")
        
    lens = [t["track_length_frames"] for t in master_tracks.values()]
    if lens:
        print("\nTrack length distribution:")
        print(f"  Min: {min(lens)}")
        print(f"  Max: {max(lens)}")
        print(f"  Mean: {statistics.mean(lens):.1f}")
        print(f"  Median: {statistics.median(lens):.1f}")
        
    print("\nPer-class size distribution:")
    # We will compute the mean size for each action class
    for cls in class_dist.keys():
        szs = [t["mean_size_px"] for t in master_tracks.values() if t["primary_action"] == cls]
        print(f"  {cls:15s} -> Min: {min(szs):5.1f} | Max: {max(szs):5.1f} | Mean: {statistics.mean(szs):5.1f}")

    # STEP 4: Quality flags
    print("\nSTEP 4: Flag data quality issues")
    issue_short = [k for k, v in master_tracks.items() if v["track_length_frames"] < 10]
    issue_zero  = [k for k, v in master_tracks.items() if any(s == 0 or math.isnan(s) for s in v["sizes"])]
    issue_rare_cls = [k for k, v in class_dist.items() if v < 10]

    print(f"-> Tracks with < 10 frames (Too short for TMS-12): {len(issue_short)}")
    if issue_short: print(f"   e.g. {issue_short[:3]}")
    
    print(f"-> Tracks where size is 0 or NaN (Annotation errors): {len(issue_zero)}")
    if issue_zero: print(f"   e.g. {issue_zero[:3]}")
        
    print(f"-> Action classes with < 10 tracks (Too few for split): {len(issue_rare_cls)}")
    if issue_rare_cls: print(f"   Affected classes: {issue_rare_cls}")

    # STEP 5: Mappings
    print("\nSTEP 5: Create the class mapping")
    okutama_to_sar_mapping = {
        "Walking": "walking",
        "Running": "running",
        "Lying": "lying_down",
        "Sitting": "stationary",
        "Standing": "stationary"
    }
    # Add any unmapped discovered
    for cls in class_dist.keys():
        if cls not in okutama_to_sar_mapping:
            okutama_to_sar_mapping[cls] = "unmapped"
            
    sar_priority = {
        "Lying": 0.9,
        "Sitting": 0.4,
        "Standing": 0.2,
        "Walking": 0.1,
        "Running": 0.1
    }
    
    mapping_out = {
        "okutama_to_sar_mapping": okutama_to_sar_mapping,
        "sar_priority": sar_priority
    }
    with open(MAPPING_JSON, "w") as f:
        json.dump(mapping_out, f, indent=2)
        
    print(f"Saved {OUT_JSON}")
    print(f"Saved {MAPPING_JSON}")

if __name__ == "__main__":
    build_master()
