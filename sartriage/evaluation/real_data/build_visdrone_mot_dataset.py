import os
import glob
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime

VISDRONE_DIR = "evaluation/datasets/visdrone_mot/"
OUT_JSON = "evaluation/real_data/visdrone_mot_tracks.json"

def get_size_bin(px):
    if px < 20: return "<20px"
    elif px < 35: return "20-35px"
    elif px < 50: return "35-50px"
    elif px < 75: return "50-75px"
    elif px < 100: return "75-100px"
    else: return ">100px"

def build_visdrone():
    print("STEP 1 & 2: Parse and filter VisDrone-MOT sequences")
    txt_files = glob.glob(os.path.join(VISDRONE_DIR, "*.txt"))
    print(f"Found {len(txt_files)} VisDrone-MOT annotation files.")
    
    master_tracks = {}
    sequences_processed = []
    total_frames = 0
    
    for fpath in txt_files:
        seq_name = os.path.basename(fpath).replace(".txt", "")
        sequences_processed.append(seq_name)
        
        tracks_in_file = defaultdict(lambda: {
            "frames": [], "centroids": [], "bboxes": [], 
            "sizes": [], "action_labels": []
        })
        
        with open(fpath, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                
                # Format: frame_id, track_id, x, y, w, h, score, object_class, truncation, occlusion
                frame = int(parts[0])
                track_id = int(parts[1])
                xmin = float(parts[2])
                ymin = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                obj_class = int(parts[7])
                
                # Filter for person classes (1: pedestrian, 2: people)
                if obj_class not in [1, 2]:
                    continue
                
                label = "pedestrian" if obj_class == 1 else "people"
                
                w = max(0.0, w)
                h = max(0.0, h)
                size = math.sqrt(w * h)
                
                tid = f"seq{seq_name}_track{track_id}"
                
                tracks_in_file[tid]["frames"].append(frame)
                tracks_in_file[tid]["centroids"].append([xmin+w/2, ymin+h/2])
                tracks_in_file[tid]["bboxes"].append([xmin, ymin, w, h])
                tracks_in_file[tid]["sizes"].append(size)
                tracks_in_file[tid]["action_labels"].append(label)
                total_frames += 1
                
        # STEP 3: Extract Trajectories
        for tid, data in tracks_in_file.items():
            labels = data["action_labels"]
            primary_action = max(set(labels), key=labels.count) if labels else "person"
            mean_sz = sum(data["sizes"]) / len(data["sizes"]) if data["sizes"] else 0.0
            
            master_tracks[tid] = {
                "sequence": seq_name,
                "frames": data["frames"],
                "centroids": data["centroids"],
                "bboxes": data["bboxes"],
                "sizes": data["sizes"],
                "primary_action": primary_action,
                "mean_size_px": round(mean_sz, 2),
                "track_length_frames": len(data["frames"]),
                "size_bin": get_size_bin(mean_sz)
            }
            
    # Dump to JSON
    print("\nSTEP 5: Building master trajectory dataset...")
    master_json = {
        "metadata": {
            "total_tracks": len(master_tracks),
            "total_frames": total_frames,
            "sequences_processed": sequences_processed,
            "extraction_date": datetime.now().strftime("%Y-%m-%d"),
            "dataset": "VisDrone2019-MOT"
        },
        "tracks": master_tracks
    }
    
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(master_json, f, indent=2)

    # STEP 4: Stats
    print("\nSTEP 4: Print comprehensive statistics")
    print(f"Total person tracks across all sequences: {len(master_tracks)}")
    
    bins_order = ["<20px", "20-35px", "35-50px", "50-75px", "75-100px", ">100px"]
    size_dist = {b: 0 for b in bins_order}
    for t in master_tracks.values():
        b = t["size_bin"]
        if b in size_dist: size_dist[b] += 1
    
    print("\nPerson size distribution:")
    for b in bins_order:
        cnt = size_dist[b]
        pct = (cnt / max(1, len(master_tracks))) * 100
        print(f"  {b:10s}: {cnt:5d} tracks ({pct:5.1f}%)")
        
    below_50 = size_dist["<20px"] + size_dist["20-35px"] + size_dist["35-50px"]
    print(f"\nCRITICAL: Total tracks with mean person size <50px: {below_50}")
        
    lens = [t["track_length_frames"] for t in master_tracks.values()]
    if lens:
        print("\nTrack length distribution:")
        print(f"  Min: {min(lens)}")
        print(f"  Max: {max(lens)}")
        print(f"  Mean: {statistics.mean(lens):.1f}")
        print(f"  Median: {statistics.median(lens):.1f}")
        
    print(f"\nSaved tracking dataset securely to: {OUT_JSON}")

if __name__ == "__main__":
    build_visdrone()
