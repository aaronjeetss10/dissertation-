import json
import math
import os
from collections import defaultdict
from pathlib import Path

def extract_okutama(in_file, out_file):
    print(f"FIX 3: Extracting Trajectories from {in_file}")
    
    if not os.path.exists(in_file):
        print(f"Error: Could not find {in_file}")
        return
        
    tracks = defaultdict(lambda: {
        "frames": [],
        "centroids": [],
        "bboxes": [],
        "sizes": [],
        "action_labels": []
    })
    
    with open(in_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
                
            track_id = parts[0]
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])
            frame = int(parts[5])
            
            # The label is parts[10] usually (e.g. "Standing")
            label = parts[10].strip('"')
            
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2
            cy = ymin + h / 2
            size = math.sqrt(w * h)
            
            tracks[track_id]["frames"].append(frame)
            tracks[track_id]["centroids"].append([cx, cy])
            tracks[track_id]["bboxes"].append([xmin, ymin, w, h])
            tracks[track_id]["sizes"].append(size)
            tracks[track_id]["action_labels"].append(label)
            
    # Compute primary action for each track
    final_output = {}
    action_counts = defaultdict(int)
    sizes_sum = 0
    total_samples = 0
    
    for tid, data in tracks.items():
        labels = data["action_labels"]
        # Find most common label
        primary_action = max(set(labels), key=labels.count)
        data["primary_action"] = primary_action
        final_output[tid] = data
        
        action_counts[primary_action] += 1
        sizes_sum += sum(data["sizes"])
        total_samples += len(data["sizes"])

    with open(out_file, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"Extraction Summary:")
    print(f"  Total Tracks: {len(final_output)}")
    print(f"  Tracks per action: {dict(action_counts)}")
    print(f"  Average person size (px): {sizes_sum / max(total_samples, 1):.1f}")
    print(f"Saved cleanly to {out_file}")

if __name__ == "__main__":
    in_file = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/datasets/okutama/1.1.1.txt"
    out_file = "/Users/aaronsandhu/dissertation-SAR/dissertation--1/sartriage/evaluation/real_data/okutama_tracks_sample.json"
    extract_okutama(in_file, out_file)
