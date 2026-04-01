import os
import json
import torch
from ultralytics import YOLO
import torchvision.models.video as video_models
import sys
import gc

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.extract_okutama_trajectories import extract_okutama
from evaluation.real_data.tms12_standalone import extract_tms12
import cv2

def run_integration_test():
    print("\n--- INTEGRATION VERIFICATION ---")
    
    # 1. Okutama Extraction
    print("1. Extracting Okutama tracks...")
    in_file = "./evaluation/datasets/okutama/1.1.1.txt"
    out_file = "./evaluation/real_data/okutama_tracks_sample.json"
    extract_okutama(in_file, out_file)
    
    with open(out_file, 'r') as f:
         data = json.load(f)
         
    # 2. Pick track > 30 frames
    selected_track = None
    for tid, d in data.items():
        if len(d["frames"]) > 30:
             selected_track = d
             print(f"2. Picked track_id {tid} with {len(d['frames'])} frames.")
             break
            
    # 3. Extract TMS-12
    print("3. Extracting TMS-12 features...")
    traj = []
    for (cx,cy), box in zip(selected_track["centroids"], selected_track["bboxes"]):
        traj.append([cx, cy, box[2], box[3]])
        
    features = extract_tms12(traj)
    print("4. Feature values:", [round(x, 4) for x in features])
    
    # 5. Load R3D-18
    print("5. Loading R3D-18...")
    model_r3d = video_models.r3d_18(weights=None)
    model_r3d.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model_r3d.fc.in_features, 9)
    )
    st = torch.load("models/action_r3d18_sar.pt", map_location="cpu")
    model_r3d.load_state_dict(st.get('model_state_dict', st), strict=True)
    model_r3d.eval()
    
    d_in = torch.randn(1, 3, 16, 112, 112)
    with torch.no_grad():
        out = model_r3d(d_in)
    print("  Output shape:", list(out.shape))
    
    # Free memory to prevent macOS SigKill
    del model_r3d
    del st
    gc.collect()
    
    # 6. Load YOLO
    print("6. Loading YOLO (Fix 1)...")
    model_yolo = YOLO("./evaluation/results/heridal_finetune/weights/best.pt")
    img = cv2.imread("./evaluation/test_videos/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg")
    results = model_yolo(img, verbose=False)
    print(f"  YOLO found {len(results[0].boxes)} detections.")
    
    print("\nALL 4 COMPONENTS VERIFIED")

if __name__ == "__main__":
    run_integration_test()
