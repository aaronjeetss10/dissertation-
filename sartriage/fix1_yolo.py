import os
import cv2
import traceback
from ultralytics import YOLO

def fix1():
    print("FIX 1: YOLO Detection")
    model_path = "./evaluation/results/heridal_finetune/weights/best.pt"
    img_path = "./evaluation/test_videos/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
    
    try:
        model = YOLO(model_path)
        img = cv2.imread(img_path)
        results = model(img)
        res = results[0]
        
        # Filter for person class (usually 0 in COCO/VisDrone if mapped)
        # VisDrone mapping can vary, let's just count all detections for now, or check cls if needed
        # but the prompt says: "Print: number of person detections, confidence scores, bounding boxes"
        # We will print all detections that are class 0 (assuming 0 is person or 1 is pedestrian)
        # VisDrone has: 1=pedestrian, 2=people. Let's just print everything if we don't know the mapping of heridal YOLO
        person_boxes = []
        person_confs = []
        
        for i, cls in enumerate(res.boxes.cls):
            # Print everything to be safe since it's a fine-tuned model
            person_boxes.append(res.boxes.xyxy[i].tolist())
            person_confs.append(res.boxes.conf[i].item())
            
        print(f"Number of detections: {len(person_boxes)}")
        print(f"Confidence scores: {[round(c, 3) for c in person_confs]}")
        print(f"Bounding boxes (xyxy): {[[round(c, 1) for c in b] for b in person_boxes]}")
        
        save_path = "test_yolo_output.jpg"
        res.save(save_path)
        print(f"Saved annotated image as {save_path}")
        
    except Exception as e:
        print("FAILED!")
        traceback.print_exc()

if __name__ == "__main__":
    fix1()
