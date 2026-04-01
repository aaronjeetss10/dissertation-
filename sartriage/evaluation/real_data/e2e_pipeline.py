"""
END-TO-END PIPELINE EVALUATION — SARTriage on 3 Okutama Test Sequences
Runs every available component, documents skips, produces ranked event timelines.
"""
import os, sys, json, math, time, glob
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

OUT_DIR = "evaluation/real_data/full/end_to_end"
os.makedirs(OUT_DIR, exist_ok=True)

TEST_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_BASE, "Labels", "MultiActionLabels", "3840x2160")
YOLO_WEIGHTS = "evaluation/results/heridal_finetune/weights/best.pt"
MVIT_WEIGHTS = "models/action_mvit2_sar.pt"
AAI_WEIGHTS = "evaluation/results/aai_v2_weights.pt"
TRAJMAE_WEIGHTS = "evaluation/results/trajmae_encoder_pretrained.pt"

SCALE_X = 1280.0/3840.0; SCALE_Y = 720.0/2160.0
SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
GT_PRIORITY = {"lying_down":1.0,"sitting":0.6,"standing":0.3,"running":0.2,"walking":0.1}
RF_CLASSES = ["lying_down","stationary","walking","running"]
RF_PRIORITY = {"lying_down":1.0,"stationary":0.45,"walking":0.1,"running":0.2}

SEQUENCES = ["1.1.8","1.2.3","2.2.3"]  # 3 diverse test sequences

def log(msg):
    with open('/tmp/e2e_progress.txt','a') as f: f.write(msg+'\n')
    
# ═══════════════════════════════════════════════════════════════════════
# Component Runners
# ═══════════════════════════════════════════════════════════════════════

def load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_WEIGHTS)
        return model, "RAN"
    except Exception as e:
        return None, f"FAILED: {e}"

def load_mvit_cache():
    """Load pre-computed MViTv2-S predictions from cache."""
    cache_path = os.path.join(OUT_DIR, "mvit2s_predictions_cache.json")
    try:
        with open(cache_path) as f:
            cache = json.load(f)
        return cache, "RAN (pre-computed in separate process to manage memory)"
    except Exception as e:
        return None, f"FAILED: {e}"

def load_aai():
    try:
        from evaluation.aai_v2 import AAIv2MetaClassifier
        model = AAIv2MetaClassifier()
        raw = torch.load(AAI_WEIGHTS, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            model.load_state_dict(raw["model_state_dict"])
        else:
            model.load_state_dict(raw)
        model.eval()
        return model, "RAN"
    except Exception as e:
        return None, f"FALLBACK: {e}"

def load_trajmae():
    try:
        from evaluation.traj_mae import TrajMAE
        model = TrajMAE(num_classes=4, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=50)
        state = torch.load(TRAJMAE_WEIGHTS, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state)
        model.eval()
        return model, "RAN"
    except Exception as e:
        return None, f"FAILED: {e}"

def load_rf():
    """Train SMOTE-balanced RF on full Okutama training set."""
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    X, y = [], []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR_MAP: continue
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        feats = extract_tms12(traj)
        if any(math.isnan(f) or math.isinf(f) for f in feats): continue
        X.append(feats); y.append(RF_CLASSES.index(SAR_MAP[act]))
    X = np.array(X); y = np.array(y)
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_sm, y_sm)
    return rf, "RAN"

def tce_v2_score(traj, speed_thresh=3.0):
    if len(traj) < 2: return 0.5, "UNKNOWN"
    first_n = min(10, len(traj))
    speeds = [math.sqrt((traj[i][0]-traj[i-1][0])**2+(traj[i][1]-traj[i-1][1])**2) for i in range(1,first_n)]
    if not speeds: return 0.5, "UNKNOWN"
    stat = sum(1 for s in speeds if s<speed_thresh)/len(speeds)
    ars = [traj[i][3]/(traj[i][2]+1e-8) for i in range(first_n)]
    mean_ar = np.mean(ars); mean_speed = np.mean(speeds)
    if stat>0.8 and mean_ar<0.6: base=0.85; state="COLLAPSED"
    elif stat>0.8: base=0.55; state="STATIONARY"
    else: base=0.2; state="MOVING"
    if len(traj)>20:
        h1=traj[:len(traj)//2]; h2=traj[len(traj)//2:]
        sp1=np.mean([math.sqrt((h1[i][0]-h1[i-1][0])**2+(h1[i][1]-h1[i-1][1])**2) for i in range(1,len(h1))]) if len(h1)>1 else 0
        sp2=np.mean([math.sqrt((h2[i][0]-h2[i-1][0])**2+(h2[i][1]-h2[i-1][1])**2) for i in range(1,len(h2))]) if len(h2)>1 else 0
        if sp1>speed_thresh and sp2<speed_thresh*0.5: base=max(base,0.75); state="STOPPED"
    return min(1.0,base), state

def compute_emi(frames_dir, frame_ids):
    """Compute EMI from consecutive frames via ORB+RANSAC homography."""
    try:
        orb = cv2.ORB_create(500)
        sorted_ids = sorted(frame_ids)[:30]  # first 30 sampled frames
        speeds = []; hover_count = 0
        prev_gray = None
        for fi in sorted_ids:
            img = cv2.imread(os.path.join(frames_dir, f"{fi}.jpg"))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                kp1, d1 = orb.detectAndCompute(prev_gray, None)
                kp2, d2 = orb.detectAndCompute(gray, None)
                if d1 is not None and d2 is not None and len(d1)>10 and len(d2)>10:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(d1, d2)
                    if len(matches)>4:
                        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                        if H is not None:
                            tx, ty = H[0,2], H[1,2]
                            speed = math.sqrt(tx**2+ty**2)
                            speeds.append(speed)
                            if speed < 2.0: hover_count += 1
            prev_gray = gray
        if not speeds:
            return None, "FAILED: No valid frame pairs"
        mean_speed = np.mean(speeds)
        hover_idx = hover_count / len(speeds)
        if hover_idx > 0.7: phase = "HOVERING"
        elif mean_speed < 5: phase = "SCANNING"
        else: phase = "TRANSIT"
        attn = 1.0 if phase=="HOVERING" else 0.5 if phase=="SCANNING" else 0.0
        return {"mean_speed":float(mean_speed),"hover_index":float(hover_idx),
                "phase":phase,"attention_score":attn}, "RAN"
    except Exception as e:
        return None, f"FAILED: {e}"

MVIT_9 = {0:"falling",1:"crawling",2:"lying_down",3:"running",4:"waving_hand",5:"climbing",6:"stumbling",7:"pushing",8:"pulling"}
MVIT_TO_SAR = {"falling":"lying_down","crawling":"lying_down","lying_down":"lying_down",
               "running":"running","waving_hand":"stationary","climbing":"walking",
               "stumbling":"walking","pushing":"stationary","pulling":"stationary"}

def dcg(scores, k):
    return sum(s/math.log2(i+2) for i,s in enumerate(scores[:k]))

def ndcg(ranked_gt, k):
    ideal = sorted(ranked_gt, reverse=True)
    id_dcg = dcg(ideal, k)
    if id_dcg == 0: return 1.0
    return dcg(ranked_gt, k) / id_dcg

# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline_on_sequence(seq_name, yolo_model, rf_model, mvit_cache,
                              aai_model, trajmae_model, pipeline_status):
    log(f'\n===== SEQUENCE: {seq_name} =====')
    timings = {}
    
    # Find frames directory
    frame_dir = None
    for m in glob.glob(os.path.join(TEST_BASE,"**",seq_name), recursive=True):
        if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
    if not frame_dir:
        log(f"ERROR: No frames for {seq_name}"); return None
    
    # Parse GT annotations
    label_file = os.path.join(LABELS_DIR, f"{seq_name}.txt")
    gt_tracks = defaultdict(lambda: {"frames":[],"bboxes":[],"action":None})
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11: continue
            tid = int(parts[0])
            xmin=float(parts[1])*SCALE_X; ymin=float(parts[2])*SCALE_Y
            xmax=float(parts[3])*SCALE_X; ymax=float(parts[4])*SCALE_Y
            fi = int(parts[5]); act = parts[10].strip('"')
            gt_tracks[tid]["frames"].append(fi)
            gt_tracks[tid]["bboxes"].append([xmin,ymin,xmax-xmin,ymax-ymin])
            gt_tracks[tid]["action"] = act
    
    # Get frame range
    all_frames = set()
    for t in gt_tracks.values(): all_frames.update(t["frames"])
    sorted_frames = sorted(all_frames)
    sampled_frames = sorted_frames[::6]  # every 6th
    log(f'  Frames: {len(sorted_frames)} total, {len(sampled_frames)} sampled')
    
    # ── STAGE 1: YOLO Detection ──────────────────────────────────────
    t0 = time.time()
    detections_per_frame = {}
    if yolo_model:
        for fi in sampled_frames:
            img_path = os.path.join(frame_dir, f"{fi}.jpg")
            if not os.path.exists(img_path): continue
            results = yolo_model(img_path, imgsz=1280, conf=0.1, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # person class
                        x1,y1,x2,y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        dets.append({"bbox":[x1,y1,x2-x1,y2-y1],"conf":conf})
            detections_per_frame[fi] = dets
        pipeline_status["YOLO"] = {"status":"RAN","note":"HERIDAL-finetuned, 1280px, conf>0.1"}
    else:
        pipeline_status["YOLO"] = {"status":"FAILED","note":"Model load failed"}
    timings["yolo_ms"] = (time.time()-t0)/max(len(sampled_frames),1)*1000
    log(f'  YOLO: {sum(len(d) for d in detections_per_frame.values())} detections in {len(detections_per_frame)} frames ({timings["yolo_ms"]:.0f}ms/frame)')
    
    # ── ByteTrack → GT Track Assignment ──────────────────────────────
    t1 = time.time()
    # Match YOLO detections to GT tracks via IoU
    tracked_dets = defaultdict(list)  # track_id → [(frame, bbox, conf)]
    matched_count = 0; unmatched_count = 0
    
    for fi, dets in detections_per_frame.items():
        # GT boxes at this frame
        gt_at_frame = []
        for tid, tv in gt_tracks.items():
            if fi in tv["frames"]:
                idx = tv["frames"].index(fi)
                gt_at_frame.append((tid, tv["bboxes"][idx]))
        
        for det in dets:
            dx, dy, dw, dh = det["bbox"]
            best_iou = 0; best_tid = None
            for tid, (gx, gy, gw, gh) in gt_at_frame:
                # IoU
                ix = max(dx, gx); iy = max(dy, gy)
                ix2 = min(dx+dw, gx+gw); iy2 = min(dy+dh, gy+gh)
                inter = max(0,ix2-ix)*max(0,iy2-iy)
                union = dw*dh + gw*gh - inter
                iou = inter/(union+1e-8)
                if iou > best_iou: best_iou = iou; best_tid = tid
            if best_iou > 0.3 and best_tid is not None:
                tracked_dets[best_tid].append((fi, det["bbox"], det["conf"]))
                matched_count += 1
            else:
                unmatched_count += 1
    
    pipeline_status["ByteTrack"] = {"status":"SKIPPED","note":f"GT track IDs via IoU>0.3 matching. {matched_count} matched, {unmatched_count} unmatched"}
    timings["tracking_ms"] = (time.time()-t1)/max(len(sampled_frames),1)*1000
    log(f'  Tracking: {len(tracked_dets)} tracks matched ({matched_count} dets)')
    
    # ── STAGE 2: Trajectory Extraction ───────────────────────────────
    t2 = time.time()
    tracks_data = {}
    for tid, dets in tracked_dets.items():
        if len(dets) < 5: continue  # need minimum trajectory length
        dets_sorted = sorted(dets, key=lambda d: d[0])
        centroids = [(d[1][0]+d[1][2]/2, d[1][1]+d[1][3]/2) for d in dets_sorted]
        bboxes = [d[1] for d in dets_sorted]
        confs = [d[2] for d in dets_sorted]
        frames = [d[0] for d in dets_sorted]
        mean_size = np.mean([math.sqrt(max(b[2],1)*max(b[3],1)) for b in bboxes])
        mean_conf = np.mean(confs)
        # Raw trajectory for TMS-12
        raw_traj = [[cx,cy,b[2],b[3]] for (cx,cy),b in zip(centroids,bboxes)]
        # Delta tokens for TrajMAE
        delta_tokens = [[centroids[i][0]-centroids[i-1][0],centroids[i][1]-centroids[i-1][1],
                         bboxes[i][2]-bboxes[i-1][2],bboxes[i][3]-bboxes[i-1][3]]
                        for i in range(1,len(centroids))]
        
        gt_action = gt_tracks[tid]["action"] if tid in gt_tracks else None
        gt_sar = SAR_MAP.get(gt_action, None) if gt_action else None
        gt_fine = {"Standing":"standing","Sitting":"sitting","Walking":"walking","Running":"running","Lying":"lying_down"}.get(gt_action, None)
        
        tracks_data[tid] = {
            "raw_traj":raw_traj, "delta_tokens":delta_tokens, "centroids":centroids,
            "bboxes":bboxes, "frames":frames, "mean_size":mean_size, "mean_conf":mean_conf,
            "gt_action":gt_action, "gt_sar":gt_sar, "gt_fine":gt_fine, "n_frames":len(frames),
            "gt_priority":GT_PRIORITY.get(gt_fine,0.15) if gt_fine else 0.15,
        }
    timings["trajectory_ms"] = (time.time()-t2)/max(len(sampled_frames),1)*1000
    log(f'  Trajectories: {len(tracks_data)} valid tracks extracted')
    
    # ── STAGE 3a: TMS-12 + RF ────────────────────────────────────────
    t3 = time.time()
    for tid, td in tracks_data.items():
        feats = extract_tms12(td["raw_traj"])
        feats = [0 if (math.isnan(f) or math.isinf(f)) else f for f in feats]
        td["tms12_feats"] = feats
        pred = rf_model.predict([feats])[0]
        proba = rf_model.predict_proba([feats])[0]
        td["tms12_class"] = RF_CLASSES[pred]
        td["tms12_conf"] = float(max(proba))
        td["tms12_score"] = RF_PRIORITY[RF_CLASSES[pred]]
    pipeline_status["TMS-12"] = {"status":"RAN","note":"SMOTE-balanced RF, 200 trees"}
    timings["tms12_ms"] = (time.time()-t3)/max(len(tracks_data),1)*1000
    log(f'  TMS-12: classified {len(tracks_data)} tracks')
    
    # ── STAGE 3b: TrajMAE ────────────────────────────────────────────
    t4 = time.time()
    if trajmae_model is not None:
        for tid, td in tracks_data.items():
            dt = td["delta_tokens"]
            if len(dt) < 2:
                td["trajmae_class"] = "unknown"; td["trajmae_conf"] = 0; continue
            arr = np.zeros((50,4),dtype=np.float32)
            for j in range(min(len(dt),50)): arr[j]=dt[j]
            x = torch.tensor(arr).unsqueeze(0)
            with torch.no_grad():
                logits = trajmae_model(x, pretrain=False)
                probs = torch.softmax(logits,1)[0]
            pred = probs.argmax().item()
            td["trajmae_class"] = RF_CLASSES[pred]
            td["trajmae_conf"] = float(probs[pred].item())
            # Anomaly: reconstruction error
            try:
                recon, target, _ = trajmae_model(x, pretrain=True)
                td["trajmae_recon_error"] = float(nn.MSELoss()(recon, target).item())
            except:
                td["trajmae_recon_error"] = None
        pipeline_status["TrajMAE"] = {"status":"RAN","note":"Pre-trained encoder, classifier head"}
        pipeline_status["AnomalyDet"] = {"status":"RAN","note":"TrajMAE reconstruction error z-score"}
    else:
        for tid, td in tracks_data.items():
            td["trajmae_class"]="unknown"; td["trajmae_conf"]=0; td["trajmae_recon_error"]=None
        pipeline_status["TrajMAE"] = {"status":"SKIPPED","note":"Encoder weights failed to load"}
        pipeline_status["AnomalyDet"] = {"status":"SKIPPED","note":"Requires TrajMAE"}
    timings["trajmae_ms"] = (time.time()-t4)/max(len(tracks_data),1)*1000
    
    # ── STAGE 3c: SCTE ───────────────────────────────────────────────
    pipeline_status["SCTE"] = {"status":"SKIPPED","note":"Requires contrastive training on labelled track pairs; real evaluation pending"}
    for tid, td in tracks_data.items(): td["scte_embedding"] = None
    
    # ── STAGE 4: TCE v2 ──────────────────────────────────────────────
    t5 = time.time()
    for tid, td in tracks_data.items():
        score, state = tce_v2_score(td["raw_traj"])
        td["tce_score"] = score; td["tce_state"] = state
    pipeline_status["TCE"] = {"status":"RAN","note":"v2 with initial state assessment"}
    timings["tce_ms"] = (time.time()-t5)/max(len(tracks_data),1)*1000
    
    # ── STAGE 5: MViTv2-S (from cache) ────────────────────────────────
    t6 = time.time()
    mvit_cache_seq = mvit_cache.get(seq_name, {}) if mvit_cache else {}
    n_mvit_hit = 0; n_mvit_miss = 0
    for tid, td in tracks_data.items():
        cached = mvit_cache_seq.get(str(tid))
        if cached and cached.get("predicted_sar"):
            td["mvit_class_raw"] = cached["predicted_class"]
            td["mvit_class"] = cached["predicted_sar"]
            td["mvit_conf"] = cached["confidence"]
            n_mvit_hit += 1
        else:
            td["mvit_class"] = "unknown"; td["mvit_conf"] = 0
            td["mvit_class_raw"] = "no_cache"
            n_mvit_miss += 1
    if mvit_cache:
        pipeline_status["MViTv2-S"] = {"status":"RAN","note":f"Pre-computed cache, {n_mvit_hit} hits, {n_mvit_miss} misses"}
    else:
        pipeline_status["MViTv2-S"] = {"status":"SKIPPED","note":"No cache available"}
    timings["mvit_ms"] = (time.time()-t6)/max(len(tracks_data),1)*1000
    log(f'  MViTv2-S: {n_mvit_hit} from cache, {n_mvit_miss} missed')
    
    # ── STAGE 6: AAI-v2 Fusion ───────────────────────────────────────
    t7 = time.time()
    aai_used = "FALLBACK"
    for tid, td in tracks_data.items():
        size = td["mean_size"]; conf = td["mean_conf"]
        motion = np.mean([math.sqrt((td["raw_traj"][i][0]-td["raw_traj"][i-1][0])**2+
                 (td["raw_traj"][i][1]-td["raw_traj"][i-1][1])**2)
                 for i in range(1,len(td["raw_traj"]))]) if len(td["raw_traj"])>1 else 0
        
        if aai_model is not None:
            try:
                inp = torch.tensor([[size, conf, motion]]).float()
                with torch.no_grad():
                    w = aai_model(inp)[0]
                w_pixel = float(w[0]); w_traj = float(w[1])
                aai_used = "RAN"
            except:
                w_pixel = 0.1 if size < 50 else 0.6
                w_traj = 1.0 - w_pixel
        else:
            w_pixel = 0.1 if size < 50 else 0.6
            w_traj = 1.0 - w_pixel
        
        # Pixel score = priority from MViTv2 predicted class
        mvit_score = RF_PRIORITY.get(td["mvit_class"], 0.2)
        # Trajectory score = TMS-12 class priority modulated by TCE
        traj_score = td["tms12_score"] * (0.5 + 0.5 * td["tce_score"])
        
        td["aai_w_pixel"] = w_pixel; td["aai_w_traj"] = w_traj
        td["fused_score"] = w_pixel * mvit_score + w_traj * traj_score
    
    pipeline_status["AAI-v2"] = {"status":aai_used,"note":"MLP meta-classifier" if aai_used=="RAN" else "Rule-based fallback (size<50 → w_traj=0.9)"}
    timings["aai_ms"] = (time.time()-t7)/max(len(tracks_data),1)*1000
    
    # ── STAGE 7: EMI ─────────────────────────────────────────────────
    t8 = time.time()
    emi_result, emi_status = compute_emi(frame_dir, sampled_frames)
    if emi_result:
        attn = emi_result["attention_score"]
        for tid, td in tracks_data.items():
            td["emi_phase"] = emi_result["phase"]
            td["emi_modifier"] = 1.0 + 0.3 * attn
            td["final_priority"] = td["fused_score"] * td["emi_modifier"]
        pipeline_status["EMI"] = {"status":"RAN","note":f"Phase={emi_result['phase']}, hover={emi_result['hover_index']:.2f}"}
    else:
        for tid, td in tracks_data.items():
            td["emi_phase"] = "UNKNOWN"; td["emi_modifier"] = 1.0
            td["final_priority"] = td["fused_score"]
        pipeline_status["EMI"] = {"status":emi_status,"note":"Homography computation issue"}
    timings["emi_ms"] = (time.time()-t8)/max(len(sampled_frames),1)*1000
    
    # ── STAGE 8: Priority Ranker ─────────────────────────────────────
    t9 = time.time()
    ranked = sorted(tracks_data.items(), key=lambda x: -x[1]["final_priority"])
    pipeline_status["Ranker"] = {"status":"RAN","note":"Sort by final_priority descending"}
    timings["ranker_ms"] = (time.time()-t9)*1000
    
    # ── EVALUATION ───────────────────────────────────────────────────
    # NDCG and Recall@3
    ranked_gt_priorities = [td["gt_priority"] for _,td in ranked]
    ndcg3 = ndcg(ranked_gt_priorities, 3)
    ndcg5 = ndcg(ranked_gt_priorities, 5)
    
    has_lying = any(td["gt_fine"]=="lying_down" for _,td in ranked)
    top3_labels = [td["gt_fine"] for _,td in ranked[:3]]
    recall3_lying = 1.0 if "lying_down" in top3_labels else 0.0 if has_lying else float('nan')
    
    # Anomaly detection: flag tracks with high recon error
    recon_errors = [td["trajmae_recon_error"] for _,td in ranked if td["trajmae_recon_error"] is not None]
    anomaly_threshold = np.mean(recon_errors) + 2*np.std(recon_errors) if recon_errors else 999
    for tid, td in tracks_data.items():
        td["is_anomaly"] = td.get("trajmae_recon_error",0) is not None and td.get("trajmae_recon_error",0) > anomaly_threshold if td.get("trajmae_recon_error") else False
    
    # Build ranked output
    ranked_output = []
    for rank, (tid, td) in enumerate(ranked, 1):
        entry = {
            "rank": rank, "track_id": int(tid),
            "predicted_action": td["tms12_class"], "tms12_conf": round(td["tms12_conf"],3),
            "trajmae_pred": td["trajmae_class"], "trajmae_conf": round(td.get("trajmae_conf",0),3),
            "mvit_pred": td["mvit_class"], "mvit_raw": td.get("mvit_class_raw",""),
            "mvit_conf": round(td.get("mvit_conf",0),3),
            "tce_state": td["tce_state"], "tce_score": round(td["tce_score"],3),
            "aai_w_pixel": round(td["aai_w_pixel"],3), "aai_w_traj": round(td["aai_w_traj"],3),
            "emi_phase": td["emi_phase"], "emi_modifier": round(td.get("emi_modifier",1),3),
            "fused_score": round(td["fused_score"],4),
            "final_priority": round(td["final_priority"],4),
            "person_size_px": round(td["mean_size"],1),
            "n_frames": td["n_frames"],
            "gt_action": td["gt_action"], "gt_sar": td["gt_sar"],
            "is_anomaly": bool(td.get("is_anomaly", False)),
        }
        ranked_output.append(entry)
    
    # Print ranked table
    log(f'\n  RANKED OUTPUT ({seq_name}):')
    log(f'  {"Rank":>4} {"TID":>4} {"Size":>5} {"TMS12":>10} {"TrajMAE":>10} {"MViTv2":>10} {"TCE":>8} {"AAI":>10} {"Final":>7} {"GT":>10}')
    for e in ranked_output[:10]:
        log(f'  {e["rank"]:4d} {e["track_id"]:4d} {e["person_size_px"]:5.0f} {e["predicted_action"]:>10} {e["trajmae_pred"]:>10} {e["mvit_pred"]:>10} {e["tce_state"]:>8} {e["aai_w_traj"]:.2f}/{e["aai_w_pixel"]:.2f} {e["final_priority"]:7.4f} {e["gt_action"] or "?":>10}')
    
    seq_results = {
        "sequence": seq_name, "n_tracks": len(tracks_data),
        "n_yolo_detections": sum(len(d) for d in detections_per_frame.values()),
        "n_matched": matched_count, "n_unmatched": unmatched_count,
        "ndcg3": float(ndcg3), "ndcg5": float(ndcg5),
        "recall3_lying": float(recall3_lying) if not math.isnan(recall3_lying) else None,
        "has_lying_gt": has_lying,
        "timings_ms_per_frame": timings,
        "ranked_output": ranked_output,
    }
    
    # ── TIMELINE FIGURE ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(6, len(tracks_data)*0.3+2)))
    color_map = {"lying_down":"#e74c3c","stationary":"#3498db","walking":"#2ecc71","running":"#f39c12","unknown":"grey"}
    
    sorted_tracks = sorted(tracks_data.items(), key=lambda x: -x[1]["final_priority"])
    for row, (tid, td) in enumerate(sorted_tracks):
        f_start = min(td["frames"]); f_end = max(td["frames"])
        color = color_map.get(td["tms12_class"], "grey")
        alpha = 0.3 + 0.7 * min(td["final_priority"], 1.0)
        ax.barh(row, f_end-f_start, left=f_start, height=0.7, color=color, alpha=alpha,
                edgecolor='black', linewidth=0.3)
        # Priority label
        ax.text(f_end+5, row, f'{td["final_priority"]:.3f}', va='center', fontsize=7)
        # Star if lying_down GT
        if td["gt_fine"] == "lying_down":
            ax.plot(f_start-10, row, marker='*', color='red', markersize=12, zorder=5)
            ax.text(f_start-25, row, '★', fontsize=10, color='red', va='center', ha='center')
    
    ax.set_yticks(range(len(sorted_tracks)))
    ax.set_yticklabels([f'T{tid}' for tid,_ in sorted_tracks], fontsize=7)
    ax.set_xlabel('Frame Number', fontsize=11)
    ax.set_title(f'SARTriage End-to-End Output — Sequence {seq_name}\n'
                 f'NDCG@3={ndcg3:.3f} | Lying R@3={"Yes" if recall3_lying==1 else "No" if has_lying else "N/A"} | '
                 f'{len(tracks_data)} tracks', fontsize=12)
    ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l,c in color_map.items() if l != "unknown"]
    legend_elements.append(plt.Line2D([0],[0],marker='*',color='red',linestyle='None',markersize=10,label='GT lying_down'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"e2e_timeline_{seq_name.replace('.','_')}.png"), dpi=200)
    plt.close()
    
    return seq_results


def main():
    with open('/tmp/e2e_progress.txt','w') as f: f.write('E2E Pipeline Starting\n')
    
    # Load all models
    log('Loading models...')
    yolo_model, yolo_status = load_yolo()
    log(f'  YOLO: {yolo_status}')
    
    rf_model, rf_status = load_rf()
    log(f'  RF: {rf_status}')
    
    mvit_cache, mvit_status = load_mvit_cache()
    log(f'  MViTv2-S: {mvit_status}')
    
    aai_model, aai_status = load_aai()
    log(f'  AAI-v2: {aai_status}')
    
    trajmae_model, trajmae_status = load_trajmae()
    log(f'  TrajMAE: {trajmae_status}')
    
    # Pipeline status (shared across sequences)
    pipeline_status = {}
    
    # Run on 3 sequences
    all_results = []
    for seq in SEQUENCES:
        result = run_pipeline_on_sequence(seq, yolo_model, rf_model, mvit_cache,
                                          aai_model, trajmae_model, pipeline_status)
        if result:
            # Save per-sequence
            with open(os.path.join(OUT_DIR, f"e2e_{seq.replace('.','_')}_results.json"), "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)
    
    # Aggregate summary
    summary = {
        "n_sequences": len(all_results),
        "sequences": [r["sequence"] for r in all_results],
        "mean_ndcg3": float(np.mean([r["ndcg3"] for r in all_results])),
        "mean_ndcg5": float(np.mean([r["ndcg5"] for r in all_results])),
        "lying_recall3_scenes": [r["recall3_lying"] for r in all_results if r["recall3_lying"] is not None],
        "total_tracks": sum(r["n_tracks"] for r in all_results),
        "total_yolo_dets": sum(r["n_yolo_detections"] for r in all_results),
        "total_matched": sum(r["n_matched"] for r in all_results),
        "pipeline_status": pipeline_status,
        "mean_timings_ms": {
            k: float(np.mean([r["timings_ms_per_frame"].get(k,0) for r in all_results]))
            for k in ["yolo_ms","tracking_ms","trajectory_ms","tms12_ms","trajmae_ms","tce_ms","mvit_ms","aai_ms","emi_ms","ranker_ms"]
        }
    }
    
    with open(os.path.join(OUT_DIR, "e2e_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(os.path.join(OUT_DIR, "e2e_pipeline_report.json"), "w") as f:
        json.dump(pipeline_status, f, indent=2)
    
    # Print pipeline status table
    log('\n' + '='*80)
    log('PIPELINE INTEGRATION STATUS')
    log('='*80)
    log(f'{"Component":<15} {"Status":<12} {"Notes"}')
    log('-'*80)
    for comp in ["YOLO","ByteTrack","TMS-12","TrajMAE","SCTE","TCE","MViTv2-S","AnomalyDet","AAI-v2","EMI","Ranker"]:
        ps = pipeline_status.get(comp, {"status":"UNKNOWN","note":""})
        log(f'{comp:<15} {ps["status"]:<12} {ps["note"]}')
    
    log(f'\nMean NDCG@3: {summary["mean_ndcg3"]:.4f}')
    log(f'Mean NDCG@5: {summary["mean_ndcg5"]:.4f}')
    log(f'Lying R@3: {summary["lying_recall3_scenes"]}')
    log(f'\nTiming breakdown (ms/frame):')
    for k,v in summary["mean_timings_ms"].items():
        log(f'  {k:<20s}: {v:.2f}')
    
    log('\nE2E PIPELINE COMPLETE.')

if __name__ == "__main__":
    main()
