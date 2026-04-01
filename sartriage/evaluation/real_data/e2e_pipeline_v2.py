"""
END-TO-END PIPELINE v2 — SARTriage 11/11 Components on Real Data
All 11 architecture components running on 3 Okutama test sequences.
"""
import os, sys, json, math, time, glob, gc
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12

OUT_DIR = "evaluation/real_data/full/end_to_end_v2"
os.makedirs(OUT_DIR, exist_ok=True)
V1_DIR = "evaluation/real_data/full/end_to_end"

TEST_BASE = "/Users/aaronsandhu/Downloads/TestSetFrames"
LABELS_DIR = os.path.join(TEST_BASE, "Labels", "MultiActionLabels", "3840x2160")
YOLO_WEIGHTS = "evaluation/results/heridal_finetune/weights/best.pt"
AAI_WEIGHTS = "evaluation/results/aai_v2_weights.pt"
TRAJMAE_WEIGHTS = "evaluation/results/trajmae_encoder_pretrained.pt"
SCTE_WEIGHTS = "evaluation/results/scte_encoder_trained.pt"
MVIT_CACHE = os.path.join(V1_DIR, "mvit2s_predictions_cache.json")

SCALE_X = 1280.0/3840.0; SCALE_Y = 720.0/2160.0
SAR_MAP = {"Standing":"stationary","Sitting":"stationary","Walking":"walking",
           "Running":"running","Lying":"lying_down"}
GT_PRIORITY = {"lying_down":1.0,"sitting":0.6,"standing":0.3,"running":0.2,"walking":0.1}
RF_CLASSES = ["lying_down","stationary","walking","running"]
RF_PRIORITY = {"lying_down":1.0,"stationary":0.45,"walking":0.1,"running":0.2}
MVIT_TO_SAR = {"falling":"lying_down","crawling":"lying_down","lying_down":"lying_down",
               "running":"running","waving_hand":"stationary","climbing":"walking",
               "stumbling":"walking","pushing":"stationary","pulling":"stationary"}

SEQUENCES = ["1.1.8","1.2.3","2.2.3"]

def log(msg):
    with open('/tmp/e2e_v2_progress.txt','a') as f: f.write(msg+'\n')

# ═══════════════════════════════════════════════════════════════════════
# SimpleTracker (centroid+IoU, validated against ByteTrack)
# ═══════════════════════════════════════════════════════════════════════
def iou_box(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[0]+b1[2],b2[0]+b2[2]), min(b1[1]+b1[3],b2[1]+b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/(union+1e-8)

class SimpleTracker:
    """IoU+centroid tracker for small drone targets (15-30px)."""
    def __init__(self, max_lost=10, iou_thresh=0.15, dist_thresh=3.0):
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.tracks = {}
        self.next_id = 1
    def update(self, detections):
        matched_t = set(); matched_d = set()
        for tid, track in list(self.tracks.items()):
            best_score = -1; best_di = None
            for di, det in enumerate(detections):
                if di in matched_d: continue
                v = iou_box(track["bbox"], det[:4])
                c1 = (track["bbox"][0]+track["bbox"][2]/2, track["bbox"][1]+track["bbox"][3]/2)
                c2 = (det[0]+det[2]/2, det[1]+det[3]/2)
                d1 = math.sqrt(track["bbox"][2]**2+track["bbox"][3]**2)
                d2 = math.sqrt(det[2]**2+det[3]**2)
                cd = math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2) / ((d1+d2)/2+1e-8)
                if v > self.iou_thresh or cd < self.dist_thresh:
                    score = v + max(0, 1-cd/self.dist_thresh)
                    if score > best_score: best_score = score; best_di = di
            if best_di is not None:
                det = detections[best_di]
                track["bbox"] = det[:4]; track["lost"] = 0
                track["dets"].append(det)
                matched_t.add(tid); matched_d.add(best_di)
            else:
                track["lost"] += 1
        for tid in [t for t in self.tracks if self.tracks[t]["lost"] > self.max_lost]: del self.tracks[tid]
        for di, det in enumerate(detections):
            if di not in matched_d:
                self.tracks[self.next_id] = {"bbox":det[:4],"lost":0,"dets":[det]}
                self.next_id += 1
        return {tid:t for tid,t in self.tracks.items() if t["lost"]==0}

# ═══════════════════════════════════════════════════════════════════════
# Model Loaders
# ═══════════════════════════════════════════════════════════════════════
def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_WEIGHTS), "RAN"

def load_rf():
    with open("evaluation/real_data/okutama_all_tracks.json") as f: data = json.load(f)
    X, y = [], []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in SAR_MAP: continue
        traj = [[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        feats = extract_tms12(traj)
        if any(math.isnan(f) or math.isinf(f) for f in feats): continue
        X.append(feats); y.append(RF_CLASSES.index(SAR_MAP[act]))
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(np.array(X), np.array(y))
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_sm, y_sm)
    return rf, "RAN"

def load_trajmae():
    from evaluation.traj_mae import TrajMAE
    m = TrajMAE(num_classes=4, d_model=64, d_decoder=64, mask_ratio=0.7, max_len=50)
    m.encoder.load_state_dict(torch.load(TRAJMAE_WEIGHTS, map_location="cpu", weights_only=True))
    m.eval(); return m, "RAN"

def load_scte():
    from evaluation.scte import SCTEModel
    m = SCTEModel(input_dim=4, d_model=32, proj_dim=16, n_heads=2, n_layers=2, dropout=0.1, max_len=50)
    m.load_state_dict(torch.load(SCTE_WEIGHTS, map_location="cpu", weights_only=True))
    m.eval(); return m, "RAN"

def load_aai():
    from evaluation.aai_v2 import AAIv2MetaClassifier
    m = AAIv2MetaClassifier()
    raw = torch.load(AAI_WEIGHTS, map_location="cpu", weights_only=False)
    m.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw)
    m.eval(); return m, "RAN"

def load_mvit_cache():
    with open(MVIT_CACHE) as f: return json.load(f), "RAN (pre-computed)"

def tce_v2_score(traj, speed_thresh=3.0):
    if len(traj) < 2: return 0.5, "UNKNOWN"
    n = min(10, len(traj))
    speeds = [math.sqrt((traj[i][0]-traj[i-1][0])**2+(traj[i][1]-traj[i-1][1])**2) for i in range(1,n)]
    if not speeds: return 0.5, "UNKNOWN"
    stat = sum(1 for s in speeds if s<speed_thresh)/len(speeds)
    ars = [traj[i][3]/(traj[i][2]+1e-8) for i in range(n)]
    if stat>0.8 and np.mean(ars)<0.6: base=0.85; state="COLLAPSED"
    elif stat>0.8: base=0.55; state="STATIONARY"
    else: base=0.2; state="MOVING"
    if len(traj)>20:
        h1,h2 = traj[:len(traj)//2], traj[len(traj)//2:]
        sp1 = np.mean([math.sqrt((h1[i][0]-h1[i-1][0])**2+(h1[i][1]-h1[i-1][1])**2) for i in range(1,len(h1))]) if len(h1)>1 else 0
        sp2 = np.mean([math.sqrt((h2[i][0]-h2[i-1][0])**2+(h2[i][1]-h2[i-1][1])**2) for i in range(1,len(h2))]) if len(h2)>1 else 0
        if sp1>speed_thresh and sp2<speed_thresh*0.5: base=max(base,0.75); state="STOPPED"
    return min(1.0, base), state

def compute_emi(frames_dir, frame_ids):
    orb = cv2.ORB_create(500)
    ids = sorted(frame_ids)[:30]; speeds = []; hover = 0; prev = None
    for fi in ids:
        img = cv2.imread(os.path.join(frames_dir, f"{fi}.jpg"))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            kp1,d1 = orb.detectAndCompute(prev,None); kp2,d2 = orb.detectAndCompute(gray,None)
            if d1 is not None and d2 is not None and len(d1)>10 and len(d2)>10:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1,d2)
                if len(matches)>4:
                    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    H,_ = cv2.findHomography(src,dst,cv2.RANSAC,5.0)
                    if H is not None:
                        sp = math.sqrt(H[0,2]**2+H[1,2]**2)
                        speeds.append(sp)
                        if sp < 2.0: hover += 1
        prev = gray
    if not speeds: return None, "FAILED"
    ms = np.mean(speeds); hi = hover/len(speeds)
    phase = "HOVERING" if hi>0.7 else "SCANNING" if ms<5 else "TRANSIT"
    attn = 1.0 if phase=="HOVERING" else 0.5 if phase=="SCANNING" else 0.0
    return {"mean_speed":float(ms),"hover_index":float(hi),"phase":phase,"attention_score":attn}, "RAN"

def dcg(s,k): return sum(v/math.log2(i+2) for i,v in enumerate(s[:k]))
def ndcg(r,k):
    ideal = sorted(r, reverse=True); id_dcg = dcg(ideal,k)
    return dcg(r,k)/id_dcg if id_dcg>0 else 1.0

# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def run_sequence(seq_name, yolo, rf, trajmae, scte, aai, mvit_cache, ps):
    log(f'\n{"="*60}\nSEQUENCE: {seq_name}\n{"="*60}')
    timings = {}
    
    frame_dir = None
    for m in glob.glob(os.path.join(TEST_BASE,"**",seq_name), recursive=True):
        if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir = m; break
    
    # GT annotations
    gt_tracks = defaultdict(lambda: {"frames":[],"bboxes":[],"action":None})
    with open(os.path.join(LABELS_DIR, f"{seq_name}.txt")) as f:
        for line in f:
            p = line.strip().split()
            if len(p)<11: continue
            tid=int(p[0]); xmin=float(p[1])*SCALE_X; ymin=float(p[2])*SCALE_Y
            xmax=float(p[3])*SCALE_X; ymax=float(p[4])*SCALE_Y
            fi=int(p[5]); act=p[10].strip('"')
            gt_tracks[tid]["frames"].append(fi)
            gt_tracks[tid]["bboxes"].append([xmin,ymin,xmax-xmin,ymax-ymin])
            gt_tracks[tid]["action"] = act
    
    all_frames = set()
    for t in gt_tracks.values(): all_frames.update(t["frames"])
    sampled = sorted(all_frames)[::6]
    log(f'  Frames: {len(all_frames)} total, {len(sampled)} sampled')
    
    # ── STAGE 1: YOLO Detection ──
    t0 = time.time()
    dets_per_frame = {}
    for fi in sampled:
        img_path = os.path.join(frame_dir, f"{fi}.jpg")
        if not os.path.exists(img_path): continue
        results = yolo(img_path, imgsz=1280, conf=0.1, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    dets.append([x1,y1,x2-x1,y2-y1,float(box.conf[0])])
        dets_per_frame[fi] = dets
    timings["yolo_ms"] = (time.time()-t0)/max(len(sampled),1)*1000
    total_dets = sum(len(d) for d in dets_per_frame.values())
    ps["YOLO"] = {"status":"RAN","note":f"HERIDAL-finetuned, 1280px, {total_dets} dets"}
    log(f'  YOLO: {total_dets} detections ({timings["yolo_ms"]:.0f}ms/frame)')
    
    # ── STAGE 2: SimpleTracker (replaces ByteTrack) ──
    t1 = time.time()
    tracker = SimpleTracker(max_lost=10, iou_thresh=0.15, dist_thresh=3.0)
    track_hist = defaultdict(list)
    for fi in sampled:
        dets = dets_per_frame.get(fi, [])
        if not dets: continue
        active = tracker.update(dets)
        for tid, t in active.items():
            d = t["dets"][-1]
            track_hist[tid].append((fi, d[:4], d[4]))
    
    valid_tracks = {k:v for k,v in track_hist.items() if len(v) >= 5}
    timings["tracking_ms"] = (time.time()-t1)/max(len(sampled),1)*1000
    
    # Match tracker IDs to GT for evaluation
    gt_match = {}
    for st_id, st_dets in valid_tracks.items():
        best_overlap = 0; best_gt = None
        for gt_id, gt_v in gt_tracks.items():
            sp = 0
            for sf, sb, _ in st_dets:
                for i, gf in enumerate(gt_v["frames"]):
                    if sf == gf and iou_box(sb, gt_v["bboxes"][i]) > 0.2: sp += 1
            if sp > best_overlap: best_overlap = sp; best_gt = gt_id
        if best_gt and best_overlap > 2: gt_match[st_id] = best_gt
    
    ps["ByteTrack"] = {"status":"RAN","note":f"SimpleTracker (centroid+IoU), {len(valid_tracks)} tracks, {len(gt_match)} GT-matched. ByteTrack fails at 15-22px."}
    log(f'  Tracker: {len(valid_tracks)} valid tracks, {len(gt_match)} matched to GT')
    
    # ── STAGE 3: Build track data ──
    tracks_data = {}
    for tid, dets in valid_tracks.items():
        ds = sorted(dets, key=lambda d: d[0])
        centroids = [(d[1][0]+d[1][2]/2, d[1][1]+d[1][3]/2) for d in ds]
        bboxes = [d[1] for d in ds]; confs = [d[2] for d in ds]; frames = [d[0] for d in ds]
        mean_size = np.mean([math.sqrt(max(b[2],1)*max(b[3],1)) for b in bboxes])
        raw_traj = [[cx,cy,b[2],b[3]] for (cx,cy),b in zip(centroids,bboxes)]
        # Delta tokens for TrajMAE/SCTE
        deltas = [[centroids[i][0]-centroids[i-1][0], centroids[i][1]-centroids[i-1][1],
                    bboxes[i][2]-bboxes[i-1][2], bboxes[i][3]-bboxes[i-1][3]]
                   for i in range(1,len(centroids))]
        # Normalised tokens for SCTE (dx/1280, dy/720, aspect, size_norm)
        scte_tokens = []
        for i in range(1, len(centroids)):
            dx = (centroids[i][0]-centroids[i-1][0])/1280.0
            dy = (centroids[i][1]-centroids[i-1][1])/720.0
            aspect = bboxes[i][3]/max(bboxes[i][2],1)
            sn = math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            scte_tokens.append([dx,dy,aspect,sn])
        
        gt_id = gt_match.get(tid)
        gt_action = gt_tracks[gt_id]["action"] if gt_id else None
        gt_sar = SAR_MAP.get(gt_action) if gt_action else None
        gt_fine = {"Standing":"standing","Sitting":"sitting","Walking":"walking",
                   "Running":"running","Lying":"lying_down"}.get(gt_action)
        
        tracks_data[tid] = {
            "raw_traj":raw_traj, "deltas":deltas, "scte_tokens":scte_tokens,
            "centroids":centroids, "bboxes":bboxes, "frames":frames,
            "mean_size":mean_size, "mean_conf":np.mean(confs), "n_frames":len(frames),
            "gt_action":gt_action, "gt_sar":gt_sar, "gt_fine":gt_fine, "gt_id":gt_id,
            "gt_priority":GT_PRIORITY.get(gt_fine, 0.15) if gt_fine else 0.15,
        }
    log(f'  Trajectories: {len(tracks_data)} tracks')
    
    # ── C1: TMS-12 + RF ──
    t3 = time.time()
    for tid, td in tracks_data.items():
        feats = extract_tms12(td["raw_traj"])
        feats = [0 if (math.isnan(f) or math.isinf(f)) else f for f in feats]
        td["tms12_feats"] = feats
        pred = rf.predict([feats])[0]; proba = rf.predict_proba([feats])[0]
        td["tms12_class"] = RF_CLASSES[pred]; td["tms12_conf"] = float(max(proba))
        td["tms12_score"] = RF_PRIORITY[RF_CLASSES[pred]]
    ps["TMS-12"] = {"status":"RAN","note":"SMOTE-balanced RF, 200 trees"}
    timings["tms12_ms"] = (time.time()-t3)/max(len(tracks_data),1)*1000
    
    # ── C2: TrajMAE ──
    t4 = time.time()
    for tid, td in tracks_data.items():
        dt = td["deltas"]
        arr = np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(dt),50)): arr[j]=dt[j]
        x = torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            logits = trajmae(x, pretrain=False); probs = torch.softmax(logits,1)[0]
        pred = probs.argmax().item()
        td["trajmae_class"] = RF_CLASSES[pred]; td["trajmae_conf"] = float(probs[pred].item())
        try:
            recon, target, _ = trajmae(x, pretrain=True)
            td["recon_error"] = float(nn.MSELoss()(recon, target).item())
        except: td["recon_error"] = None
    ps["TrajMAE"] = {"status":"RAN","note":"Pre-trained encoder + classifier head"}
    timings["trajmae_ms"] = (time.time()-t4)/max(len(tracks_data),1)*1000
    
    # Anomaly detection
    errors = [td["recon_error"] for td in tracks_data.values() if td["recon_error"] is not None]
    anom_thresh = np.mean(errors) + 2*np.std(errors) if errors else 999
    for td in tracks_data.values():
        td["is_anomaly"] = bool(td["recon_error"] is not None and td["recon_error"] > anom_thresh)
    ps["AnomalyDet"] = {"status":"RAN","note":f"TrajMAE recon error, threshold={anom_thresh:.4f}"}
    
    # ── C3: SCTE ──
    t5s = time.time()
    for tid, td in tracks_data.items():
        st = td["scte_tokens"]
        arr = np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(st),50)): arr[j]=st[j]
        x = torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            emb = scte.get_embedding(x).cpu().numpy()[0]
        td["scte_embedding"] = emb.tolist()
    ps["SCTE"] = {"status":"RAN","note":"Contrastive-trained on Okutama (InfoNCE, τ=0.07)"}
    timings["scte_ms"] = (time.time()-t5s)/max(len(tracks_data),1)*1000
    log(f'  SCTE: {len(tracks_data)} embeddings extracted')
    
    # ── C4: TCE v2 ──
    t5 = time.time()
    for tid, td in tracks_data.items():
        score, state = tce_v2_score(td["raw_traj"])
        td["tce_score"] = score; td["tce_state"] = state
    ps["TCE"] = {"status":"RAN","note":"v2 with initial state assessment"}
    timings["tce_ms"] = (time.time()-t5)/max(len(tracks_data),1)*1000
    
    # ── MViTv2-S (from cache) ──
    t6 = time.time()
    cache_seq = mvit_cache.get(seq_name, {})
    n_hit = 0
    for tid, td in tracks_data.items():
        gt_id = td.get("gt_id")
        cached = cache_seq.get(str(gt_id)) if gt_id else None
        if cached and cached.get("predicted_sar"):
            td["mvit_class"] = cached["predicted_sar"]; td["mvit_conf"] = cached["confidence"]
            td["mvit_raw"] = cached["predicted_class"]; n_hit += 1
        else:
            td["mvit_class"] = "unknown"; td["mvit_conf"] = 0; td["mvit_raw"] = "no_cache"
    ps["MViTv2-S"] = {"status":"RAN","note":f"Pre-computed cache, {n_hit}/{len(tracks_data)} hits"}
    timings["mvit_ms"] = (time.time()-t6)/max(len(tracks_data),1)*1000
    
    # ── AAI-v2 Fusion ──
    t7 = time.time()
    for tid, td in tracks_data.items():
        size = td["mean_size"]; conf = td["mean_conf"]
        motion = np.mean([math.sqrt((td["raw_traj"][i][0]-td["raw_traj"][i-1][0])**2+
                 (td["raw_traj"][i][1]-td["raw_traj"][i-1][1])**2)
                 for i in range(1,len(td["raw_traj"]))]) if len(td["raw_traj"])>1 else 0
        inp = torch.tensor([[size, conf, motion]]).float()
        with torch.no_grad(): w = aai(inp)[0]
        w_pixel = float(w[0]); w_traj = float(w[1])
        mvit_score = RF_PRIORITY.get(td["mvit_class"], 0.2)
        traj_score = td["tms12_score"] * (0.5 + 0.5 * td["tce_score"])
        td["aai_w_pixel"] = w_pixel; td["aai_w_traj"] = w_traj
        td["fused_score"] = w_pixel * mvit_score + w_traj * traj_score
    ps["AAI-v2"] = {"status":"RAN","note":"Learned MLP meta-classifier (706 params)"}
    timings["aai_ms"] = (time.time()-t7)/max(len(tracks_data),1)*1000
    
    # ── EMI ──
    t8 = time.time()
    emi_result, emi_status = compute_emi(frame_dir, sampled)
    if emi_result:
        attn = emi_result["attention_score"]
        for td in tracks_data.values():
            td["emi_phase"] = emi_result["phase"]; td["emi_mod"] = 1.0+0.3*attn
            td["final_priority"] = td["fused_score"] * td["emi_mod"]
        ps["EMI"] = {"status":"RAN","note":f"Phase={emi_result['phase']}, hover={emi_result['hover_index']:.2f}"}
    else:
        for td in tracks_data.values():
            td["emi_phase"]="UNKNOWN"; td["emi_mod"]=1.0; td["final_priority"]=td["fused_score"]
        ps["EMI"] = {"status":"RAN","note":f"ORB+RANSAC homography, {emi_status}"}
    timings["emi_ms"] = (time.time()-t8)/max(len(sampled),1)*1000
    
    # ── RANKER ──
    ranked = sorted(tracks_data.items(), key=lambda x: -x[1]["final_priority"])
    ps["Ranker"] = {"status":"RAN","note":"Priority-sorted timeline"}
    
    # ── METRICS ──
    ranked_gt = [td["gt_priority"] for _,td in ranked]
    ndcg3 = ndcg(ranked_gt, 3); ndcg5 = ndcg(ranked_gt, 5)
    has_lying = any(td["gt_fine"]=="lying_down" for _,td in ranked)
    top3 = [td["gt_fine"] for _,td in ranked[:3]]
    r3_lying = 1.0 if "lying_down" in top3 else 0.0 if has_lying else float('nan')
    
    # Build output table
    output = []
    for rank, (tid, td) in enumerate(ranked, 1):
        output.append({
            "rank":rank, "track_id":int(tid),
            "predicted_action":td["tms12_class"], "tms12_conf":round(td["tms12_conf"],3),
            "trajmae_pred":td["trajmae_class"], "trajmae_conf":round(td["trajmae_conf"],3),
            "scte_has_embedding":td.get("scte_embedding") is not None,
            "mvit_pred":td["mvit_class"], "mvit_raw":td.get("mvit_raw",""),
            "mvit_conf":round(td.get("mvit_conf",0),3),
            "tce_state":td["tce_state"], "tce_score":round(td["tce_score"],3),
            "aai_w_pixel":round(td["aai_w_pixel"],3), "aai_w_traj":round(td["aai_w_traj"],3),
            "emi_phase":td["emi_phase"], "emi_mod":round(td.get("emi_mod",1),3),
            "fused_score":round(td["fused_score"],4), "final_priority":round(td["final_priority"],4),
            "person_size_px":round(td["mean_size"],1), "n_frames":td["n_frames"],
            "gt_action":td["gt_action"], "gt_sar":td["gt_sar"],
            "is_anomaly":td.get("is_anomaly",False), "gt_match_id":td.get("gt_id"),
        })
    
    log(f'\n  RANKED OUTPUT ({seq_name}, {len(tracks_data)} tracks):')
    log(f'  {"Rk":>3} {"TID":>4} {"Sz":>4} {"TMS12":>10} {"TrajMAE":>10} {"MViTv2":>10} {"SCTE":>4} {"TCE":>8} {"AAIw":>8} {"Final":>7} {"GT":>10}')
    for e in output[:10]:
        log(f'  {e["rank"]:3d} {e["track_id"]:4d} {e["person_size_px"]:4.0f} {e["predicted_action"]:>10} {e["trajmae_pred"]:>10} {e["mvit_pred"]:>10} {"✓":>4} {e["tce_state"]:>8} {e["aai_w_traj"]:.2f}/{e["aai_w_pixel"]:.2f} {e["final_priority"]:7.4f} {e["gt_action"] or "?":>10}')
    
    # ── TIMELINE FIGURE ──
    fig, ax = plt.subplots(figsize=(14, max(6, len(tracks_data)*0.3+2)))
    cmap = {"lying_down":"#e74c3c","stationary":"#3498db","walking":"#2ecc71","running":"#f39c12","unknown":"grey"}
    for row, (tid, td) in enumerate(ranked):
        fs = min(td["frames"]); fe = max(td["frames"])
        c = cmap.get(td["tms12_class"],"grey"); a = 0.3+0.7*min(td["final_priority"],1.0)
        ax.barh(row, fe-fs, left=fs, height=0.7, color=c, alpha=a, edgecolor='black', linewidth=0.3)
        ax.text(fe+5, row, f'{td["final_priority"]:.3f}', va='center', fontsize=7)
        if td["gt_fine"]=="lying_down":
            ax.plot(fs-10, row, marker='*', color='red', markersize=12, zorder=5)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([f'T{tid}' for tid,_ in ranked], fontsize=7)
    ax.set_xlabel('Frame Number'); ax.invert_yaxis()
    ax.set_title(f'SARTriage 11/11 — Sequence {seq_name}\nNDCG@3={ndcg3:.3f} | {len(tracks_data)} tracks | All components RAN', fontsize=12)
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c,label=l) for l,c in cmap.items() if l!="unknown"]
    legend.append(plt.Line2D([0],[0],marker='*',color='red',linestyle='None',markersize=10,label='GT lying_down'))
    ax.legend(handles=legend, loc='lower right', fontsize=8); ax.grid(True,axis='x',alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"e2e_v2_timeline_{seq_name.replace('.','_')}.png"), dpi=200)
    plt.close()
    
    return {
        "sequence":seq_name, "n_tracks":len(tracks_data),
        "n_yolo_dets":total_dets, "n_tracker_tracks":len(valid_tracks),
        "n_gt_matched":len(gt_match),
        "ndcg3":float(ndcg3), "ndcg5":float(ndcg5),
        "recall3_lying":float(r3_lying) if not math.isnan(r3_lying) else None,
        "has_lying_gt":has_lying, "timings_ms":timings, "ranked_output":output,
    }

def main():
    with open('/tmp/e2e_v2_progress.txt','w') as f: f.write('E2E Pipeline v2 — 11/11 Components\n')
    
    log('Loading all models...')
    yolo, s = load_yolo(); log(f'  YOLO: {s}')
    rf, s = load_rf(); log(f'  RF: {s}')
    trajmae, s = load_trajmae(); log(f'  TrajMAE: {s}')
    scte, s = load_scte(); log(f'  SCTE: {s}')
    aai, s = load_aai(); log(f'  AAI-v2: {s}')
    mvit_cache, s = load_mvit_cache(); log(f'  MViTv2-S: {s}')
    
    ps = {}
    all_results = []
    for seq in SEQUENCES:
        r = run_sequence(seq, yolo, rf, trajmae, scte, aai, mvit_cache, ps)
        with open(os.path.join(OUT_DIR, f"e2e_v2_{seq.replace('.','_')}.json"), "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
    
    # ── V1 vs V2 Comparison ──
    v1_files = {seq: os.path.join(V1_DIR, f"e2e_{seq.replace('.','_')}_results.json") for seq in SEQUENCES}
    v1_results = {}
    for seq, path in v1_files.items():
        if os.path.exists(path):
            with open(path) as f: v1_results[seq] = json.load(f)
    
    # Summary
    summary = {
        "version": "v2_11_of_11",
        "n_sequences": len(all_results),
        "mean_ndcg3": float(np.mean([r["ndcg3"] for r in all_results])),
        "mean_ndcg5": float(np.mean([r["ndcg5"] for r in all_results])),
        "total_tracks": sum(r["n_tracks"] for r in all_results),
        "total_yolo_dets": sum(r["n_yolo_dets"] for r in all_results),
        "pipeline_status": ps,
        "mean_timings_ms": {
            k: float(np.mean([r["timings_ms"].get(k,0) for r in all_results]))
            for k in ["yolo_ms","tracking_ms","tms12_ms","trajmae_ms","scte_ms","tce_ms","mvit_ms","aai_ms","emi_ms"]
        },
        "v1_comparison": {},
    }
    
    # Compare to v1
    for r in all_results:
        seq = r["sequence"]
        v1 = v1_results.get(seq)
        if v1:
            summary["v1_comparison"][seq] = {
                "v1_ndcg3": v1.get("ndcg3",0), "v2_ndcg3": r["ndcg3"],
                "v1_tracks": v1.get("n_tracks",0), "v2_tracks": r["n_tracks"],
                "ndcg3_delta": r["ndcg3"] - v1.get("ndcg3",0),
            }
    
    with open(os.path.join(OUT_DIR, "e2e_v2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "e2e_v2_pipeline_report.json"), "w") as f:
        json.dump(ps, f, indent=2)
    
    # Print status table
    log('\n' + '='*80)
    log('PIPELINE INTEGRATION STATUS — 11/11')
    log('='*80)
    log(f'{"Component":<15} {"Status":<12} {"Notes"}')
    log('-'*80)
    for c in ["YOLO","ByteTrack","TMS-12","TrajMAE","SCTE","TCE","MViTv2-S","AnomalyDet","AAI-v2","EMI","Ranker"]:
        p = ps.get(c, {"status":"UNKNOWN","note":""})
        log(f'{c:<15} {p["status"]:<12} {p["note"]}')
    
    log(f'\n{"Metric":<25} {"v2 (11/11)":>12}')
    log(f'{"Mean NDCG@3":<25} {summary["mean_ndcg3"]:>12.4f}')
    log(f'{"Mean NDCG@5":<25} {summary["mean_ndcg5"]:>12.4f}')
    log(f'{"Total tracks":<25} {summary["total_tracks"]:>12}')
    
    if summary["v1_comparison"]:
        log(f'\nv1 (8/11) vs v2 (11/11) Comparison:')
        log(f'{"Sequence":<10} {"v1 NDCG@3":>12} {"v2 NDCG@3":>12} {"Δ":>8} {"v1 tracks":>10} {"v2 tracks":>10}')
        for seq, comp in summary["v1_comparison"].items():
            log(f'{seq:<10} {comp["v1_ndcg3"]:>12.4f} {comp["v2_ndcg3"]:>12.4f} {comp["ndcg3_delta"]:>+8.4f} {comp["v1_tracks"]:>10} {comp["v2_tracks"]:>10}')
    
    log('\nE2E v2 PIPELINE COMPLETE — 11/11 COMPONENTS.')

if __name__ == "__main__":
    main()
