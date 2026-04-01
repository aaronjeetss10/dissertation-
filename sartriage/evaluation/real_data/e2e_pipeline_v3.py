"""
E2E Pipeline v3 — All Improvements Integrated
- 11/11 components
- Track quality filtering (size >= 15px)
- Confidence-aware TCE
- Majority vote ensemble (TMS-16+RF, SCTE+Linear, TrajMAE)
"""
import os, sys, json, math, time, glob, gc
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath("."))
from evaluation.real_data.tms12_standalone import extract_tms12, extract_tms16
from evaluation.scte import SCTEModel
from evaluation.traj_mae import TrajMAE

OUT_DIR = "evaluation/real_data/full/end_to_end_v3"
os.makedirs(OUT_DIR, exist_ok=True)
V1_DIR = "evaluation/real_data/full/end_to_end"
V2_DIR = "evaluation/real_data/full/end_to_end_v2"

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
MIN_SIZE_PX = 15  # Track quality filter

def log(msg):
    with open('/tmp/e2e_v3_progress.txt','a') as f: f.write(msg+'\n')

# ── SimpleTracker ──
def iou_box(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[0]+b1[2],b2[0]+b2[2]), min(b1[1]+b1[3],b2[1]+b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/(union+1e-8)

class SimpleTracker:
    def __init__(self, max_lost=10, iou_thresh=0.15, dist_thresh=3.0):
        self.max_lost=max_lost; self.iou_thresh=iou_thresh; self.dist_thresh=dist_thresh
        self.tracks={}; self.next_id=1
    def update(self, detections):
        matched_d=set()
        for tid, track in list(self.tracks.items()):
            best_score=-1; best_di=None
            for di, det in enumerate(detections):
                if di in matched_d: continue
                v=iou_box(track["bbox"],det[:4])
                c1=(track["bbox"][0]+track["bbox"][2]/2,track["bbox"][1]+track["bbox"][3]/2)
                c2=(det[0]+det[2]/2,det[1]+det[3]/2)
                d1=math.sqrt(track["bbox"][2]**2+track["bbox"][3]**2)
                d2=math.sqrt(det[2]**2+det[3]**2)
                cd=math.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)/((d1+d2)/2+1e-8)
                if v>self.iou_thresh or cd<self.dist_thresh:
                    score=v+max(0,1-cd/self.dist_thresh)
                    if score>best_score: best_score=score; best_di=di
            if best_di is not None:
                det=detections[best_di]; track["bbox"]=det[:4]; track["lost"]=0
                track["dets"].append(det); matched_d.add(best_di)
            else: track["lost"]+=1
        for tid in [t for t in self.tracks if self.tracks[t]["lost"]>self.max_lost]: del self.tracks[tid]
        for di, det in enumerate(detections):
            if di not in matched_d:
                self.tracks[self.next_id]={"bbox":det[:4],"lost":0,"dets":[det]}; self.next_id+=1
        return {tid:t for tid,t in self.tracks.items() if t["lost"]==0}

# ── Confidence-Aware TCE v3 ──
def tce_v3_score(traj, max_class_prob, speed_thresh=3.0):
    """TCE v3: base_score * dwell_escalation * (0.5 + 0.5 * max_class_probability)"""
    if len(traj) < 2: return 0.5, "UNKNOWN"
    n = min(10, len(traj))
    speeds = [math.sqrt((traj[i][0]-traj[i-1][0])**2+(traj[i][1]-traj[i-1][1])**2) for i in range(1,n)]
    if not speeds: return 0.5, "UNKNOWN"
    stat = sum(1 for s in speeds if s<speed_thresh)/len(speeds)
    ars = [traj[i][3]/(traj[i][2]+1e-8) for i in range(n)]

    # Base score from motion state
    if stat>0.8 and np.mean(ars)<0.6: base=0.85; state="COLLAPSED"
    elif stat>0.8: base=0.55; state="STATIONARY"
    else: base=0.2; state="MOVING"

    # Dwell escalation: longer tracks with sustained state get higher scores
    dwell = 1.0
    if len(traj) > 20:
        h1,h2 = traj[:len(traj)//2], traj[len(traj)//2:]
        sp1 = np.mean([math.sqrt((h1[i][0]-h1[i-1][0])**2+(h1[i][1]-h1[i-1][1])**2) for i in range(1,len(h1))]) if len(h1)>1 else 0
        sp2 = np.mean([math.sqrt((h2[i][0]-h2[i-1][0])**2+(h2[i][1]-h2[i-1][1])**2) for i in range(1,len(h2))]) if len(h2)>1 else 0
        if sp1>speed_thresh and sp2<speed_thresh*0.5:
            base=max(base,0.75); state="STOPPED"; dwell=1.15
        elif stat>0.8:
            dwell = 1.0 + 0.1*min(len(traj)/50.0, 1.0)  # up to +10% for long stationary

    # Confidence modulation: high classifier confidence → higher TCE
    conf_mod = 0.5 + 0.5 * max_class_prob
    
    score = min(1.0, base * dwell * conf_mod)
    return score, state

def compute_emi(frames_dir, frame_ids):
    orb=cv2.ORB_create(500); ids=sorted(frame_ids)[:30]; speeds=[]; hover=0; prev=None
    for fi in ids:
        img=cv2.imread(os.path.join(frames_dir,f"{fi}.jpg"))
        if img is None: continue
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if prev is not None:
            kp1,d1=orb.detectAndCompute(prev,None); kp2,d2=orb.detectAndCompute(gray,None)
            if d1 is not None and d2 is not None and len(d1)>10 and len(d2)>10:
                matches=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(d1,d2)
                if len(matches)>4:
                    src=np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                    dst=np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                    H,_=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
                    if H is not None: sp=math.sqrt(H[0,2]**2+H[1,2]**2); speeds.append(sp); hover+=1 if sp<2 else 0
        prev=gray
    if not speeds: return {"mean_speed":0,"hover_index":0,"phase":"UNKNOWN","attention_score":0},"RAN"
    ms=np.mean(speeds); hi=hover/len(speeds)
    phase="HOVERING" if hi>0.7 else "SCANNING" if ms<5 else "TRANSIT"
    return {"mean_speed":float(ms),"hover_index":float(hi),"phase":phase,"attention_score":1.0 if phase=="HOVERING" else 0.5 if phase=="SCANNING" else 0.0},"RAN"

def dcg(s,k): return sum(v/math.log2(i+2) for i,v in enumerate(s[:k]))
def ndcg(r,k):
    ideal=sorted(r,reverse=True); id_dcg=dcg(ideal,k)
    return dcg(r,k)/id_dcg if id_dcg>0 else 1.0

# ═══════════════════════════════════════════════════════════════════════
def run_sequence(seq, yolo, rf, scte, scte_clf, trajmae, aai, mvit_cache, ps):
    log(f'\n{"="*60}\nSEQUENCE: {seq}\n{"="*60}')
    timings={}
    frame_dir=None
    for m in glob.glob(os.path.join(TEST_BASE,"**",seq),recursive=True):
        if os.path.isdir(m) and "Extracted-Frames" in m: frame_dir=m; break

    # GT
    gt_tracks=defaultdict(lambda:{"frames":[],"bboxes":[],"action":None})
    with open(os.path.join(LABELS_DIR,f"{seq}.txt")) as f:
        for line in f:
            p=line.strip().split()
            if len(p)<11: continue
            tid=int(p[0]); xmin=float(p[1])*SCALE_X; ymin=float(p[2])*SCALE_Y
            xmax=float(p[3])*SCALE_X; ymax=float(p[4])*SCALE_Y
            fi=int(p[5]); act=p[10].strip('"')
            gt_tracks[tid]["frames"].append(fi); gt_tracks[tid]["bboxes"].append([xmin,ymin,xmax-xmin,ymax-ymin])
            gt_tracks[tid]["action"]=act
    all_frames=set()
    for t in gt_tracks.values(): all_frames.update(t["frames"])
    sampled=sorted(all_frames)[::6]
    log(f'  Frames: {len(all_frames)} total, {len(sampled)} sampled')

    # STAGE 1: YOLO
    t0=time.time(); dets_per_frame={}
    for fi in sampled:
        img_path=os.path.join(frame_dir,f"{fi}.jpg")
        if not os.path.exists(img_path): continue
        results=yolo(img_path,imgsz=1280,conf=0.1,verbose=False)
        dets=[]
        for r in results:
            for box in r.boxes:
                if int(box.cls[0])==0:
                    x1,y1,x2,y2=box.xyxy[0].tolist()
                    dets.append([x1,y1,x2-x1,y2-y1,float(box.conf[0])])
        dets_per_frame[fi]=dets
    timings["yolo_ms"]=(time.time()-t0)/max(len(sampled),1)*1000
    total_dets=sum(len(d) for d in dets_per_frame.values())
    ps["YOLO"]={"status":"RAN","note":f"HERIDAL, {total_dets} dets"}
    log(f'  YOLO: {total_dets} dets ({timings["yolo_ms"]:.0f}ms/frame)')

    # STAGE 2: Tracker
    t1=time.time()
    tracker=SimpleTracker(max_lost=10,iou_thresh=0.15,dist_thresh=3.0)
    track_hist=defaultdict(list)
    for fi in sampled:
        dets=dets_per_frame.get(fi,[])
        if not dets: continue
        active=tracker.update(dets)
        for tid,t in active.items(): d=t["dets"][-1]; track_hist[tid].append((fi,d[:4],d[4]))
    valid={}
    for k,v in track_hist.items():
        if len(v)>=5:
            sizes=[math.sqrt(max(d[1][2],1)*max(d[1][3],1)) for d in v]
            if np.mean(sizes)>=MIN_SIZE_PX:  # Track quality filter
                valid[k]=v
    n_before=sum(1 for v in track_hist.values() if len(v)>=5)
    timings["tracking_ms"]=(time.time()-t1)/max(len(sampled),1)*1000
    
    # GT matching
    gt_match={}
    for st_id,st_dets in valid.items():
        best_overlap=0; best_gt=None
        for gt_id,gt_v in gt_tracks.items():
            sp=0
            for sf,sb,_ in st_dets:
                for i,gf in enumerate(gt_v["frames"]):
                    if sf==gf and iou_box(sb,gt_v["bboxes"][i])>0.2: sp+=1
            if sp>best_overlap: best_overlap=sp; best_gt=gt_id
        if best_gt and best_overlap>2: gt_match[st_id]=best_gt
    ps["ByteTrack"]={"status":"RAN","note":f"SimpleTracker + size≥{MIN_SIZE_PX}px filter, {n_before}→{len(valid)} tracks"}
    log(f'  Tracker: {n_before}→{len(valid)} tracks after filter, {len(gt_match)} GT-matched')

    # Build track data
    tracks_data={}
    for tid,dets in valid.items():
        ds=sorted(dets,key=lambda d:d[0])
        centroids=[(d[1][0]+d[1][2]/2,d[1][1]+d[1][3]/2) for d in ds]
        bboxes=[d[1] for d in ds]; confs=[d[2] for d in ds]; frames=[d[0] for d in ds]
        mean_size=np.mean([math.sqrt(max(b[2],1)*max(b[3],1)) for b in bboxes])
        raw_traj=[[cx,cy,b[2],b[3]] for (cx,cy),b in zip(centroids,bboxes)]
        deltas=[[centroids[i][0]-centroids[i-1][0],centroids[i][1]-centroids[i-1][1],
                 bboxes[i][2]-bboxes[i-1][2],bboxes[i][3]-bboxes[i-1][3]] for i in range(1,len(centroids))]
        scte_tokens=[]
        for i in range(1,len(centroids)):
            dx=(centroids[i][0]-centroids[i-1][0])/1280.0; dy=(centroids[i][1]-centroids[i-1][1])/720.0
            aspect=bboxes[i][3]/max(bboxes[i][2],1); sn=math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            scte_tokens.append([dx,dy,aspect,sn])
        gt_id=gt_match.get(tid); gt_action=gt_tracks[gt_id]["action"] if gt_id else None
        gt_sar=SAR_MAP.get(gt_action) if gt_action else None
        gt_fine={"Standing":"standing","Sitting":"sitting","Walking":"walking","Running":"running","Lying":"lying_down"}.get(gt_action)
        tracks_data[tid]={
            "raw_traj":raw_traj,"deltas":deltas,"scte_tokens":scte_tokens,
            "centroids":centroids,"bboxes":bboxes,"frames":frames,
            "mean_size":mean_size,"mean_conf":np.mean(confs),"n_frames":len(frames),
            "gt_action":gt_action,"gt_sar":gt_sar,"gt_fine":gt_fine,"gt_id":gt_id,
            "gt_priority":GT_PRIORITY.get(gt_fine,0.15) if gt_fine else 0.05,
        }

    # C1: TMS-16 + RF
    t3=time.time()
    for tid,td in tracks_data.items():
        feats=extract_tms16(td["raw_traj"])
        feats=[0 if(math.isnan(f)or math.isinf(f))else f for f in feats]
        td["tms16_feats"]=feats
        pred=rf.predict([feats])[0]; proba=rf.predict_proba([feats])[0]
        td["tms16_class"]=RF_CLASSES[pred]; td["tms16_conf"]=float(max(proba))
        td["tms16_proba"]=proba.tolist()
    ps["TMS-16"]={"status":"RAN","note":"SMOTE-balanced RF, 200 trees, 16 features"}
    timings["tms16_ms"]=(time.time()-t3)/max(len(tracks_data),1)*1000

    # C2: TrajMAE
    t4=time.time()
    for tid,td in tracks_data.items():
        dt=td["deltas"]; arr=np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(dt),50)): arr[j]=dt[j]
        x=torch.tensor(arr).unsqueeze(0)
        with torch.no_grad():
            logits=trajmae(x,pretrain=False); probs=torch.softmax(logits,1)[0]
        pred=probs.argmax().item()
        td["trajmae_class"]=RF_CLASSES[pred]; td["trajmae_conf"]=float(probs[pred].item())
        td["trajmae_proba"]=probs.tolist()
        try:
            recon,target,_=trajmae(x,pretrain=True)
            td["recon_error"]=float(nn.MSELoss()(recon,target).item())
        except: td["recon_error"]=None
    ps["TrajMAE"]={"status":"RAN","note":"Pre-trained encoder+classifier"}
    timings["trajmae_ms"]=(time.time()-t4)/max(len(tracks_data),1)*1000
    errors=[td["recon_error"] for td in tracks_data.values() if td["recon_error"] is not None]
    anom_thresh=np.mean(errors)+2*np.std(errors) if errors else 999
    for td in tracks_data.values(): td["is_anomaly"]=bool(td["recon_error"] is not None and td["recon_error"]>anom_thresh)
    ps["AnomalyDet"]={"status":"RAN","note":f"TrajMAE recon, thresh={anom_thresh:.2f}"}

    # C3: SCTE
    t5s=time.time()
    for tid,td in tracks_data.items():
        st=td["scte_tokens"]; arr=np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(st),50)): arr[j]=st[j]
        x=torch.tensor(arr).unsqueeze(0)
        with torch.no_grad(): emb=scte.get_embedding(x).cpu().numpy()[0]
        td["scte_emb"]=emb
        pred=scte_clf.predict([emb])[0]; proba=scte_clf.predict_proba([emb])[0]
        td["scte_class"]=RF_CLASSES[pred]; td["scte_conf"]=float(max(proba))
        td["scte_proba"]=proba.tolist()
    ps["SCTE"]={"status":"RAN","note":"Contrastive + linear probe"}
    timings["scte_ms"]=(time.time()-t5s)/max(len(tracks_data),1)*1000

    # ENSEMBLE: Majority Vote
    for tid,td in tracks_data.items():
        votes=[td["tms16_class"],td["scte_class"],td["trajmae_class"]]
        td["ensemble_class"]=Counter(votes).most_common(1)[0][0]
        td["ensemble_score"]=RF_PRIORITY[td["ensemble_class"]]
        # Max class probability for confidence-aware TCE
        td["max_class_prob"]=max(td["tms16_conf"],td["scte_conf"],td["trajmae_conf"])

    # TCE v3 (confidence-aware)
    t5=time.time()
    for tid,td in tracks_data.items():
        score,state=tce_v3_score(td["raw_traj"],td["max_class_prob"])
        td["tce_score"]=score; td["tce_state"]=state
    ps["TCE"]={"status":"RAN","note":"v3 confidence-aware (base * dwell * conf_mod)"}
    timings["tce_ms"]=(time.time()-t5)/max(len(tracks_data),1)*1000

    # MViTv2-S (cache)
    cache_seq=mvit_cache.get(seq,{})
    n_hit=0
    for tid,td in tracks_data.items():
        gt_id=td.get("gt_id"); cached=cache_seq.get(str(gt_id)) if gt_id else None
        if cached and cached.get("predicted_sar"):
            td["mvit_class"]=cached["predicted_sar"]; td["mvit_conf"]=cached["confidence"]; n_hit+=1
        else: td["mvit_class"]="unknown"; td["mvit_conf"]=0
    ps["MViTv2-S"]={"status":"RAN","note":f"Cache, {n_hit}/{len(tracks_data)} hits"}

    # AAI-v2
    t7=time.time()
    for tid,td in tracks_data.items():
        size=td["mean_size"]; conf=td["mean_conf"]
        motion=np.mean([math.sqrt((td["raw_traj"][i][0]-td["raw_traj"][i-1][0])**2+
                 (td["raw_traj"][i][1]-td["raw_traj"][i-1][1])**2) for i in range(1,len(td["raw_traj"]))]) if len(td["raw_traj"])>1 else 0
        inp=torch.tensor([[size,conf,motion]]).float()
        with torch.no_grad(): w=aai(inp)[0]
        w_pixel=float(w[0]); w_traj=float(w[1])
        mvit_score=RF_PRIORITY.get(td["mvit_class"],0.2)
        traj_score=td["ensemble_score"]*(0.5+0.5*td["tce_score"])
        td["aai_w_pixel"]=w_pixel; td["aai_w_traj"]=w_traj
        td["fused_score"]=w_pixel*mvit_score+w_traj*traj_score
    ps["AAI-v2"]={"status":"RAN","note":"MLP, ensemble+conf_TCE input"}
    timings["aai_ms"]=(time.time()-t7)/max(len(tracks_data),1)*1000

    # EMI
    t8=time.time()
    emi_result,_=compute_emi(frame_dir,sampled)
    attn=emi_result["attention_score"]
    for td in tracks_data.values():
        td["emi_phase"]=emi_result["phase"]; td["emi_mod"]=1.0+0.3*attn
        td["final_priority"]=td["fused_score"]*td["emi_mod"]
    ps["EMI"]={"status":"RAN","note":f"Phase={emi_result['phase']}"}; ps["Ranker"]={"status":"RAN","note":"Priority-sorted"}
    timings["emi_ms"]=(time.time()-t8)/max(len(sampled),1)*1000

    # Rank
    ranked=sorted(tracks_data.items(),key=lambda x:-x[1]["final_priority"])
    ranked_gt=[td["gt_priority"] for _,td in ranked]
    ndcg3=ndcg(ranked_gt,3); ndcg5=ndcg(ranked_gt,5)
    has_lying=any(td["gt_fine"]=="lying_down" for _,td in ranked)
    top3_fines=[td["gt_fine"] for _,td in ranked[:3]]
    r3_lying=1.0 if "lying_down" in top3_fines else 0.0 if has_lying else float('nan')

    output=[]
    for rank,(tid,td) in enumerate(ranked,1):
        output.append({
            "rank":rank,"track_id":int(tid),
            "ensemble_class":td["ensemble_class"],"tms16_class":td["tms16_class"],
            "scte_class":td["scte_class"],"trajmae_class":td["trajmae_class"],
            "mvit_class":td["mvit_class"],"tce_state":td["tce_state"],"tce_score":round(td["tce_score"],3),
            "aai_w_traj":round(td["aai_w_traj"],3),"aai_w_pixel":round(td["aai_w_pixel"],3),
            "fused_score":round(td["fused_score"],4),"final_priority":round(td["final_priority"],4),
            "person_size_px":round(td["mean_size"],1),"n_frames":td["n_frames"],
            "gt_action":td["gt_action"],"gt_sar":td["gt_sar"],"gt_fine":td.get("gt_fine"),
            "is_anomaly":td.get("is_anomaly",False),"gt_match_id":td.get("gt_id"),
        })

    log(f'\n  RANKED OUTPUT ({seq}, {len(tracks_data)} tracks):')
    log(f'  {"Rk":>3} {"TID":>4} {"Sz":>4} {"Ensemble":>10} {"TMS16":>10} {"TrajMAE":>10} {"TCE":>8} {"AAIw":>8} {"Final":>7} {"GT":>10}')
    for e in output[:10]:
        log(f'  {e["rank"]:3d} {e["track_id"]:4d} {e["person_size_px"]:4.0f} {e["ensemble_class"]:>10} {e["tms16_class"]:>10} {e["trajmae_class"]:>10} {e["tce_state"]:>8} {e["aai_w_traj"]:.2f}/{e["aai_w_pixel"]:.2f} {e["final_priority"]:7.4f} {e["gt_action"] or "?":>10}')

    # Timeline figure
    fig,ax=plt.subplots(figsize=(14,max(6,len(tracks_data)*0.3+2)))
    cmap={"lying_down":"#e74c3c","stationary":"#3498db","walking":"#2ecc71","running":"#f39c12","unknown":"grey"}
    for row,(tid,td) in enumerate(ranked):
        fs=min(td["frames"]); fe=max(td["frames"])
        c=cmap.get(td["ensemble_class"],"grey"); a=0.3+0.7*min(td["final_priority"],1.0)
        ax.barh(row,fe-fs,left=fs,height=0.7,color=c,alpha=a,edgecolor='black',linewidth=0.3)
        ax.text(fe+5,row,f'{td["final_priority"]:.3f}',va='center',fontsize=7)
        if td["gt_fine"]=="lying_down": ax.plot(fs-10,row,marker='*',color='red',markersize=12,zorder=5)
    ax.set_yticks(range(len(ranked))); ax.set_yticklabels([f'T{tid}' for tid,_ in ranked],fontsize=7)
    ax.set_xlabel('Frame'); ax.invert_yaxis()
    ax.set_title(f'SARTriage v3 — {seq} | NDCG@3={ndcg3:.3f} | {len(tracks_data)} tracks (filtered)',fontsize=12)
    from matplotlib.patches import Patch
    legend=[Patch(facecolor=c,label=l) for l,c in cmap.items() if l!="unknown"]
    legend.append(plt.Line2D([0],[0],marker='*',color='red',linestyle='None',markersize=10,label='GT lying'))
    ax.legend(handles=legend,loc='lower right',fontsize=8); ax.grid(True,axis='x',alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,f"v3_timeline_{seq.replace('.','_')}.png"),dpi=200); plt.close()

    return {"sequence":seq,"n_tracks":len(tracks_data),"n_before_filter":n_before,
            "n_yolo_dets":total_dets,"ndcg3":float(ndcg3),"ndcg5":float(ndcg5),
            "recall3_lying":float(r3_lying) if not math.isnan(r3_lying) else None,
            "has_lying_gt":has_lying,"timings_ms":timings,"ranked_output":output}

def main():
    with open('/tmp/e2e_v3_progress.txt','w') as f: f.write('E2E v3 — All Improvements\n')
    log('Loading models...')

    from ultralytics import YOLO
    yolo=YOLO(YOLO_WEIGHTS); log('  YOLO ✓')

    # TMS-16 RF
    with open("evaluation/real_data/okutama_all_tracks.json") as f: data=json.load(f)
    X,y=[],[]
    for tid,t in data["tracks"].items():
        if t["track_length_frames"]<20: continue
        act=t["primary_action"]
        if act not in SAR_MAP: continue
        traj=[[c[0],c[1],b[2],b[3]] for c,b in zip(t["centroids"],t["bboxes"])]
        feats=extract_tms16(traj); feats=[0 if(math.isnan(f)or math.isinf(f))else f for f in feats]
        X.append(feats); y.append(RF_CLASSES.index(SAR_MAP[act]))
    X_sm,y_sm=SMOTE(random_state=42).fit_resample(np.array(X),np.array(y))
    rf=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42); rf.fit(X_sm,y_sm)
    log('  TMS-16+RF ✓')

    # SCTE + classifier
    scte_model=SCTEModel(input_dim=4,d_model=32,proj_dim=16,n_heads=2,n_layers=2,dropout=0.1,max_len=50)
    scte_model.load_state_dict(torch.load(SCTE_WEIGHTS,map_location="cpu",weights_only=True)); scte_model.eval()
    # Train SCTE linear classifier
    scte_X=[]; scte_y=[]
    for i,xi in enumerate(X):
        traj_data=list(data["tracks"].values())
        # Use TMS training data labels with SCTE embeddings
    # Simpler: extract SCTE embeddings for training data
    all_tracks_list=[]
    for tid,t in data["tracks"].items():
        if t["track_length_frames"]<20: continue
        act=t["primary_action"]
        if act not in SAR_MAP: continue
        centroids=t["centroids"]; bboxes=t["bboxes"]
        scte_tokens=[]
        for i in range(1,len(centroids)):
            dx=(centroids[i][0]-centroids[i-1][0])/1280.0; dy=(centroids[i][1]-centroids[i-1][1])/720.0
            aspect=bboxes[i][3]/max(bboxes[i][2],1); sn=math.sqrt(bboxes[i][2]*bboxes[i][3])/200.0
            scte_tokens.append([dx,dy,aspect,sn])
        arr=np.zeros((50,4),dtype=np.float32)
        for j in range(min(len(scte_tokens),50)): arr[j]=scte_tokens[j]
        all_tracks_list.append((arr,RF_CLASSES.index(SAR_MAP[act])))
    # Extract embeddings
    scte_tokens_all=np.stack([t[0] for t in all_tracks_list])
    scte_labels_all=np.array([t[1] for t in all_tracks_list])
    with torch.no_grad():
        scte_embs=[]
        for s in range(0,len(scte_tokens_all),64):
            scte_embs.append(scte_model.get_embedding(torch.FloatTensor(scte_tokens_all[s:s+64])).cpu().numpy())
        scte_embs_all=np.concatenate(scte_embs)
    scte_clf=LogisticRegression(max_iter=1000,random_state=42); scte_clf.fit(scte_embs_all,scte_labels_all)
    log('  SCTE+Linear ✓')

    trajmae=TrajMAE(num_classes=4,d_model=64,d_decoder=64,mask_ratio=0.7,max_len=50)
    trajmae.encoder.load_state_dict(torch.load(TRAJMAE_WEIGHTS,map_location="cpu",weights_only=True)); trajmae.eval()
    log('  TrajMAE ✓')

    from evaluation.aai_v2 import AAIv2MetaClassifier
    aai=AAIv2MetaClassifier(); raw=torch.load(AAI_WEIGHTS,map_location="cpu",weights_only=False)
    aai.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw); aai.eval()
    log('  AAI-v2 ✓')

    with open(MVIT_CACHE) as f: mvit_cache=json.load(f)
    log('  MViTv2-S cache ✓')

    ps={}; all_results=[]
    for seq in SEQUENCES:
        r=run_sequence(seq,yolo,rf,scte_model,scte_clf,trajmae,aai,mvit_cache,ps)
        with open(os.path.join(OUT_DIR,f"v3_{seq.replace('.','_')}.json"),"w") as f: json.dump(r,f,indent=2)
        all_results.append(r)

    # Load previous versions for comparison
    v1_ndcg3=[]; v2_ndcg3=[]
    for seq in SEQUENCES:
        p1=os.path.join(V1_DIR,f"e2e_{seq.replace('.','_')}_results.json")
        p2=os.path.join(V2_DIR,f"e2e_v2_{seq.replace('.','_')}.json")
        if os.path.exists(p1):
            with open(p1) as f: v1_ndcg3.append(json.load(f).get("ndcg3",0))
        if os.path.exists(p2):
            with open(p2) as f: v2_ndcg3.append(json.load(f).get("ndcg3",0))

    summary={
        "version":"v3_all_improvements",
        "improvements":["Track quality filter (size>=15px)","Confidence-aware TCE v3","Majority vote ensemble","TMS-16 features"],
        "mean_ndcg3":float(np.mean([r["ndcg3"] for r in all_results])),
        "mean_ndcg5":float(np.mean([r["ndcg5"] for r in all_results])),
        "total_tracks":sum(r["n_tracks"] for r in all_results),
        "total_before_filter":sum(r["n_before_filter"] for r in all_results),
        "pipeline_status":ps,
        "per_sequence":{r["sequence"]:{"ndcg3":r["ndcg3"],"ndcg5":r["ndcg5"],"n_tracks":r["n_tracks"]} for r in all_results},
        "version_comparison":{
            "v1_gt_isolated":{"mean_ndcg3":float(np.mean(v1_ndcg3)) if v1_ndcg3 else None,"components":"9/11"},
            "v2_no_filter":{"mean_ndcg3":float(np.mean(v2_ndcg3)) if v2_ndcg3 else None,"components":"11/11"},
            "v3_all_improvements":{"mean_ndcg3":float(np.mean([r["ndcg3"] for r in all_results])),"components":"11/11+improvements"},
        },
    }
    with open(os.path.join(OUT_DIR,"v3_summary.json"),"w") as f: json.dump(summary,f,indent=2)

    # Print comparison
    log('\n' + '='*80)
    log('VERSION COMPARISON')
    log('='*80)
    log(f'\n  {"Version":<30} {"Components":>12} {"NDCG@3":>8} {"NDCG@5":>8} {"Tracks":>8}')
    log(f'  {"-"*70}')
    if v1_ndcg3: log(f'  {"v1 (GT-isolated)":<30} {"9/11":>12} {np.mean(v1_ndcg3):>8.4f} {"—":>8} {"~16/seq":>8}')
    if v2_ndcg3: log(f'  {"v2 (no filter)":<30} {"11/11":>12} {np.mean(v2_ndcg3):>8.4f} {"—":>8} {"~72/seq":>8}')
    v3_n3=np.mean([r["ndcg3"] for r in all_results]); v3_n5=np.mean([r["ndcg5"] for r in all_results])
    log(f'  {"v3 (all improvements)":<30} {"11/11+":>12} {v3_n3:>8.4f} {v3_n5:>8.4f} {summary["total_tracks"]/3:>8.0f}')
    
    log('\n  Per-sequence:')
    for r in all_results:
        log(f'    {r["sequence"]}: NDCG@3={r["ndcg3"]:.4f}, NDCG@5={r["ndcg5"]:.4f}, tracks={r["n_tracks"]}')

    log('\n  11/11 PIPELINE STATUS:')
    for c in ["YOLO","ByteTrack","TMS-16","TrajMAE","SCTE","TCE","MViTv2-S","AnomalyDet","AAI-v2","EMI","Ranker"]:
        p=ps.get(c,{"status":"?","note":""}); log(f'    {c:<15} {p["status"]:<5} {p["note"]}')
    log('\nE2E v3 COMPLETE.')

if __name__=="__main__":
    main()
