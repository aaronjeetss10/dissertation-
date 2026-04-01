"""
EXPERIMENT 5: TrajMAE-style Trajectory Classifier on Real Okutama
Uses a lightweight LSTM encoder (fast, CPU-friendly) operating on (dx,dy,dw,dh) tokens.
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

FULL_DIR = "evaluation/real_data/full"
os.makedirs(FULL_DIR, exist_ok=True)

MAPPING = {"Standing":"stationary","Sitting":"stationary","Walking":"walking","Running":"running","Lying":"lying_down"}
CLASS_LIST = ["lying_down","stationary","walking","running"]
MAX_LEN = 50

def load_tracks():
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)
    tracks = []
    for tid, t in data["tracks"].items():
        if t["track_length_frames"] < 20: continue
        act = t["primary_action"]
        if act not in MAPPING: continue
        centroids = t["centroids"]; bboxes = t["bboxes"]
        tokens = []
        for i in range(1, len(centroids)):
            tokens.append([centroids[i][0]-centroids[i-1][0], centroids[i][1]-centroids[i-1][1],
                           bboxes[i][2]-bboxes[i-1][2], bboxes[i][3]-bboxes[i-1][3]])
        if len(tokens) < 5: continue
        tracks.append({"id":tid,"tokens":tokens,"label":MAPPING[act],"size":t["mean_size_px"]})
    return tracks

def pad_seq(tokens):
    arr = np.zeros((MAX_LEN, 4), dtype=np.float32)
    for i in range(min(len(tokens), MAX_LEN)): arr[i] = tokens[i]
    return arr

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden=64, n_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, n_classes))
    
    def forward(self, x):
        out, (h, _) = self.lstm(x)
        # Use last hidden state from both directions
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.classifier(h)

def main():
    print("="*60)
    print("EXPERIMENT 5: Trajectory Sequence Classifier (LSTM)")
    print("="*60, flush=True)
    
    tracks = load_tracks()
    print(f"Loaded {len(tracks)} tracks.", flush=True)
    
    test_seqs = {"1.1.8","1.1.9","1.2.1","1.2.3","1.2.10","2.1.8","2.1.9","2.2.1","2.2.3","2.2.10"}
    train_t = [t for t in tracks if t["id"].split("_")[0].replace("seq","") not in test_seqs]
    test_t = [t for t in tracks if t["id"].split("_")[0].replace("seq","") in test_seqs]
    if len(test_t) < 10:
        np.random.seed(42); idx = np.random.permutation(len(tracks))
        s = int(0.7*len(tracks))
        train_t = [tracks[i] for i in idx[:s]]; test_t = [tracks[i] for i in idx[s:]]
    print(f"Train: {len(train_t)} | Test: {len(test_t)}", flush=True)
    
    X_tr = torch.tensor(np.array([pad_seq(t["tokens"]) for t in train_t]))
    y_tr = torch.tensor([CLASS_LIST.index(t["label"]) for t in train_t])
    X_te = torch.tensor(np.array([pad_seq(t["tokens"]) for t in test_t]))
    y_te = torch.tensor([CLASS_LIST.index(t["label"]) for t in test_t])
    
    mask = X_tr != 0
    mean = X_tr[mask].mean(); std = X_tr[mask].std()
    if std < 1e-8: std = torch.tensor(1.0)
    X_tr = (X_tr-mean)/std; X_te = (X_te-mean)/std
    
    model = TrajectoryLSTM(4, 64, len(CLASS_LIST))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([5.0,1.0,1.0,3.0]))
    
    batch_size = 128
    print("Training (15 epochs)...", flush=True)
    for epoch in range(15):
        model.train(); perm = torch.randperm(len(X_tr)); correct=0; total=0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(1)==y_tr[idx]).sum().item(); total += len(idx)
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/50: Train Acc={correct/total*100:.1f}%", flush=True)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits_te = model(X_te)
    all_preds = logits_te.argmax(1).numpy()
    all_true = y_te.numpy()
    acc = (all_preds==all_true).mean()
    
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
    kappa = cohen_kappa_score(all_true, all_preds)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f}")
    report_str = classification_report(all_true, all_preds, target_names=CLASS_LIST, zero_division=0)
    print(report_str)
    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:"); print(cm)
    
    print("\nBy size:", flush=True)
    bins_info = {}
    for name, lo, hi in [("<50px",0,50),("50-75px",50,75),("75-100px",75,100),(">100px",100,9999)]:
        idx_b = [i for i,t in enumerate(test_t) if lo<=t["size"]<hi]
        if not idx_b: continue
        b_acc = (all_preds[idx_b]==all_true[idx_b]).mean()
        bins_info[name] = {"n":len(idx_b),"accuracy":float(b_acc)}
        print(f"  {name:10s}: N={len(idx_b):3d} Acc={b_acc*100:.1f}%")
    
    with open(os.path.join(FULL_DIR,"trajmae_results.json"),"w") as f:
        json.dump({"accuracy":float(acc),"kappa":float(kappa),
                   "report":classification_report(all_true,all_preds,target_names=CLASS_LIST,output_dict=True,zero_division=0),
                   "confusion_matrix":cm.tolist(),"size_bins":bins_info,
                   "n_train":len(train_t),"n_test":len(test_t),
                   "architecture":"BiLSTM(4→64×2, 2-layer) + MLP head",
                   "note":"Operates on (dx,dy,dw,dh) trajectory tokens, same as TrajMAE tokenization"},f,indent=2)
    print("EXPERIMENT 5 COMPLETE.", flush=True)

if __name__ == "__main__":
    main()
