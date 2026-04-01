"""
evaluation/scte.py
=====================
Scale-Contrastive Trajectory Embedding (SCTE).

Trains the TrajMAE encoder to produce altitude-invariant 64-dim [CLS]
embeddings.  The key insight: the *same* person performing the *same*
action looks different at 50 m vs 100 m altitude — centroid displacements
(dx, dy) and bbox size shrink proportionally — but the action identity
is unchanged.  SCTE uses InfoNCE contrastive learning to collapse these
altitude-variant representations into a shared embedding space.

Training protocol
-----------------
1.  Generate trajectory pairs: for each anchor trajectory at altitude_1,
    create a positive by rescaling to altitude_2 (simulating the same
    action seen from a different flight height).
2.  In-batch negatives: all other trajectories in the mini-batch from
    different actions serve as negatives.
3.  InfoNCE loss: pull positive pairs together, push negatives apart in
    the 64-dim [CLS] embedding space.

Altitude scaling model
-----------------------
At altitude h, a person of real-world height H subtends:

    person_pixels ∝ (f · H) / h

So doubling altitude halves px displacements and bbox size.  The scale
factor between altitude h₁ and h₂ is simply  s = h₁ / h₂.  We apply:

    dx′ = dx · s,   dy′ = dy · s,   size′ = size · s,   aspect′ ≈ aspect

Plus Gaussian jitter to simulate detector noise at the new scale.

Usage
-----
    python -m sartriage.evaluation.scte                # full pipeline
    python -m sartriage.evaluation.scte --temperature 0.05 --epochs 120
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SAR_ACTIONS = [
    "falling", "running", "lying_down", "crawling",
    "waving", "collapsed", "stumbling", "walking",
]

# Simulated altitude → person-size mapping (metres → px at 1080p)
ALTITUDE_PROFILES = {
    30:  {"person_px": 120, "desc": "low hover"},
    50:  {"person_px": 72,  "desc": "standard patrol"},
    80:  {"person_px": 45,  "desc": "survey"},
    100: {"person_px": 36,  "desc": "wide area"},
    150: {"person_px": 24,  "desc": "high survey"},
    200: {"person_px": 18,  "desc": "extreme altitude"},
    300: {"person_px": 12,  "desc": "dot scale"},
}


# ════════════════════════════════════════════════════════════════════════
# Altitude-Scaling Augmentation
# ════════════════════════════════════════════════════════════════════════

def altitude_scale_trajectory(
    seq: np.ndarray,
    source_alt: float,
    target_alt: float,
    noise_std: float = 0.0005,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Rescale a trajectory to simulate a different flight altitude.

    Parameters
    ----------
    seq : (T, 4)  — (dx, dy, aspect, size_norm) per frame.
    source_alt : float
        Altitude at which the trajectory was originally generated.
    target_alt : float
        Simulated target altitude.
    noise_std : float
        Gaussian noise σ added to scaled dx/dy to simulate increased
        detector jitter at higher altitudes.
    rng : numpy Generator, optional

    Returns
    -------
    scaled : (T, 4) — rescaled trajectory.

    Physics
    -------
    Scale factor  s = source_alt / target_alt.
    - s < 1 → target is lower  → person appears larger  → bigger dx/dy/size
    - s > 1 → target is higher → person appears smaller → smaller dx/dy/size
    Aspect ratio is approximately altitude-invariant (body proportions
    don't change with distance).
    """
    rng = rng or np.random.default_rng()
    s = source_alt / target_alt  # < 1 if going higher, > 1 if going lower

    scaled = seq.copy()
    # dx, dy: scale pixel displacements
    scaled[:, 0] *= s
    scaled[:, 1] *= s
    # aspect: nearly invariant (add tiny perturbation for realism)
    scaled[:, 2] += rng.normal(0, 0.01, size=scaled.shape[0])
    scaled[:, 2] = np.clip(scaled[:, 2], 0.15, 3.0)
    # size_norm: scales linearly with altitude
    scaled[:, 3] *= s
    scaled[:, 3] = np.clip(scaled[:, 3], 1e-4, 1.0)

    # Add altitude-proportional detector noise (higher alt → more jitter)
    jitter_scale = max(1.0, target_alt / source_alt)  # more noise at higher alt
    scaled[:, 0] += rng.normal(0, noise_std * jitter_scale, size=scaled.shape[0])
    scaled[:, 1] += rng.normal(0, noise_std * jitter_scale, size=scaled.shape[0])

    return scaled


# ════════════════════════════════════════════════════════════════════════
# Contrastive Pair Dataset
# ════════════════════════════════════════════════════════════════════════

class SCTEPairDataset(Dataset):
    """Dataset that yields (anchor, positive, label) triplets.

    For each trajectory:
    - anchor  = trajectory at altitude_1
    - positive = *same* trajectory rescaled to altitude_2
    - label   = action class index

    Negative pairs are formed in-batch via the InfoNCE formulation
    (all other samples in the batch are negatives).
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        source_alt: float = 50.0,
        altitude_range: Tuple[float, float] = (30.0, 300.0),
        noise_std: float = 0.0005,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        sequences : (N, T, 4) — trajectory token sequences.
        labels : (N,) — integer action labels.
        source_alt : float — altitude of the source trajectories.
        altitude_range : (lo, hi) — range for random target altitude.
        noise_std : float — detector jitter noise for scaled trajectories.
        seed : int — reproducibility.
        """
        self.sequences = sequences
        self.labels = labels
        self.source_alt = source_alt
        self.alt_lo, self.alt_hi = altitude_range
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq = self.sequences[idx]  # (T, 4)
        label = int(self.labels[idx])

        # Sample a random target altitude different from source
        target_alt = self.rng.uniform(self.alt_lo, self.alt_hi)
        # Ensure at least 20% altitude difference for meaningful scaling
        while abs(target_alt - self.source_alt) / self.source_alt < 0.2:
            target_alt = self.rng.uniform(self.alt_lo, self.alt_hi)

        positive = altitude_scale_trajectory(
            seq, self.source_alt, target_alt,
            noise_std=self.noise_std, rng=self.rng,
        )

        return (
            torch.FloatTensor(seq),
            torch.FloatTensor(positive),
            label,
        )


# ════════════════════════════════════════════════════════════════════════
# InfoNCE Loss
# ════════════════════════════════════════════════════════════════════════

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for altitude-invariant embeddings.

    Given a batch of N anchor embeddings z_a and N positive embeddings z_p
    (same action, different altitude), the loss treats each (z_a[i], z_p[i])
    as a positive pair and all other 2(N-1) embeddings as negatives.

    The loss is the symmetric InfoNCE:

        L = ½ · [ L_a→p + L_p→a ]

    where L_a→p = -log( exp(sim(a_i, p_i)/τ) / Σ_j exp(sim(a_i, p_j)/τ) )

    This is equivalent to NT-Xent (Normalised Temperature-scaled
    Cross-Entropy) from SimCLR.

    Parameters
    ----------
    temperature : float
        Softmax temperature τ (default 0.07).  Lower → sharper contrast.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Parameters
        ----------
        z_anchor : (B, D) — L2-normalised anchor embeddings.
        z_positive : (B, D) — L2-normalised positive embeddings.
        labels : (B,) int, optional — used for supervised variant
                 (same-class samples also treated as positives).

        Returns
        -------
        loss : scalar tensor.
        """
        B = z_anchor.size(0)
        device = z_anchor.device

        # L2-normalise (in case caller didn't)
        z_a = F.normalize(z_anchor, dim=1)
        z_p = F.normalize(z_positive, dim=1)

        # ── Similarity matrices ──
        # Anchor → Positive:  sim_ap[i, j] = z_a[i] · z_p[j]  (B × B)
        sim_ap = torch.mm(z_a, z_p.t()) / self.temperature
        # Positive → Anchor:  sim_pa[i, j] = z_p[i] · z_a[j]  (B × B)
        sim_pa = torch.mm(z_p, z_a.t()) / self.temperature

        # ── Mask: positive pair is on the diagonal ──
        # target[i] = i  (the positive for anchor i is positive i)
        targets = torch.arange(B, device=device)

        if labels is not None:
            # Supervised variant: same-class samples are also positives
            # Mask out same-class pairs from denominator penalty
            # But the primary positive remains the altitude-scaled version
            label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
            # Zero out same-class off-diagonal similarities so they
            # don't contribute as hard negatives
            # (optional — standard InfoNCE doesn't do this, but it helps
            #  when the batch contains multiple samples of the same class)
            same_class_mask = label_mask.float() - torch.eye(B, device=device)
            sim_ap = sim_ap - same_class_mask * 1e9  # mask same-class negatives
            sim_pa = sim_pa - same_class_mask * 1e9

        # ── Cross-entropy: each row's positive is on the diagonal ──
        loss_ap = F.cross_entropy(sim_ap, targets)
        loss_pa = F.cross_entropy(sim_pa, targets)

        return (loss_ap + loss_pa) / 2.0


# ════════════════════════════════════════════════════════════════════════
# SCTE Projection Head
# ════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Maps the 64-dim [CLS] embedding to a lower-dim space where the
    InfoNCE loss is applied.  Following SimCLR/BYOL convention, the
    projection head is discarded after training — only the encoder
    embeddings are used downstream.

    Architecture: 64 → 64 → 32  (BN + ReLU between layers)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════
# SCTE Model (Encoder + Projection Head)
# ════════════════════════════════════════════════════════════════════════

class SCTEModel(nn.Module):
    """Scale-Contrastive Trajectory Embedding model.

    Wraps the TrajMAE encoder with a projection head for InfoNCE training.

    The encoder produces a 64-dim [CLS] embedding (the *representation*).
    The projection head maps it to 32-dim (the *projection*) where
    contrastive loss is computed.  After training, only the 64-dim
    representation is used — the projection head is discarded.
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        proj_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 50,
    ):
        super().__init__()
        from evaluation.traj_mae import TrajMAEEncoder

        self.encoder = TrajMAEEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.projection = ProjectionHead(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=proj_dim,
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 64-dim [CLS] representation (no projection).

        This is what's used downstream after SCTE pre-training.
        """
        tokens, pad_mask = self.encoder.embed_tokens(x)
        encoded = self.encoder(tokens, padding_mask=pad_mask)
        return self.encoder.cls_embedding(encoded)  # (B, 64)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both representation and projection.

        Returns
        -------
        embedding : (B, 64) — CLS representation (used downstream).
        projection : (B, 32) — projected embedding (used for InfoNCE).
        """
        embedding = self.get_embedding(x)              # (B, 64)
        projection = self.projection(embedding)        # (B, 32)
        return embedding, projection


# ════════════════════════════════════════════════════════════════════════
# Data Generation
# ════════════════════════════════════════════════════════════════════════

def generate_scte_data(
    n_per_class: int = 300,
    noise_std: float = 0.003,
    max_len: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate trajectory sequences for SCTE training.

    Returns (sequences, labels).
    """
    from evaluation.trajectory_transformer import generate_full_dataset
    X_seq, _, y = generate_full_dataset(
        n_per_class=n_per_class, noise_std=noise_std, max_len=max_len
    )
    return X_seq, y


# ════════════════════════════════════════════════════════════════════════
# SCTE Training Loop
# ════════════════════════════════════════════════════════════════════════

def train_scte(
    model: SCTEModel,
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    temperature: float = 0.07,
    source_alt: float = 50.0,
    altitude_range: Tuple[float, float] = (30.0, 300.0),
    device: Optional[torch.device] = None,
) -> Tuple[List[float], Dict]:
    """Train SCTE encoder via InfoNCE on altitude-scaled pairs.

    Parameters
    ----------
    model : SCTEModel
    train_sequences : (N, T, 4) numpy array.
    train_labels : (N,) numpy array of int labels.
    epochs : int
    batch_size : int
    lr : float
    temperature : float — InfoNCE temperature τ.
    source_alt : float — base altitude of training trajectories.
    altitude_range : (lo, hi) — range for random altitude scaling.
    device : torch.device, optional

    Returns
    -------
    losses : list of per-epoch mean losses.
    metrics : dict with training metrics.
    """
    if device is None:
        device = _get_device()
    model = model.to(device)

    dataset = SCTEPairDataset(
        train_sequences, train_labels,
        source_alt=source_alt,
        altitude_range=altitude_range,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )

    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for anchors, positives, labels in dataloader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            labels = labels.to(device)

            # Forward both views through shared encoder
            _, z_a = model(anchors)
            _, z_p = model(positives)

            # InfoNCE loss
            loss = criterion(z_a, z_p, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    [SCTE] epoch {epoch + 1:>3d}/{epochs}  "
                  f"InfoNCE = {avg_loss:.4f}  τ = {temperature}")

    metrics = {
        "final_loss": losses[-1] if losses else 0.0,
        "epochs": epochs,
        "temperature": temperature,
        "batch_size": batch_size,
    }
    return losses, metrics


# ════════════════════════════════════════════════════════════════════════
# Evaluation: Altitude Invariance
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_altitude_invariance(
    model: SCTEModel,
    sequences: np.ndarray,
    labels: np.ndarray,
    source_alt: float = 50.0,
    test_altitudes: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """Measure how well embeddings cluster by action (not altitude).

    For each trajectory, compute embeddings at multiple altitudes.
    Then measure:
    1. Intra-action cosine similarity across altitudes (should be HIGH)
    2. Inter-action cosine similarity (should be LOW)
    3. Altitude retrieval accuracy: for a query at alt_1, can we find
       the same trajectory at alt_2 using nearest-neighbour?

    Returns a dict of metrics.
    """
    if device is None:
        device = _get_device()
    model = model.to(device)
    model.eval()

    if test_altitudes is None:
        test_altitudes = [30.0, 50.0, 100.0, 200.0, 300.0]

    rng = np.random.default_rng(123)
    N = min(len(sequences), 200)  # subsample for speed
    idx = rng.choice(len(sequences), N, replace=False)
    sub_seq = sequences[idx]
    sub_labels = labels[idx]

    # Compute embeddings at each altitude
    alt_embeddings = {}  # alt → (N, 64)
    for alt in test_altitudes:
        scaled_seqs = np.stack([
            altitude_scale_trajectory(sub_seq[i], source_alt, alt, rng=rng)
            for i in range(N)
        ])
        x_t = torch.FloatTensor(scaled_seqs).to(device)
        embs = model.get_embedding(x_t).cpu().numpy()
        # L2-normalise
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        alt_embeddings[alt] = embs / np.maximum(norms, 1e-8)

    # ── Metric 1: Intra-action cross-altitude similarity ──
    # For each trajectory, compute cosine sim between its embedding at
    # different altitudes.  High = altitude-invariant.
    cross_alt_sims = []
    for i in range(N):
        embs_i = [alt_embeddings[alt][i] for alt in test_altitudes]
        for a_idx in range(len(embs_i)):
            for b_idx in range(a_idx + 1, len(embs_i)):
                sim = float(np.dot(embs_i[a_idx], embs_i[b_idx]))
                cross_alt_sims.append(sim)
    mean_cross_alt_sim = float(np.mean(cross_alt_sims))

    # ── Metric 2: Intra-class vs inter-class similarity ──
    # At a fixed altitude, measure within-class vs between-class cosine sim
    ref_embs = alt_embeddings[source_alt]  # (N, 64)
    sim_matrix = ref_embs @ ref_embs.T     # (N, N)

    intra_sims, inter_sims = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if sub_labels[i] == sub_labels[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0

    # ── Metric 3: Cross-altitude retrieval accuracy ──
    # For each trajectory at alt_query, find its nearest neighbour
    # at alt_gallery.  Correct if it's the same trajectory.
    retrieval_results = {}
    for alt_q, alt_g in [(50.0, 200.0), (50.0, 300.0), (30.0, 200.0)]:
        if alt_q not in alt_embeddings or alt_g not in alt_embeddings:
            continue
        embs_q = alt_embeddings[alt_q]  # (N, 64)
        embs_g = alt_embeddings[alt_g]  # (N, 64)
        sim = embs_q @ embs_g.T         # (N, N)
        # For each query, find top-1 match in gallery
        top1 = sim.argmax(axis=1)
        correct = (top1 == np.arange(N)).sum()
        retrieval_results[f"{int(alt_q)}m→{int(alt_g)}m"] = {
            "top1_acc": round(float(correct / N), 4),
            "correct": int(correct),
            "total": N,
        }

    # ── Metric 4: Per-altitude-pair similarity ──
    alt_pair_sims = {}
    for alt1 in test_altitudes:
        for alt2 in test_altitudes:
            if alt2 <= alt1:
                continue
            sims = [
                float(np.dot(alt_embeddings[alt1][i], alt_embeddings[alt2][i]))
                for i in range(N)
            ]
            key = f"{int(alt1)}m_vs_{int(alt2)}m"
            alt_pair_sims[key] = round(float(np.mean(sims)), 4)

    return {
        "cross_altitude_similarity": round(mean_cross_alt_sim, 4),
        "intra_class_similarity": round(mean_intra, 4),
        "inter_class_similarity": round(mean_inter, 4),
        "class_separation": round(mean_intra - mean_inter, 4),
        "retrieval": retrieval_results,
        "altitude_pair_similarities": alt_pair_sims,
        "n_samples": N,
        "test_altitudes": test_altitudes,
    }


# ════════════════════════════════════════════════════════════════════════
# Downstream Classification Probe
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def linear_probe(
    model: SCTEModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict:
    """Train a linear classifier on frozen SCTE embeddings.

    This measures how good the 64-dim [CLS] embedding is for
    action classification after contrastive pre-training.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if device is None:
        device = _get_device()
    model = model.to(device)
    model.eval()

    # Extract embeddings
    def get_embs(X):
        x_t = torch.FloatTensor(X).to(device)
        embs = []
        for start in range(0, len(X), 64):
            batch = x_t[start:start + 64]
            embs.append(model.get_embedding(batch).cpu().numpy())
        return np.concatenate(embs, axis=0)

    Z_train = get_embs(X_train)
    Z_test = get_embs(X_test)

    # Logistic regression on 64-dim embeddings
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    return {
        "linear_probe_acc": round(acc, 4),
        "linear_probe_f1": round(f1, 4),
        "embedding_dim": Z_train.shape[1],
    }


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    n_per_class: int = 300,
    epochs: int = 100,
    batch_size: int = 128,
    temperature: float = 0.07,
    lr: float = 1e-3,
) -> Dict:
    """Full SCTE pipeline: generate → train → evaluate → plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"\n{'═' * 68}")
    print(f"  SCTE — Scale-Contrastive Trajectory Embedding")
    print(f"{'═' * 68}")
    print(f"  Device:      {device}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Epochs:      {epochs}")

    # ── 1. Generate data ──
    print(f"\n  [1/5] Generating trajectories ({n_per_class}/class)...")
    X_seq, y = generate_scte_data(n_per_class=n_per_class)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, X_te = X_seq[train_idx], X_seq[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"    Train: {len(X_tr)}, Test: {len(X_te)}")

    # ── 2. Build model ──
    print(f"\n  [2/5] Building SCTE model...")
    model = SCTEModel(d_model=64, proj_dim=32, dropout=0.1)
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    proj_p = sum(p.numel() for p in model.projection.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    print(f"    Encoder:    {enc_p:>7,} params")
    print(f"    Projection: {proj_p:>7,} params")
    print(f"    Total:      {total_p:>7,} params")

    # ── 3. Contrastive pre-training ──
    print(f"\n  [3/5] SCTE contrastive training (InfoNCE, τ={temperature})...")
    t0 = time.time()
    losses, train_metrics = train_scte(
        model, X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        device=device,
    )
    train_time = time.time() - t0
    print(f"    Training: {train_time:.1f}s, "
          f"final InfoNCE = {losses[-1]:.4f}")

    # ── 4. Evaluate altitude invariance ──
    print(f"\n  [4/5] Evaluating altitude invariance...")
    inv_metrics = evaluate_altitude_invariance(
        model, X_te, y_te,
        source_alt=50.0,
        test_altitudes=[30.0, 50.0, 100.0, 200.0, 300.0],
        device=device,
    )

    print(f"    Cross-altitude similarity: {inv_metrics['cross_altitude_similarity']:.4f}")
    print(f"    Intra-class similarity:    {inv_metrics['intra_class_similarity']:.4f}")
    print(f"    Inter-class similarity:    {inv_metrics['inter_class_similarity']:.4f}")
    print(f"    Class separation:          {inv_metrics['class_separation']:.4f}")
    if inv_metrics["retrieval"]:
        print(f"    Cross-altitude retrieval:")
        for pair, res in inv_metrics["retrieval"].items():
            print(f"      {pair}: top-1 = {res['top1_acc']:.1%}")

    # ── 4b. Baseline: untrained encoder ──
    print(f"\n    Baseline (untrained encoder)...")
    baseline_model = SCTEModel(d_model=64, proj_dim=32, dropout=0.1).to(device)
    baseline_inv = evaluate_altitude_invariance(
        baseline_model, X_te, y_te,
        source_alt=50.0,
        test_altitudes=[30.0, 50.0, 100.0, 200.0, 300.0],
        device=device,
    )
    print(f"    Baseline cross-alt sim: {baseline_inv['cross_altitude_similarity']:.4f}")
    print(f"    Baseline class sep:     {baseline_inv['class_separation']:.4f}")

    # ── 5. Linear probe ──
    print(f"\n  [5/5] Linear probe (frozen encoder → LogisticRegression)...")
    probe_results = linear_probe(model, X_tr, y_tr, X_te, y_te, device=device)
    baseline_probe = linear_probe(baseline_model, X_tr, y_tr, X_te, y_te, device=device)
    print(f"    SCTE linear probe:     acc = {probe_results['linear_probe_acc']:.1%}  "
          f"F1 = {probe_results['linear_probe_f1']:.3f}")
    print(f"    Baseline linear probe: acc = {baseline_probe['linear_probe_acc']:.1%}  "
          f"F1 = {baseline_probe['linear_probe_f1']:.3f}")

    # ── Save results ──
    all_results = {
        "scte_training": {
            "final_info_nce": round(losses[-1], 4),
            "epochs": epochs,
            "temperature": temperature,
            "batch_size": batch_size,
            "train_time_s": round(train_time, 1),
        },
        "altitude_invariance": inv_metrics,
        "baseline_invariance": baseline_inv,
        "linear_probe_scte": probe_results,
        "linear_probe_baseline": baseline_probe,
        "improvement": {
            "cross_alt_sim_delta": round(
                inv_metrics["cross_altitude_similarity"]
                - baseline_inv["cross_altitude_similarity"], 4
            ),
            "class_sep_delta": round(
                inv_metrics["class_separation"]
                - baseline_inv["class_separation"], 4
            ),
            "probe_acc_delta": round(
                probe_results["linear_probe_acc"]
                - baseline_probe["linear_probe_acc"], 4
            ),
        },
        "model": {
            "encoder_params": enc_p,
            "projection_params": proj_p,
            "cls_embedding_dim": 64,
            "projection_dim": 32,
        },
    }
    with open(RESULTS_DIR / "scte_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Results → scte_results.json")

    # ════════════════════════════════════════════════════════════════════
    # Publication-quality figure
    # ════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # Panel 1: InfoNCE loss curve
    ax = axes[0, 0]
    ax.plot(range(1, len(losses) + 1), losses, color="#e74c3c", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("InfoNCE Loss", fontsize=11)
    ax.set_title("SCTE Contrastive Training\n"
                 f"(τ = {temperature}, batch = {batch_size})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: Cross-altitude similarity (SCTE vs baseline)
    ax = axes[0, 1]
    alt_pairs = list(inv_metrics["altitude_pair_similarities"].keys())
    scte_sims = [inv_metrics["altitude_pair_similarities"][k] for k in alt_pairs]
    base_sims = [baseline_inv["altitude_pair_similarities"].get(k, 0) for k in alt_pairs]

    x = np.arange(len(alt_pairs))
    w = 0.35
    bars1 = ax.bar(x - w / 2, scte_sims, w, label="SCTE (trained)",
                   color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + w / 2, base_sims, w, label="Baseline (untrained)",
                   color="#e74c3c", alpha=0.65)
    for bar, val in zip(bars1, scte_sims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, base_sims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=7, color="#999")

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_vs_", "\nvs ") for k in alt_pairs],
                       fontsize=8)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Same Trajectory Across Altitudes\n"
                 "(Higher = More Altitude-Invariant)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.15)

    # Panel 3: 2D embedding visualisation (t-SNE)
    ax = axes[1, 0]
    try:
        from sklearn.manifold import TSNE

        # Get embeddings at two altitudes
        model.eval()
        rng = np.random.default_rng(42)
        n_vis = min(100, len(X_te))
        vis_idx = rng.choice(len(X_te), n_vis, replace=False)

        embs_50, embs_200 = [], []
        vis_labels = y_te[vis_idx]
        for i in vis_idx:
            s50 = altitude_scale_trajectory(X_te[i], 50.0, 50.0, rng=rng)
            s200 = altitude_scale_trajectory(X_te[i], 50.0, 200.0, rng=rng)
            embs_50.append(s50)
            embs_200.append(s200)

        with torch.no_grad():
            z50 = model.get_embedding(
                torch.FloatTensor(np.stack(embs_50)).to(device)
            ).cpu().numpy()
            z200 = model.get_embedding(
                torch.FloatTensor(np.stack(embs_200)).to(device)
            ).cpu().numpy()

        all_embs = np.concatenate([z50, z200], axis=0)
        all_labs = np.concatenate([vis_labels, vis_labels])
        all_alts = np.array([50] * n_vis + [200] * n_vis)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        coords = tsne.fit_transform(all_embs)

        action_colors = {
            0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12",
            4: "#9b59b6", 5: "#1abc9c", 6: "#e67e22",
        }
        # 50m = circles, 200m = triangles
        for lab in range(len(SAR_ACTIONS)):
            mask_50 = (all_labs == lab) & (all_alts == 50)
            mask_200 = (all_labs == lab) & (all_alts == 200)
            ax.scatter(coords[mask_50, 0], coords[mask_50, 1],
                       c=action_colors[lab], marker="o", s=40, alpha=0.7,
                       label=f"{SAR_ACTIONS[lab]} (50m)" if lab < 3 else None)
            ax.scatter(coords[mask_200, 0], coords[mask_200, 1],
                       c=action_colors[lab], marker="^", s=40, alpha=0.5)

        ax.set_title("t-SNE of SCTE Embeddings\n(●=50m altitude, ▲=200m)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.2)
    except Exception as e:
        ax.text(0.5, 0.5, f"t-SNE unavailable\n({type(e).__name__})",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.set_title("t-SNE Embedding")
        print(f"    ⚠ t-SNE skipped: {e}")

    # Panel 4: Summary metrics comparison
    ax = axes[1, 1]
    metric_names = [
        "Cross-Alt\nSimilarity",
        "Class\nSeparation",
        "Linear Probe\nAccuracy",
    ]
    scte_vals = [
        inv_metrics["cross_altitude_similarity"],
        inv_metrics["class_separation"],
        probe_results["linear_probe_acc"],
    ]
    base_vals = [
        baseline_inv["cross_altitude_similarity"],
        baseline_inv["class_separation"],
        baseline_probe["linear_probe_acc"],
    ]
    x = np.arange(len(metric_names))
    bars1 = ax.bar(x - w / 2, scte_vals, w, label="SCTE", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + w / 2, base_vals, w, label="Baseline", color="#95a5a6", alpha=0.65)
    for bar, val in zip(bars1, scte_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.02,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, base_vals):
        y_pos = val + 0.02 if val >= 0 else val - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, max(y_pos, 0.01),
                f"{val:.3f}", ha="center", fontsize=8, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("SCTE vs Untrained Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Scale-Contrastive Trajectory Embedding (SCTE)\n"
        "Altitude-Invariant Embeddings via InfoNCE",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scte.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ scte.png")

    # ── Summary ──
    print(f"\n{'═' * 68}")
    print(f"  SUMMARY")
    print(f"{'═' * 68}")
    print(f"  {'Metric':<30} {'SCTE':>10} {'Baseline':>10} {'Δ':>8}")
    print(f"  {'-' * 60}")
    rows = [
        ("Cross-alt similarity", inv_metrics["cross_altitude_similarity"],
         baseline_inv["cross_altitude_similarity"]),
        ("Intra-class similarity", inv_metrics["intra_class_similarity"],
         baseline_inv["intra_class_similarity"]),
        ("Inter-class similarity", inv_metrics["inter_class_similarity"],
         baseline_inv["inter_class_similarity"]),
        ("Class separation", inv_metrics["class_separation"],
         baseline_inv["class_separation"]),
        ("Linear probe accuracy", probe_results["linear_probe_acc"],
         baseline_probe["linear_probe_acc"]),
    ]
    for name, scte_v, base_v in rows:
        delta = scte_v - base_v
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<30} {scte_v:>9.4f} {base_v:>9.4f} {sign}{delta:>7.4f}")

    if inv_metrics["retrieval"]:
        print(f"\n  Cross-altitude retrieval (top-1 accuracy):")
        for pair, res in inv_metrics["retrieval"].items():
            print(f"    {pair}: {res['top1_acc']:.1%}")

    print(f"\n  ✅ Done\n")
    return all_results


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SCTE — Scale-Contrastive Trajectory Embedding"
    )
    parser.add_argument("--n_per_class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature τ (default 0.07)")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_full_pipeline(
        n_per_class=args.n_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
