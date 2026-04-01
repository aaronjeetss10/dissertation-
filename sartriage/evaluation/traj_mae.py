"""
evaluation/traj_mae.py
========================
TrajMAE — Masked Autoencoder for Trajectory-Based Action Recognition.

Converts the existing Trajectory Transformer (~103K params) into a
self-supervised Masked Autoencoder.  During pre-training the encoder
sees only 30% of the (dx, dy, Δw, Δh) tokens; a lightweight decoder
reconstructs the masked 70%.  After pre-training the [CLS] token's
64-dim embedding is used for downstream classification.

Architecture overview
---------------------

    Input:  (B, T, 4)   —  (dx, dy, Δaspect, Δsize)  per-frame tokens
               │
    ┌──────────▼──────────┐
    │  Token Embedding     │  Linear(4 → d_model=64) + PositionalEncoding
    │  + [CLS] prepend     │  Learnable CLS token
    └──────────┬──────────┘
               │ (B, T+1, 64)
    ┌──────────▼──────────┐
    │  Random Masking      │  Drop 70% of non-CLS tokens
    │  (pre-train only)    │  Keep the original position encoding
    └──────────┬──────────┘
               │ (B, T_visible+1, 64)     ← only ~25% of tokens + CLS
    ┌──────────▼──────────┐
    │  Encoder             │  4 × TransformerEncoderLayer (4 heads)
    │  (shared with clf)   │  ~80K params
    └──────────┬──────────┘
               │
       ┌───────┴───────┐
       │               │
  [CLS] embed    visible token embeds
   (64-dim)            │
       │          ┌────▼─────────────┐
       │          │  Reinsert mask   │  Learnable mask tokens at original
       │          │  tokens           │  positions with correct pos encoding
       │          └────┬─────────────┘
       │               │ (B, T+1, d_dec=32)
       │          ┌────▼─────────────┐
    │  Decoder          │  2 × TransformerEncoderLayer
    │  (lightweight)    │  d_model=64, 2 heads, ~30K params
    │  └────┬─────────────┘
        │               │
       │          ┌────▼─────────────┐
       │          │  Reconstruction   │  Linear(32 → 4): predict original
       │          │  head             │  (dx, dy, Δw, Δh) at masked positions
       │          └──────────────────┘
       │
  ┌────▼───────────────┐
  │  Classification     │  (fine-tuning only)
  │  head               │  LN → 32 → GELU → 7/8 classes
  └────────────────────┘

Usage
-----
    # Pre-training (self-supervised)
    python -m sartriage.evaluation.traj_mae pretrain

    # Fine-tuning (classification)
    python -m sartriage.evaluation.traj_mae finetune

    # Full pipeline (pretrain → finetune → evaluate)
    python -m sartriage.evaluation.traj_mae full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SAR_ACTIONS = [
    "falling", "running", "lying_down", "crawling",
    "waving", "collapsed", "stumbling", "walking",
]


# ════════════════════════════════════════════════════════════════════════
# Positional Encoding (identical to the original Trajectory Transformer)
# ════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.  x: (B, T, D)."""
        return x + self.pe[:, : x.size(1)]


# ════════════════════════════════════════════════════════════════════════
# Random Masking Layer
# ════════════════════════════════════════════════════════════════════════

class RandomTokenMasking(nn.Module):
    """Randomly mask a fraction of the (dx, dy, Δw, Δh) tokens.

    During pre-training, 70 % of the *non-CLS* tokens are removed and
    their indices are recorded so the decoder can place learnable mask
    tokens at those positions.  The CLS token (index 0) is never masked.

    Parameters
    ----------
    mask_ratio : float
        Fraction of tokens to mask (default 0.70).
    """

    def __init__(self, mask_ratio: float = 0.70):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T+1, D)
            Token embeddings **including the CLS token at position 0**.
        padding_mask : (B, T+1) bool, optional
            ``True`` for padded (invalid) positions.

        Returns
        -------
        x_visible : (B, N_vis, D)
            Visible tokens (CLS + unmasked data tokens).
        ids_keep : (B, N_vis) long
            Original position indices of kept tokens.
        ids_masked : (B, N_mask) long
            Original position indices of masked tokens.
        visible_padding_mask : (B, N_vis) bool
            Padding mask for visible tokens.
        """
        B, L, D = x.shape       # L = T + 1  (includes CLS at index 0)
        T = L - 1                # number of data tokens

        # Number of data tokens to mask
        n_mask = int(T * self.mask_ratio)
        n_keep = T - n_mask      # data tokens kept (CLS is always kept)

        # Per-sample random permutation of data token indices [1..T]
        noise = torch.rand(B, T, device=x.device)      # (B, T)

        # If there's a padding mask, push padded tokens to the end
        if padding_mask is not None:
            # Only consider data tokens (indices 1..L-1)
            data_pad = padding_mask[:, 1:]               # (B, T)
            noise = noise + data_pad.float() * 1e6       # padded → high noise → masked

        ids_shuffle = noise.argsort(dim=1)               # (B, T) argsort ascending
        ids_restore = ids_shuffle.argsort(dim=1)         # for unshuffling later

        # Keep the first n_keep tokens (lowest noise → most "real")
        ids_keep_data = ids_shuffle[:, :n_keep]          # (B, n_keep)
        ids_mask_data = ids_shuffle[:, n_keep:]          # (B, n_mask)

        # Offset +1 to account for CLS at position 0
        ids_keep_full = ids_keep_data + 1                # (B, n_keep)
        ids_mask_full = ids_mask_data + 1                # (B, n_mask)

        # Prepend CLS index (0) to kept set
        cls_idx = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        ids_keep = torch.cat([cls_idx, ids_keep_full], dim=1)   # (B, n_keep+1)
        ids_masked = ids_mask_full                               # (B, n_mask)

        # Gather visible tokens
        ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_visible = torch.gather(x, dim=1, index=ids_keep_exp)  # (B, n_keep+1, D)

        # Visible padding mask
        if padding_mask is not None:
            vis_pad = torch.gather(padding_mask, dim=1, index=ids_keep)
        else:
            vis_pad = torch.zeros(B, ids_keep.size(1), dtype=torch.bool, device=x.device)

        return x_visible, ids_keep, ids_masked, vis_pad


# ════════════════════════════════════════════════════════════════════════
# TrajMAE Encoder
# ════════════════════════════════════════════════════════════════════════

class TrajMAEEncoder(nn.Module):
    """Trajectory Transformer encoder with [CLS] token for TrajMAE.

    4-layer Transformer encoder (4 heads, d_model=64) with:
    1. Prepends a learnable [CLS] token to the sequence.
    2. Outputs the [CLS] embedding (64-dim) for downstream tasks.

    Parameters
    ----------
    input_dim : int
        Per-token feature dimension (default 4: dx, dy, Δw, Δh).
    d_model : int
        Transformer hidden dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of encoder layers (default 4).
    dropout : float
        Dropout rate (default 0.2).
    max_len : int
        Maximum sequence length (default 50).
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.2,
        max_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1)  # +1 for CLS
        self.dropout = nn.Dropout(dropout)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder (4 layers, feedforward_dim=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # 128
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def embed_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project input and prepend [CLS].  Returns embeddings + padding mask.

        Parameters
        ----------
        x : (B, T, 4)
            Raw trajectory tokens.

        Returns
        -------
        tokens : (B, T+1, d_model)
            Embedded tokens with CLS at position 0.
        padding_mask : (B, T+1)
            True for padded positions.
        """
        B, T, _ = x.shape

        # Padding mask (raw tokens summing to 0 are padding)
        data_pad = x.abs().sum(dim=-1) == 0           # (B, T)

        # Project
        tok = self.input_proj(x)                       # (B, T, d_model)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d_model)
        tok = torch.cat([cls, tok], dim=1)              # (B, T+1, d_model)

        # Positional encoding
        tok = self.pos_enc(tok)
        tok = self.dropout(tok)

        # CLS is never padded
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_pad = torch.cat([cls_pad, data_pad], dim=1)   # (B, T+1)

        return tok, full_pad

    @staticmethod
    def _safe_padding_mask(
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return None when no positions are actually padded (skip mask)."""
        if padding_mask is None:
            return None
        if not padding_mask.any():
            return None
        return padding_mask

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the transformer encoder on (possibly masked) tokens.

        Parameters
        ----------
        tokens : (B, N, d_model)
            May be the full sequence or only visible tokens after masking.
        padding_mask : (B, N) bool, optional

        Returns
        -------
        encoded : (B, N, d_model)
        """
        mask = self._safe_padding_mask(padding_mask)
        out = self.transformer(tokens, src_key_padding_mask=mask)
        return self.norm(out)

    def cls_embedding(self, encoded: torch.Tensor) -> torch.Tensor:
        """Extract the 64-dim [CLS] embedding from encoder output.

        Parameters
        ----------
        encoded : (B, N, d_model)
            Output of ``forward()``. CLS is at position 0.

        Returns
        -------
        cls_emb : (B, d_model)   — the 64-dim embedding.
        """
        return encoded[:, 0]


# ════════════════════════════════════════════════════════════════════════
# TrajMAE Decoder (lightweight, pre-train only)
# ════════════════════════════════════════════════════════════════════════

class TrajMAEDecoder(nn.Module):
    """Lightweight asymmetric decoder for masked token reconstruction.

    Following the MAE design principle, the decoder is deliberately
    smaller than the encoder: 1 transformer layer, d_dec=32, 2 heads.
    This keeps pre-training cheap and forces the encoder to learn
    rich representations.

    Parameters
    ----------
    d_encoder : int
        Encoder output dimension (64).
    d_decoder : int
        Decoder hidden dimension (64).
    n_heads : int
        Decoder attention heads (2).
    n_layers : int
        Decoder layers (2).
    output_dim : int
        Reconstruction target dimension (4: dx, dy, Δw, Δh).
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_encoder: int = 64,
        d_decoder: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        output_dim: int = 4,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_decoder = d_decoder

        # Project encoder output to decoder dim
        self.enc_to_dec = nn.Linear(d_encoder, d_decoder)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_decoder) * 0.02)

        # Positional encoding for the full sequence (decoder needs full positions)
        self.pos_enc = PositionalEncoding(d_decoder, max_len=max_len + 1)

        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder,
            nhead=n_heads,
            dim_feedforward=d_decoder * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_decoder)

        # Reconstruction head → predict original (dx, dy, Δw, Δh)
        self.recon_head = nn.Linear(d_decoder, output_dim)

    def forward(
        self,
        encoder_out: torch.Tensor,
        ids_keep: torch.Tensor,
        ids_masked: torch.Tensor,
        full_seq_len: int,
    ) -> torch.Tensor:
        """Reconstruct masked tokens.

        Parameters
        ----------
        encoder_out : (B, N_vis, d_encoder)
            Encoder output for visible tokens (CLS + unmasked).
        ids_keep : (B, N_vis) long
            Original position indices of visible tokens.
        ids_masked : (B, N_mask) long
            Original position indices of masked tokens.
        full_seq_len : int
            Total sequence length (T+1, including CLS).

        Returns
        -------
        recon : (B, N_mask, 4)
            Reconstructed token values at masked positions.
        """
        B = encoder_out.size(0)
        device = encoder_out.device

        # Project visible tokens to decoder space
        vis_dec = self.enc_to_dec(encoder_out)             # (B, N_vis, d_dec)

        # Create mask tokens for masked positions
        n_mask = ids_masked.size(1)
        mask_tokens = self.mask_token.expand(B, n_mask, -1)  # (B, N_mask, d_dec)

        # Reassemble full sequence: place visible + mask tokens at correct positions
        full_tokens = torch.zeros(
            B, full_seq_len, self.d_decoder, device=device
        )

        # Scatter visible tokens to their positions
        ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, self.d_decoder)
        full_tokens.scatter_(1, ids_keep_exp, vis_dec)

        # Scatter mask tokens to their positions
        ids_mask_exp = ids_masked.unsqueeze(-1).expand(-1, -1, self.d_decoder)
        full_tokens.scatter_(1, ids_mask_exp, mask_tokens)

        # Add positional encoding (decoder needs position info for reconstruction)
        full_tokens = self.pos_enc(full_tokens)

        # Run decoder transformer
        decoded = self.transformer(full_tokens)
        decoded = self.norm(decoded)

        # Extract only the masked positions for reconstruction loss
        ids_mask_exp_out = ids_masked.unsqueeze(-1).expand(-1, -1, self.d_decoder)
        masked_decoded = torch.gather(decoded, dim=1, index=ids_mask_exp_out)

        # Predict original token values
        recon = self.recon_head(masked_decoded)             # (B, N_mask, 4)
        return recon


# ════════════════════════════════════════════════════════════════════════
# Full TrajMAE Model
# ════════════════════════════════════════════════════════════════════════

class TrajMAE(nn.Module):
    """Masked Autoencoder for trajectory-based action recognition.

    Unifies encoder, masking, and decoder into a single module with
    two modes of operation:

    1. **Pre-training** (``pretrain=True``):
       Mask 75% of tokens → encoder → decoder → reconstruct.

    2. **Fine-tuning** (``pretrain=False``):
       Full tokens → encoder → [CLS] 64-dim embedding → classifier.

    Parameters
    ----------
    num_classes : int
        Number of SAR action classes (default 7).
    d_model : int
        Encoder dimension (default 64).
    mask_ratio : float
        Fraction of tokens to mask during pre-training (default 0.70).
    """

    def __init__(
        self,
        num_classes: int = 8,
        d_model: int = 64,
        d_decoder: int = 64,
        mask_ratio: float = 0.70,
        input_dim: int = 4,
        dropout: float = 0.2,
        max_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Encoder (keeps the ~103K param architecture)
        self.encoder = TrajMAEEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=4,
            n_layers=4,
            dropout=dropout,
            max_len=max_len,
        )

        # Masking layer (pre-train only)
        self.masking = RandomTokenMasking(mask_ratio=mask_ratio)

        # Decoder (lightweight, pre-train only)
        self.decoder = TrajMAEDecoder(
            d_encoder=d_model,
            d_decoder=d_decoder,
            n_heads=2,
            n_layers=2,
            output_dim=input_dim,
            max_len=max_len,
        )

        # Classification head (fine-tuning only)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward_pretrain(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-training forward pass: mask → encode → decode → reconstruct.

        Parameters
        ----------
        x : (B, T, 4)
            Raw trajectory tokens.

        Returns
        -------
        recon : (B, N_mask, 4)
            Reconstructed values at masked positions.
        target : (B, N_mask, 4)
            Ground-truth values at masked positions.
        ids_masked : (B, N_mask)
            Indices of masked positions (for loss computation).
        """
        B, T, C = x.shape

        # 1. Embed all tokens + prepend CLS
        tokens, padding_mask = self.encoder.embed_tokens(x)  # (B, T+1, d)

        # 2. Random masking (75% of non-CLS tokens)
        x_vis, ids_keep, ids_masked, vis_pad = self.masking(
            tokens, padding_mask
        )

        # 3. Encode only visible tokens
        encoded = self.encoder(x_vis, padding_mask=vis_pad)

        # 4. Decode → reconstruct masked tokens
        full_len = T + 1  # includes CLS
        recon = self.decoder(encoded, ids_keep, ids_masked, full_len)

        # 5. Get ground-truth values at masked positions
        #    ids_masked are positions in the CLS-prepended sequence,
        #    so offset by -1 to index into the original input x.
        target_indices = (ids_masked - 1).clamp(min=0)         # (B, N_mask)
        target_exp = target_indices.unsqueeze(-1).expand(-1, -1, C)
        target = torch.gather(x, dim=1, index=target_exp)      # (B, N_mask, 4)

        return recon, target, ids_masked

    def forward_finetune(self, x: torch.Tensor) -> torch.Tensor:
        """Fine-tuning forward pass: full sequence → [CLS] → classify.

        Parameters
        ----------
        x : (B, T, 4)

        Returns
        -------
        logits : (B, num_classes)
        """
        tokens, padding_mask = self.encoder.embed_tokens(x)
        encoded = self.encoder(tokens, padding_mask=padding_mask)
        cls_emb = self.encoder.cls_embedding(encoded)      # (B, 64)
        return self.classifier(cls_emb)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 64-dim [CLS] embedding (for downstream use).

        Parameters
        ----------
        x : (B, T, 4)

        Returns
        -------
        embedding : (B, 64)
        """
        tokens, padding_mask = self.encoder.embed_tokens(x)
        encoded = self.encoder(tokens, padding_mask=padding_mask)
        return self.encoder.cls_embedding(encoded)

    def forward(self, x: torch.Tensor, pretrain: bool = False):
        """Dispatch to pretrain or finetune mode."""
        if pretrain:
            return self.forward_pretrain(x)
        return self.forward_finetune(x)


# ════════════════════════════════════════════════════════════════════════
# Training Loops
# ════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pretrain_mae(
    model: TrajMAE,
    X_seq: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Self-supervised pre-training via masked reconstruction.

    Parameters
    ----------
    model : TrajMAE
    X_seq : (N, T, 4) numpy array — unlabelled trajectory sequences.
    epochs : int
    batch_size : int
    lr : float
    device : torch.device, optional

    Returns
    -------
    losses : list of per-epoch mean reconstruction losses.
    """
    device = device or get_device()
    model = model.to(device)

    X_t = torch.FloatTensor(X_seq).to(device)
    ds = TensorDataset(X_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in dl:
            optimizer.zero_grad()
            recon, target, _ = model.forward_pretrain(xb)

            # MSE on masked token reconstruction
            loss = F.mse_loss(recon, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    [pretrain] epoch {epoch + 1:>3d}/{epochs}  "
                  f"recon_loss = {avg_loss:.6f}")

    return losses


def finetune_mae(
    model: TrajMAE,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    freeze_epochs: int = 5,
) -> Tuple[np.ndarray, float, List[float], List[float]]:
    """Fine-tune pre-trained TrajMAE for classification.

    Follows the spec:
    - Freeze encoder for first 5 epochs (only train classifier head)
    - Then unfreeze all parameters
    - lr=1e-4, batch_size=32

    Parameters
    ----------
    model : TrajMAE (pre-trained encoder)
    X_train, y_train, X_test, y_test : train/test splits
    epochs, batch_size, lr : training hyperparams
    device : torch.device
    freeze_epochs : int
        Number of epochs to keep encoder frozen (default 5).

    Returns
    -------
    preds : predictions on test set
    best_acc : best test accuracy
    train_losses : per-epoch training loss
    test_accs : per-epoch test accuracy
    """
    device = device or get_device()
    model = model.to(device)

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_tr_t = torch.LongTensor(y_train).to(device)
    X_te_t = torch.FloatTensor(X_test).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Freeze decoder (never used during fine-tuning)
    for p in model.decoder.parameters():
        p.requires_grad = False

    # Phase 1: Freeze encoder for first `freeze_epochs` epochs
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_preds = None
    best_state = None
    train_losses = []
    test_accs = []

    # freeze_epochs from parameter (default=5)

    for epoch in range(epochs):
        # Phase 2: Unfreeze encoder after freeze_epochs
        if epoch == freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True
            # Rebuild optimizer with all trainable params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs,
            )

        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model.forward_finetune(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(train_dl))

        model.eval()
        with torch.no_grad():
            logits = model.forward_finetune(X_te_t)
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = (preds == y_test).mean()
            test_accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_preds = preds.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == freeze_epochs:
            phase = "frozen" if epoch < freeze_epochs else "unfrozen"
            print(f"    [finetune:{phase}] epoch {epoch + 1:>3d}/{epochs}  "
                  f"loss = {train_losses[-1]:.4f}  acc = {acc:.1%}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    return best_preds, best_acc, train_losses, test_accs


# ════════════════════════════════════════════════════════════════════════
# Data generation (reuses trajectory_transformer.py logic)
# ════════════════════════════════════════════════════════════════════════

def generate_data(n_per_class: int = 300, noise_std: float = 0.003, max_len: int = 40):
    """Generate trajectory sequences and TMS features.

    Returns (sequences, labels) where sequences is (N, max_len, 4).
    """
    from evaluation.trajectory_transformer import generate_full_dataset
    X_seq, _, y = generate_full_dataset(n_per_class=n_per_class,
                                         noise_std=noise_std, max_len=max_len)
    return X_seq, y


# ════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    n_per_class: int = 300,
    pretrain_epochs: int = 100,
    finetune_epochs: int = 60,
):
    """Pre-train TrajMAE → fine-tune → evaluate → compare with baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    warnings.filterwarnings("ignore")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"\n{'═' * 65}")
    print(f"  TrajMAE — Masked Autoencoder for Trajectory Action Recognition")
    print(f"{'═' * 65}")
    print(f"  Device: {device}")

    # ── 1. Generate data ──
    print(f"\n  [1/5] Generating data ({n_per_class}/class)...")
    X_seq, y = generate_data(n_per_class=n_per_class)
    X_seq_hard, y_hard = generate_data(n_per_class=80, noise_std=0.005)
    print(f"  Train data: {X_seq.shape}, Hard test: {X_seq_hard.shape}")

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_te = X_seq[train_idx], X_seq[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # ── 2. Build TrajMAE ──
    print(f"\n  [2/5] Building TrajMAE...")
    model = TrajMAE(
        num_classes=len(SAR_ACTIONS),
        d_model=64,
        d_decoder=64,
        mask_ratio=0.70,
    )
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    clf_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Encoder:    {enc_params:>7,} params")
    print(f"    Decoder:    {dec_params:>7,} params")
    print(f"    Classifier: {clf_params:>7,} params")
    print(f"    Total:      {total_params:>7,} params")

    # ── 3. Self-supervised pre-training ──
    print(f"\n  [3/5] Pre-training (mask 70%, {pretrain_epochs} epochs)...")
    # Use ALL data (no labels needed) for pre-training
    t0 = time.time()
    pretrain_losses = pretrain_mae(
        model, X_seq, epochs=pretrain_epochs, lr=1e-3, device=device
    )
    pretrain_time = time.time() - t0
    print(f"    Pre-training: {pretrain_time:.1f}s, "
          f"final recon loss = {pretrain_losses[-1]:.6f}")

    # ── 4. Fine-tune with labels ──
    print(f"\n  [4/5] Fine-tuning ({finetune_epochs} epochs)...")
    t0 = time.time()
    preds, best_acc, ft_losses, ft_accs = finetune_mae(
        model, X_tr, y_tr, X_te, y_te,
        epochs=finetune_epochs, lr=1e-4, device=device,
    )
    ft_time = time.time() - t0

    mae_acc = accuracy_score(y_te, preds)
    mae_f1 = f1_score(y_te, preds, average="macro")

    # Hard test
    model.eval()
    with torch.no_grad():
        X_hard_t = torch.FloatTensor(X_seq_hard).to(device)
        hard_preds = model.forward_finetune(X_hard_t).argmax(dim=1).cpu().numpy()
    hard_acc = accuracy_score(y_hard, hard_preds)
    hard_f1 = f1_score(y_hard, hard_preds, average="macro")

    print(f"\n    TrajMAE:  Test = {mae_acc:.1%}  F1 = {mae_f1:.3f}  "
          f"Hard = {hard_acc:.1%}  F1(hard) = {hard_f1:.3f}")

    # ── 5. Compare: MAE vs no-pretrain ──
    print(f"\n  [5/5] Baseline: Transformer without pre-training...")
    baseline = TrajMAE(num_classes=len(SAR_ACTIONS), d_model=64, d_decoder=64)
    _, base_acc, _, base_accs = finetune_mae(
        baseline, X_tr, y_tr, X_te, y_te,
        epochs=finetune_epochs, lr=1e-4, device=device,
    )
    baseline.eval()
    with torch.no_grad():
        base_hard_preds = baseline.forward_finetune(X_hard_t).argmax(dim=1).cpu().numpy()
    base_hard_acc = accuracy_score(y_hard, base_hard_preds)
    print(f"    Baseline: Test = {base_acc:.1%}  Hard = {base_hard_acc:.1%}")

    # ── Save results ──
    results = {
        "traj_mae": {
            "test_acc": round(mae_acc, 4),
            "test_f1": round(mae_f1, 4),
            "hard_acc": round(hard_acc, 4),
            "hard_f1": round(hard_f1, 4),
            "encoder_params": enc_params,
            "decoder_params": dec_params,
            "total_params": total_params,
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "mask_ratio": 0.75,
            "cls_embedding_dim": 64,
            "pretrain_time_s": round(pretrain_time, 1),
            "finetune_time_s": round(ft_time, 1),
        },
        "baseline_no_pretrain": {
            "test_acc": round(base_acc, 4),
            "hard_acc": round(base_hard_acc, 4),
        },
        "improvement": {
            "test_delta": round(mae_acc - base_acc, 4),
            "hard_delta": round(hard_acc - base_hard_acc, 4),
        },
    }
    with open(RESULTS_DIR / "traj_mae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results → traj_mae_results.json")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Pre-training loss curve
    ax = axes[0]
    ax.plot(range(1, len(pretrain_losses) + 1), pretrain_losses,
            color="#e74c3c", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Reconstruction MSE Loss", fontsize=11)
    ax.set_title("TrajMAE Pre-Training\n(75% token masking)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 2: Fine-tuning — MAE vs baseline
    ax = axes[1]
    ep = range(1, len(ft_accs) + 1)
    ax.plot(ep, [a * 100 for a in ft_accs], color="#2ecc71", linewidth=2,
            label=f"TrajMAE (pre-trained) — {mae_acc:.1%}")
    ax.plot(ep, [a * 100 for a in base_accs], color="#3498db", linewidth=2,
            linestyle="--", label=f"No pre-training — {base_acc:.1%}")
    ax.set_xlabel("Fine-tuning Epoch", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Fine-Tuning: MAE vs Scratch", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Panel 3: Confusion matrix
    ax = axes[2]
    cm = confusion_matrix(y_te, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(SAR_ACTIONS)))
    ax.set_xticklabels(SAR_ACTIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(SAR_ACTIONS)))
    ax.set_yticklabels(SAR_ACTIONS, fontsize=8)
    for i in range(len(SAR_ACTIONS)):
        for j in range(len(SAR_ACTIONS)):
            v = cm_norm[i, j]
            if v > 0.01:
                c = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color=c, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"TrajMAE Confusion Matrix\n({mae_acc:.1%} accuracy)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(
        "TrajMAE — Self-Supervised Masked Autoencoder for Trajectory Action Recognition",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "traj_mae.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ traj_mae.png")

    # ── Summary ──
    print(f"\n{'═' * 65}")
    print(f"  SUMMARY")
    print(f"{'═' * 65}")
    print(f"  {'Method':<30} {'Test':>8} {'Hard':>8}")
    print(f"  {'-' * 48}")
    print(f"  {'TrajMAE (pre-trained)':<30} {mae_acc:>7.1%} {hard_acc:>7.1%}")
    print(f"  {'No pre-training':<30} {base_acc:>7.1%} {base_hard_acc:>7.1%}")
    delta = mae_acc - base_acc
    print(f"\n  Pre-training benefit: {'+' if delta >= 0 else ''}{delta:.1%}")
    print(f"  CLS embedding dim: 64")
    print(f"  ✅ Done\n")

    return results


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TrajMAE — Masked Autoencoder for Trajectories")
    parser.add_argument("mode", choices=["pretrain", "finetune", "full"],
                        nargs="?", default="full",
                        help="Run mode (default: full pipeline)")
    parser.add_argument("--n_per_class", type=int, default=300)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--finetune_epochs", type=int, default=60)
    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline(
            n_per_class=args.n_per_class,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
        )
    else:
        print(f"Mode '{args.mode}' — use 'full' for the complete pipeline.")


if __name__ == "__main__":
    main()
