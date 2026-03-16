"""
core/attention_viz.py
======================
Grad-CAM & Temporal Attention visualization for MViTv2-S.

Produces heatmap overlays showing **exactly which spatial regions
and temporal frames** the model attends to when classifying an action.
This is critical for SAR explainability — rescuers need to see *why*
the AI flagged a person as "falling" vs "running."

Supports two visualization modes:
  1. **Spatial Grad-CAM**: Which image regions drive the classification
  2. **Temporal Attention**: Which frames in the clip matter most

Works with both MViTv2-S and R3D-18 backbones (auto-detects).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("sartriage.attention")


class GradCAMVideoExplainer:
    """Generate Grad-CAM heatmaps for video classification models.

    Hooks into the last feature layer before the classification head
    and computes gradient-weighted activation maps to show which
    spatial regions and temporal frames the model focuses on.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._model_type = self._detect_model_type()
        self._register_hooks()

    def _detect_model_type(self) -> str:
        """Detect whether this is MViTv2 or R3D-18."""
        if hasattr(self.model, "blocks"):
            return "mvit"
        elif hasattr(self.model, "layer4"):
            return "r3d"
        else:
            return "unknown"

    def _register_hooks(self) -> None:
        """Register forward/backward hooks on the target layer."""
        if self._model_type == "mvit":
            # Hook into the last transformer block's output
            # MViTv2: model.blocks[-1] is the last MultiscaleBlock
            target = self.model.blocks[-1]
        elif self._model_type == "r3d":
            # Hook into layer4 (last conv block)
            target = self.model.layer4
        else:
            log.warning("Unknown model type — Grad-CAM unavailable")
            return

        def forward_hook(module, input, output):
            if isinstance(output, (tuple, list)):
                self._activations = output[0].detach()
            else:
                self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (tuple, list)):
                self._gradients = grad_output[0].detach()
            else:
                self._gradients = grad_output.detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))
        log.info("Grad-CAM hooks registered on %s (%s)", type(target).__name__, self._model_type)

    def generate(
        self,
        clip_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate Grad-CAM heatmaps for a video clip.

        Parameters
        ----------
        clip_tensor : torch.Tensor
            Pre-processed clip, shape (1, 3, T, H, W).
        target_class : int, optional
            Class index to explain. If None, uses the predicted class.

        Returns
        -------
        dict with keys:
            - spatial_heatmap: (H, W) numpy array [0,1] — averaged over time
            - temporal_heatmap: (T,) numpy array [0,1] — per-frame importance
            - per_frame_heatmaps: list of (H, W) numpy arrays per frame
            - predicted_class: int
            - predicted_prob: float
            - top_k: list of (class_idx, prob) tuples
        """
        self.model.eval()
        clip_tensor = clip_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        logits = self.model(clip_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        predicted_prob = probs[0, target_class].item()

        # Top-k predictions
        topk_probs, topk_indices = probs[0].topk(min(5, probs.shape[1]))
        top_k = [(idx.item(), p.item()) for idx, p in zip(topk_indices, topk_probs)]

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            log.warning("Grad-CAM: no activations/gradients captured")
            return {
                "spatial_heatmap": np.zeros((224, 224)),
                "temporal_heatmap": np.zeros(16),
                "per_frame_heatmaps": [],
                "predicted_class": target_class,
                "predicted_prob": predicted_prob,
                "top_k": top_k,
            }

        activations = self._activations  # (B, ..., C) or (B, C, T', H', W')
        gradients = self._gradients

        if self._model_type == "mvit":
            heatmaps = self._gradcam_mvit(activations, gradients, clip_tensor.shape)
        elif self._model_type == "r3d":
            heatmaps = self._gradcam_r3d(activations, gradients, clip_tensor.shape)
        else:
            heatmaps = self._gradcam_fallback(clip_tensor.shape)

        heatmaps["predicted_class"] = target_class
        heatmaps["predicted_prob"] = predicted_prob
        heatmaps["top_k"] = top_k

        return heatmaps

    def _gradcam_mvit(
        self, activations: torch.Tensor, gradients: torch.Tensor, input_shape: tuple
    ) -> Dict[str, Any]:
        """Grad-CAM for MViTv2 (token-based transformer output)."""
        B, N, C = activations.shape  # B=1, N=num_tokens, C=channels

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, C)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=-1)  # (B, N)
        cam = F.relu(cam)  # only positive contributions

        # Remove CLS token if present
        cam = cam[0]  # (N,)

        # Infer spatial dimensions from token count
        # MViTv2-S last block typically has T'×H'×W' tokens
        T_in = input_shape[2]  # original T
        n_tokens = cam.shape[0]

        # Try to factor into T' × H' × W'
        t_prime, h_prime, w_prime = self._factor_tokens(n_tokens, T_in)

        cam_3d = cam.view(t_prime, h_prime, w_prime).cpu().numpy()

        # Resize each temporal slice to input spatial size
        H_in, W_in = input_shape[3], input_shape[4]
        per_frame = []
        for t in range(t_prime):
            frame_cam = cv2.resize(cam_3d[t], (W_in, H_in))
            frame_cam = (frame_cam - frame_cam.min()) / (frame_cam.max() - frame_cam.min() + 1e-8)
            per_frame.append(frame_cam)

        spatial_heatmap = np.mean(per_frame, axis=0)
        temporal_heatmap = np.array([frame.mean() for frame in per_frame])
        temporal_heatmap = temporal_heatmap / (temporal_heatmap.max() + 1e-8)

        return {
            "spatial_heatmap": spatial_heatmap,
            "temporal_heatmap": temporal_heatmap,
            "per_frame_heatmaps": per_frame,
        }

    def _gradcam_r3d(
        self, activations: torch.Tensor, gradients: torch.Tensor, input_shape: tuple
    ) -> Dict[str, Any]:
        """Grad-CAM for R3D-18 (3D CNN with shape B, C, T', H', W')."""
        # Global average pooling of gradients across spatial+temporal dims
        weights = gradients.mean(dim=[2, 3, 4], keepdim=True)  # (B, C, 1, 1, 1)

        cam = (weights * activations).sum(dim=1)  # (B, T', H', W')
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()  # (T', H', W')

        H_in, W_in = input_shape[3], input_shape[4]
        per_frame = []
        for t in range(cam.shape[0]):
            frame_cam = cv2.resize(cam[t], (W_in, H_in))
            frame_cam = (frame_cam - frame_cam.min()) / (frame_cam.max() - frame_cam.min() + 1e-8)
            per_frame.append(frame_cam)

        spatial_heatmap = np.mean(per_frame, axis=0)
        temporal_heatmap = np.array([f.mean() for f in per_frame])
        temporal_heatmap = temporal_heatmap / (temporal_heatmap.max() + 1e-8)

        return {
            "spatial_heatmap": spatial_heatmap,
            "temporal_heatmap": temporal_heatmap,
            "per_frame_heatmaps": per_frame,
        }

    def _gradcam_fallback(self, input_shape: tuple) -> Dict[str, Any]:
        H, W = input_shape[3], input_shape[4]
        return {
            "spatial_heatmap": np.zeros((H, W)),
            "temporal_heatmap": np.zeros(input_shape[2]),
            "per_frame_heatmaps": [],
        }

    @staticmethod
    def _factor_tokens(n: int, T_hint: int) -> Tuple[int, int, int]:
        """Factor N tokens into (T', H', W') spatial grid."""
        # MViTv2-S typical last-block resolutions
        for t in [T_hint, T_hint // 2, T_hint // 4, T_hint // 8, 1]:
            if n % t != 0:
                continue
            spatial = n // t
            s = int(spatial ** 0.5)
            if s * s == spatial:
                return t, s, s
            # Try rectangular
            for h in range(s, 0, -1):
                if spatial % h == 0:
                    return t, h, spatial // h

        # Ultimate fallback
        s = int(n ** 0.5)
        return 1, s, max(1, n // s)

    def cleanup(self) -> None:
        """Remove hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ── Overlay rendering ─────────────────────────────────────────────────────

def render_attention_overlay(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a heatmap on a video frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3).
    heatmap : np.ndarray
        Attention heatmap (H, W) with values in [0, 1].
    alpha : float
        Blend weight for the heatmap.

    Returns
    -------
    np.ndarray
        Blended BGR image.
    """
    h, w = frame.shape[:2]
    hmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
    hmap_uint8 = (hmap_resized * 255).clip(0, 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap_uint8, colormap)
    blended = cv2.addWeighted(frame, 1 - alpha, hmap_color, alpha, 0)
    return blended


def render_temporal_bar(
    temporal_heatmap: np.ndarray,
    width: int = 600,
    height: int = 60,
) -> np.ndarray:
    """Render a temporal importance bar chart.

    Returns a BGR image showing per-frame importance as a coloured bar.
    """
    n_frames = len(temporal_heatmap)
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)  # dark background

    bar_width = width // n_frames
    for i, val in enumerate(temporal_heatmap):
        x1 = i * bar_width
        x2 = (i + 1) * bar_width
        bar_height = int(val * (height - 10))

        # Colour: blue (low) → red (high)
        colour_val = int(val * 255)
        colour = cv2.applyColorMap(
            np.array([[colour_val]], dtype=np.uint8), cv2.COLORMAP_JET
        )[0, 0].tolist()

        cv2.rectangle(bar, (x1 + 1, height - bar_height - 5), (x2 - 1, height - 5), colour, -1)

        # Frame number
        cv2.putText(bar, str(i), (x1 + 2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    return bar


def save_attention_frames(
    event_dict: Dict[str, Any],
    clip_frames: List[np.ndarray],
    heatmaps: Dict[str, Any],
    output_dir: Path,
    event_id: str,
) -> Dict[str, str]:
    """Save Grad-CAM visualizations for an event.

    Generates:
      1. attention_<eid>.jpg — spatial heatmap overlaid on mid-frame
      2. temporal_<eid>.jpg — temporal importance bar
      3. attention_grid_<eid>.jpg — all frames with heatmaps in a grid

    Returns dict of filename → relative URL path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    spatial_hmap = heatmaps.get("spatial_heatmap")
    temporal_hmap = heatmaps.get("temporal_heatmap")
    per_frame = heatmaps.get("per_frame_heatmaps", [])

    # 1. Spatial heatmap on mid-frame
    if spatial_hmap is not None and clip_frames:
        mid_idx = len(clip_frames) // 2
        mid_frame = clip_frames[mid_idx].copy()
        overlay = render_attention_overlay(mid_frame, spatial_hmap, alpha=0.45)

        # Add label
        pred_class = heatmaps.get("predicted_class", -1)
        pred_prob = heatmaps.get("predicted_prob", 0)
        label = f"Grad-CAM | Predicted: class {pred_class} ({pred_prob:.1%})"
        cv2.putText(overlay, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        fname = f"attention_{event_id}.jpg"
        cv2.imwrite(str(output_dir / fname), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["attention_url"] = fname

    # 2. Temporal importance bar
    if temporal_hmap is not None and len(temporal_hmap) > 0:
        bar = render_temporal_bar(temporal_hmap, width=640, height=80)
        fname = f"temporal_{event_id}.jpg"
        cv2.imwrite(str(output_dir / fname), bar, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["temporal_url"] = fname

    # 3. Grid of all frames with heatmaps
    if per_frame and clip_frames and len(per_frame) == len(clip_frames):
        grid_size = 224
        n = len(clip_frames)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        grid = np.zeros((rows * grid_size, cols * grid_size, 3), dtype=np.uint8)

        for i in range(n):
            frame = cv2.resize(clip_frames[i], (grid_size, grid_size))
            overlay = render_attention_overlay(frame, per_frame[i], alpha=0.45)
            cv2.putText(overlay, f"t={i}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            r, c = i // cols, i % cols
            grid[r * grid_size:(r + 1) * grid_size,
                 c * grid_size:(c + 1) * grid_size] = overlay

        fname = f"attention_grid_{event_id}.jpg"
        cv2.imwrite(str(output_dir / fname), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved["attention_grid_url"] = fname

    return saved
