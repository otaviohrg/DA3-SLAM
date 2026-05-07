"""
DA3 inference wrapper.

Thin layer over the Depth Anything 3 API that normalises outputs into
a consistent, pipeline-friendly format:
  - extrinsics padded to (N, 4, 4)
  - confidence normalised to [0, 1]
  - depth, confidence, extrinsics, intrinsics all as float32 numpy arrays
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

@dataclass
class DepthPrediction:
    """Normalised outputs from a single DA3 inference call."""

    # (N, H, W) float32 — metric depth in metres
    depth: np.ndarray

    # (N, H, W) float32 — confidence in [0, 1]
    confidence: np.ndarray

    # (N, 4, 4) float32 — world-to-cam extrinsics
    extrinsics: np.ndarray

    # (N, 3, 3) float32 — camera intrinsic matrices
    intrinsics: np.ndarray

    # (N, H, W, 3) uint8 — images at DA3's processed resolution
    processed_images: np.ndarray

    # Number of frames
    @property
    def n_frames(self) -> int:
        return self.depth.shape[0]

    def confidence_mask(self, percentile: float = 65.0) -> np.ndarray:
        """
        Boolean mask (N, H, W) keeping pixels above `percentile` confidence.

        The threshold is computed **globally** across all frames in the batch
        so that consistently low-quality frames contribute fewer points than
        high-quality ones (per-frame normalisation would always keep the same
        fraction regardless of actual quality).
        """
        threshold = float(np.percentile(self.confidence, percentile))
        return self.confidence >= threshold

    def to_pointcloud(
        self, frame_idx: int, confidence_percentile: float = 65.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lift a single depth frame to a 3D point cloud in camera space.

        Only pixels whose confidence is in the top (100 - confidence_percentile)%
        across the entire submap batch are included.  Also discards pixels
        with non-positive or non-finite depth.

        Returns:
            points: (M, 3) float32 — 3D points in camera coordinates
            mask:   (H, W) bool   — pixels that were kept
        """
        K = self.intrinsics[frame_idx]
        depth = self.depth[frame_idx]

        high_confidence = self.confidence_mask(confidence_percentile)[frame_idx]
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        mask = high_confidence & valid_depth

        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = depth[mask]
        x = (u[mask] - cx) * z / fx
        y = (v[mask] - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).astype(np.float32)
        return points, mask


class DepthEstimator:
    """Wraps Depth Anything 3 for use in the SLAM pipeline."""

    def __init__(
        self,
        model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE",
        process_resolution: int = 504,
        device: torch.device | None = None,
    ):

        self.process_resolution = process_resolution
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[DepthEstimator] Loading {model_id} on {self.device}...")
        self.model = DepthAnything3.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("[DepthEstimator] Ready.")

    @torch.no_grad()
    def infer(self, images: list[str | np.ndarray]) -> DepthPrediction:
        """
        Run DA3 on a batch of images.

        Args:
            images: list of file paths (recommended) or HxWx3 uint8 numpy arrays.
                    Note: numpy array inputs may trigger CUDA nvrtc JIT compilation
                    on small batches; prefer file paths in the pipeline.

        Returns:
            DepthPrediction with normalised outputs
        """
        raw = self.model.inference(images, process_res=self.process_resolution)

        depth = raw.depth.astype(np.float32)           # (N, H, W)
        confidence = _normalize_confidence(raw.conf.astype(np.float32))  # (N, H, W) → [0,1]
        extrinsics = _pad_extrinsics(raw.extrinsics.astype(np.float32))  # (N,3,4)→(N,4,4)
        intrinsics = raw.intrinsics.astype(np.float32)  # (N, 3, 3)

        return DepthPrediction(
            depth=depth,
            confidence=confidence,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            processed_images=raw.processed_images,
        )


# ── internal helpers ──────────────────────────────────────────────────────────

def _normalize_confidence(confidence: np.ndarray) -> np.ndarray:
    """
    Normalise confidence to [0, 1] per frame.
    DA3 returns raw logit-like scores (observed range: ~1–13).
    """
    out = np.empty_like(confidence)
    for i, frame_conf in enumerate(confidence):
        lo, hi = frame_conf.min(), frame_conf.max()
        out[i] = (frame_conf - lo) / (hi - lo) if hi > lo else np.ones_like(frame_conf)
    return out


def _pad_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """
    Pad (N, 3, 4) extrinsic matrices to (N, 4, 4) by appending [0, 0, 0, 1].
    """
    N = extrinsics.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (N, 1, 1))
    return np.concatenate([extrinsics, bottom], axis=1)
