"""
DA3 inference wrapper.

Thin layer over the Depth Anything 3 API that normalises outputs into
a consistent, pipeline-friendly format:
  - extrinsics padded to (N, 4, 4)
  - confidence normalised to [0, 1]
  - depth, conf, extrinsics, intrinsics all as float32 numpy arrays
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DepthPrediction:
    """Normalised outputs from a single DA3 inference call."""

    # (N, H, W) float32 — metric depth in metres
    depth: np.ndarray

    # (N, H, W) float32 — confidence in [0, 1]
    conf: np.ndarray

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

    def confident_mask(self, percentile: float = 40.0) -> np.ndarray:
        """
        Boolean mask (N, H, W) keeping pixels above `percentile` confidence.
        Mirrors the filtering strategy used in VGGT-SLAM.
        """
        threshold = np.percentile(self.conf, percentile, axis=(1, 2), keepdims=True)
        return self.conf >= threshold

    def to_pointcloud(
        self, frame_idx: int, conf_percentile: float = 40.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lift a single depth frame to a 3D point cloud in camera space.

        Returns:
            points: (M, 3) float32 — 3D points in camera coordinates
            mask:   (H, W) bool   — pixels that were kept
        """
        K = self.intrinsics[frame_idx]
        depth = self.depth[frame_idx]
        mask = self.confident_mask(conf_percentile)[frame_idx]

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
        process_res: int = 504,
        device: torch.device | None = None,
    ):
        from depth_anything_3.api import DepthAnything3

        self.process_res = process_res
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
        raw = self.model.inference(images, process_res=self.process_res)

        depth = raw.depth.astype(np.float32)           # (N, H, W)
        conf = _normalise_conf(raw.conf.astype(np.float32))  # (N, H, W) → [0,1]
        extrinsics = _pad_extrinsics(raw.extrinsics.astype(np.float32))  # (N,3,4)→(N,4,4)
        intrinsics = raw.intrinsics.astype(np.float32)  # (N, 3, 3)

        return DepthPrediction(
            depth=depth,
            conf=conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            processed_images=raw.processed_images,
        )


# ── internal helpers ──────────────────────────────────────────────────────────

def _normalise_conf(conf: np.ndarray) -> np.ndarray:
    """
    Normalise confidence to [0, 1] per frame.
    DA3 returns raw logit-like scores (observed range: ~1–13).
    """
    out = np.empty_like(conf)
    for i, c in enumerate(conf):
        lo, hi = c.min(), c.max()
        out[i] = (c - lo) / (hi - lo) if hi > lo else np.ones_like(c)
    return out


def _pad_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """
    Pad (N, 3, 4) extrinsic matrices to (N, 4, 4) by appending [0, 0, 0, 1].
    """
    N = extrinsics.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (N, 1, 1))
    return np.concatenate([extrinsics, bottom], axis=1)
