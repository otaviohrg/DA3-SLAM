"""
Submap construction.

A submap is a local map built from a batch of keyframes processed together
by DA3. It stores per-frame point clouds in both camera and world coordinates,
along with colours for visualisation.

The SubmapBuilder orchestrates:
  1. DA3 inference on the keyframe batch
  2. Confidence-filtered point cloud extraction per frame
  3. Camera-to-world transform using DA3's estimated extrinsics
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from da3_slam.backend.inference.depth_estimator import DepthEstimator, DepthPrediction


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """Single keyframe within a submap."""

    # Index of this frame within the original image sequence
    seq_idx: int

    # (H, W, 3) uint8 RGB — original image at DA3's processed resolution
    image: np.ndarray

    # (M, 3) float32 — 3D points in camera space
    points_cam: np.ndarray

    # (M, 3) float32 — 3D points in world space
    points_world: np.ndarray

    # (M, 3) uint8 — RGB colours corresponding to each point
    colors: np.ndarray

    # (4, 4) float32 — world-to-cam extrinsic matrix
    extrinsic: np.ndarray

    # (3, 3) float32 — camera intrinsic matrix
    intrinsic: np.ndarray

    @property
    def n_points(self) -> int:
        return len(self.points_world)

    @property
    def cam_to_world(self) -> np.ndarray:
        """(4, 4) cam-to-world transform (inverse of extrinsic)."""
        return np.linalg.inv(self.extrinsic)

    @property
    def position_world(self) -> np.ndarray:
        """(3,) camera centre in world coordinates."""
        return self.cam_to_world[:3, 3]


@dataclass
class Submap:
    """Local map built from a batch of keyframes."""

    # Position of this submap in the global sequence
    idx: int

    # Ordered list of frames in this submap
    frames: list[Frame] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def points_world(self) -> np.ndarray:
        """All world-space points concatenated — (N_total, 3)."""
        return np.concatenate([f.points_world for f in self.frames], axis=0)

    @property
    def colors(self) -> np.ndarray:
        """All colours concatenated — (N_total, 3) uint8."""
        return np.concatenate([f.colors for f in self.frames], axis=0)

    @property
    def extrinsics(self) -> np.ndarray:
        """(N_frames, 4, 4) extrinsic matrices for all frames."""
        return np.stack([f.extrinsic for f in self.frames], axis=0)

    @property
    def positions_world(self) -> np.ndarray:
        """(N_frames, 3) camera centres in world space."""
        return np.stack([f.position_world for f in self.frames], axis=0)


# ── builder ───────────────────────────────────────────────────────────────────

class SubmapBuilder:
    """Builds a Submap from a batch of keyframe image paths."""

    def __init__(
        self,
        estimator: DepthEstimator,
        confidence_percentile: float = 40.0,
    ):
        self.estimator = estimator
        self.confidence_percentile = confidence_percentile

    def build(
        self,
        image_paths: list[str],
        seq_indices: list[int],
        submap_idx: int = 0,
    ) -> Submap:
        """
        Args:
            image_paths:  ordered list of keyframe file paths for this submap
            seq_indices:  corresponding indices in the original full sequence
            submap_idx:   position of this submap in the global sequence

        Returns:
            Submap with per-frame points in camera and world coordinates
        """
        assert len(image_paths) == len(seq_indices)

        prediction: DepthPrediction = self.estimator.infer(image_paths)
        submap = Submap(idx=submap_idx)

        for i, seq_idx in enumerate(seq_indices):
            points_cam, mask = prediction.to_pointcloud(i, self.confidence_percentile)
            points_world = _transform_to_world(points_cam, prediction.extrinsics[i])
            colors = _extract_colors(prediction.processed_images[i], mask)

            frame = Frame(
                seq_idx=seq_idx,
                image=prediction.processed_images[i],
                points_cam=points_cam,
                points_world=points_world,
                colors=colors,
                extrinsic=prediction.extrinsics[i],
                intrinsic=prediction.intrinsics[i],
            )
            submap.frames.append(frame)

        return submap


# ── internal helpers ──────────────────────────────────────────────────────────

def _transform_to_world(
    points_cam: np.ndarray, extrinsic: np.ndarray
) -> np.ndarray:
    """
    Transform (M, 3) camera-space points to world space.

    extrinsic is world-to-cam (4x4), so cam-to-world = inv(extrinsic).
    """
    cam_to_world = np.linalg.inv(extrinsic)
    rotation    = cam_to_world[:3, :3]
    translation = cam_to_world[:3, 3]
    return (points_cam @ rotation.T) + translation


def _extract_colors(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract RGB colours at masked pixel positions.

    Args:
        image: (H, W, 3) uint8 RGB
        mask:  (H, W) bool

    Returns:
        (M, 3) uint8
    """
    return image[mask].astype(np.uint8)
