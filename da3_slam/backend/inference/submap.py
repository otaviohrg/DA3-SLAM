"""
Submap construction.

A submap is a local map built from a batch of keyframes processed together
by DA3.  All frames in one batch share a single arbitrary "local world"
coordinate system and a single arbitrary metric scale chosen by DA3; the
pose graph (factor_graph.py) is responsible for stitching submaps into a
globally consistent map.

Each Frame stores:
  - A sparse confidence-filtered point cloud in camera and local-world coords
  - The raw depth and confidence maps (so points can be re-thresholded later)
  - DA3's estimated extrinsic (world-to-cam) and intrinsic matrices
  - Optional retrieval/semantic descriptors (set by LoopClosureDetector and
    SemanticEmbedder respectively)

The SubmapBuilder orchestrates:
  1. DA3 inference on the keyframe batch
  2. Confidence-filtered point cloud extraction per frame, using a single
     threshold computed globally across the whole batch
  3. Camera-to-world transform using DA3's estimated extrinsics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from da3_slam.backend.inference.depth_estimator import DepthEstimator, DepthPrediction


# ── geometry helper ───────────────────────────────────────────────────────────

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a (4, 4) homogeneous transform to (M, 3) points.

    Returns (M, 3) float32.  Used to project camera-space points into world
    space throughout the pipeline (submaps, PLY export, debugging scripts).
    """
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return ((points @ rotation.T) + translation).astype(np.float32)


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """Single keyframe within a submap."""

    # Index of this frame within the original image sequence
    seq_idx: int

    # (H, W, 3) uint8 RGB — original image at DA3's processed resolution
    image: np.ndarray

    # (M, 3) float32 — confidence-filtered 3D points in camera space
    points_cam: np.ndarray

    # (M, 3) float32 — confidence-filtered 3D points in local world space
    points_world: np.ndarray

    # (M, 3) uint8 — RGB colours for the filtered points
    colors: np.ndarray

    # (4, 4) float32 — world-to-cam extrinsic matrix
    extrinsic: np.ndarray

    # (3, 3) float32 — camera intrinsic matrix
    intrinsic: np.ndarray

    # (H, W) float32 — raw metric depth in metres (kept for re-thresholding)
    depth: np.ndarray

    # (H, W) float32 — per-pixel confidence in [0, 1] (kept for re-thresholding)
    confidence: np.ndarray

    # (H, W) bool — pixels that survived the confidence filter at build time
    confidence_mask: np.ndarray

    # (D,) float32 — DINO-SALAD retrieval descriptor; set by LoopClosureDetector
    retrieval_vector: np.ndarray | None = None

    # (D,) float32 — CLIP semantic descriptor; set by SemanticEmbedder
    semantic_vector: np.ndarray | None = None

    @property
    def n_points(self) -> int:
        return len(self.points_world)

    @property
    def cam_to_world(self) -> np.ndarray:
        """(4, 4) cam-to-world transform (inverse of extrinsic)."""
        return np.linalg.inv(self.extrinsic)

    @property
    def position_world(self) -> np.ndarray:
        """(3,) camera centre in local world coordinates."""
        return self.cam_to_world[:3, 3]

    def get_dense_pointcloud_cam(self, conf_threshold: float | None = None) -> np.ndarray:
        """
        Reconstruct a dense (H, W, 3) point cloud in camera space from the
        stored depth map.  Pixels that fail the optional threshold or have
        invalid depth are set to NaN so the spatial layout is preserved.
        """
        K = self.intrinsic
        H, W = self.depth.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        mask = np.isfinite(self.depth) & (self.depth > 0.0)
        if conf_threshold is not None:
            mask &= (self.confidence >= conf_threshold)

        z = np.where(mask, self.depth, np.nan)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.stack([x, y, z], axis=-1).astype(np.float32)


@dataclass
class Submap:
    """Local map built from a batch of keyframes."""

    # Position of this submap in the global sequence.  Negative indices mark
    # 2-frame loop-closure submaps created by re-inference (see loop_closure.py).
    idx: int

    # Ordered list of frames in this submap
    frames: list[Frame] = field(default_factory=list)

    # True for 2-frame loop-closure submaps built from DA3 re-inference.
    # LC submaps are excluded from trajectory export and from loop-closure
    # candidate search.
    is_lc_submap: bool = False

    # Original keyframe file paths (provenance / debugging; empty for
    # LC submaps, whose frames come from in-memory images)
    image_paths: list[str] = field(default_factory=list)

    # Global confidence threshold used at build time (absolute value in [0, 1])
    conf_threshold: float | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def points_world(self) -> np.ndarray:
        """All local-world-space points concatenated — (N_total, 3)."""
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
        """(N_frames, 3) camera centres in local world space."""
        return np.stack([f.position_world for f in self.frames], axis=0)

    def set_all_retrieval_vectors(self, vectors: Sequence[np.ndarray]) -> None:
        """Attach per-frame DINO-SALAD retrieval descriptors."""
        for frame, vec in zip(self.frames, vectors):
            frame.retrieval_vector = vec

    def set_all_semantic_vectors(self, vectors: Sequence[np.ndarray]) -> None:
        """Attach per-frame CLIP semantic embeddings."""
        for frame, vec in zip(self.frames, vectors):
            frame.semantic_vector = vec

    def get_points_in_world_frame(self, opt_result) -> np.ndarray:
        """
        All points in the *global* world frame, using per-frame optimised poses.

        Each frame's camera-space points are projected via the frame's
        optimised cam-to-world pose from the given OptimizationResult.

        Returns (N_total, 3) float32.
        """
        all_pts = [
            transform_points(
                frame.points_cam,
                opt_result.pose(frame.seq_idx).astype(np.float64),
            )
            for frame in self.frames
            if len(frame.points_cam) > 0
        ]
        if not all_pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(all_pts, axis=0)


# ── builder ───────────────────────────────────────────────────────────────────

class SubmapBuilder:
    """Builds a Submap from a batch of keyframe images."""

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
        images: list[np.ndarray],
        seq_indices: list[int],
        submap_idx: int = 0,
    ) -> Submap:
        """
        Run DA3 on a batch of keyframes and build a Submap.

        Args:
            image_paths:  ordered list of keyframe file paths (metadata only)
            images:       pre-loaded HxWx3 uint8 RGB arrays, one per keyframe
            seq_indices:  corresponding indices in the original full sequence
            submap_idx:   position of this submap in the global sequence
        """
        assert len(image_paths) == len(seq_indices) == len(images)
        prediction = self.estimator.infer(images)
        submap = self.build_from_prediction(prediction, submap_idx, seq_indices)
        submap.image_paths = list(image_paths)
        return submap

    def build_from_prediction(
        self,
        prediction: DepthPrediction,
        submap_idx: int,
        seq_indices: Sequence[int] | None = None,
    ) -> Submap:
        """
        Build a Submap from an already-computed DepthPrediction.

        Also used by LoopClosureDetector to build 2-frame LC submaps from DA3
        re-inference without re-running the model.

        Args:
            prediction:  normalised DA3 outputs for the batch
            submap_idx:  position of this submap in the global sequence
                         (negative for LC submaps)
            seq_indices: per-frame indices into the original sequence;
                         defaults to 0..N-1 (used for LC submaps, whose
                         frames don't correspond to sequence positions)
        """
        if seq_indices is None:
            seq_indices = range(prediction.n_frames)

        # One global threshold across the whole batch: consistently
        # low-confidence frames contribute fewer points than high-confidence
        # ones (a per-frame threshold would always keep the same fraction).
        conf_threshold = prediction.confidence_threshold(self.confidence_percentile)
        submap = Submap(idx=submap_idx, conf_threshold=conf_threshold)

        for i, seq_idx in enumerate(seq_indices):
            points_cam, mask = prediction.to_pointcloud(i, conf_threshold)
            cam_to_world = np.linalg.inv(prediction.extrinsics[i])
            submap.frames.append(Frame(
                seq_idx=seq_idx,
                image=prediction.processed_images[i],
                points_cam=points_cam,
                points_world=transform_points(points_cam, cam_to_world),
                colors=prediction.processed_images[i][mask].astype(np.uint8),
                extrinsic=prediction.extrinsics[i],
                intrinsic=prediction.intrinsics[i],
                depth=prediction.depth[i],
                confidence=prediction.confidence[i],
                confidence_mask=mask,
            ))

        return submap
