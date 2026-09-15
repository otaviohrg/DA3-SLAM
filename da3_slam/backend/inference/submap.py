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


@dataclass
class Submap:
    """Local map built from a batch of keyframes."""

    # Position of this submap in the global sequence.  Negative indices mark
    # loop-closure submaps created by re-inference (see loop_closure.py).
    idx: int

    # Ordered list of frames in this submap
    frames: list[Frame] = field(default_factory=list)

    # True for loop-closure submaps built from DA3 re-inference (matched
    # frame pair + optional context neighbours).  Loop-closure submaps are
    # excluded from trajectory export and from loop-closure candidate search.
    is_loop_closure_submap: bool = False

    # Original keyframe file paths (provenance / debugging; empty for
    # loop-closure submaps, whose frames come from in-memory images)
    image_paths: list[str] = field(default_factory=list)

    # Global confidence threshold used at build time (absolute value in [0, 1])
    confidence_threshold: float | None = None

    # Running product of boundary scale ratios up to this submap (set by
    # _processing): converts this batch's DA3 unit to submap 0's.  Graph
    # translations are multiplied by it, so camera-space points must be too,
    # or the submap renders at the wrong size relative to its own cameras
    # (seams / duplicated geometry between neighbouring submaps).
    global_scale: float = 1.0

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
        for frame, vector in zip(self.frames, vectors):
            frame.retrieval_vector = vector

    def set_all_semantic_vectors(self, vectors: Sequence[np.ndarray]) -> None:
        """Attach per-frame CLIP semantic embeddings."""
        for frame, vector in zip(self.frames, vectors):
            frame.semantic_vector = vector


# ── builder ───────────────────────────────────────────────────────────────────

class SubmapBuilder:
    """Builds a Submap from a batch of keyframe images."""

    def __init__(
        self,
        estimator: DepthEstimator,
        confidence_percentile: float = 40.0,
        build_pointclouds: bool = True,
    ):
        """
        Args:
            estimator:             DA3 wrapper used for inference
            confidence_percentile: global percentile for point filtering
            build_pointclouds:     False = lean mode for runs where nothing
                                   consumes point clouds (no map.ply, no live
                                   viewer — e.g. benchmark sweeps): skips point
                                   extraction and stores empty points/colors/
                                   confidence/mask on every Frame.  Keeps
                                   image (loop-closure re-inference +
                                   descriptors), depth (scale estimation) and
                                   the camera matrices.
                                   Cuts per-frame memory ~3x, which matters
                                   because every keyframe of every submap stays
                                   resident for the whole run.
        """
        self.estimator = estimator
        self.confidence_percentile = confidence_percentile
        self.build_pointclouds = build_pointclouds

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

        Also used by LoopClosureDetector to build loop-closure submaps
        from DA3 re-inference without re-running the model.

        Args:
            prediction:  normalised DA3 outputs for the batch
            submap_idx:  position of this submap in the global sequence
                         (negative for loop-closure submaps)
            seq_indices: per-frame indices into the original sequence;
                         defaults to 0..N-1 (used for loop-closure submaps,
                         whose frames don't correspond to sequence positions)
        """
        if seq_indices is None:
            seq_indices = range(prediction.n_frames)

        # One global threshold across the whole batch: consistently
        # low-confidence frames contribute fewer points than high-confidence
        # ones (a per-frame threshold would always keep the same fraction).
        confidence_threshold = prediction.confidence_threshold(self.confidence_percentile)
        submap = Submap(idx=submap_idx, confidence_threshold=confidence_threshold)

        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        empty_mask = np.empty((0, 0), dtype=bool)
        empty_confidence = np.empty((0, 0), dtype=np.float32)

        for i, seq_idx in enumerate(seq_indices):
            if self.build_pointclouds:
                points_cam, mask = prediction.to_pointcloud(i, confidence_threshold)
                points_world = transform_points(
                    points_cam, np.linalg.inv(prediction.extrinsics[i]))
                colors = prediction.processed_images[i][mask].astype(np.uint8)
                confidence = prediction.confidence[i]
            else:
                # Lean mode: nothing downstream consumes points/colors/
                # confidence — store empties (see __init__ docstring).
                points_cam = points_world = empty_points
                colors, mask, confidence = empty_colors, empty_mask, empty_confidence
            submap.frames.append(Frame(
                seq_idx=seq_idx,
                image=prediction.processed_images[i],
                points_cam=points_cam,
                points_world=points_world,
                colors=colors,
                extrinsic=prediction.extrinsics[i],
                intrinsic=prediction.intrinsics[i],
                depth=prediction.depth[i],
                confidence=confidence,
                confidence_mask=mask,
            ))

        return submap
