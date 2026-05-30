"""
Submap construction.

A submap is a local map built from a batch of keyframes processed together
by DA3. It stores:
  - Per-frame sparse point clouds (confidence-filtered) in camera and world coords
  - Per-frame raw depth and confidence maps for re-thresholding
  - Per-frame retrieval vectors (DINOv2 descriptors, populated by LoopClosureDetector)
  - Per-frame semantic vectors (CLIP embeddings, populated by SemanticEmbedder)
  - Inverse intrinsics (proj_mat) for reprojection
  - Image names and parsed frame IDs for logging and retrieval
  - Loop closure metadata (is_lc_submap, last_non_loop_frame_index)

The SubmapBuilder orchestrates:
  1. DA3 inference on the keyframe batch
  2. Confidence-filtered point cloud extraction per frame
  3. Camera-to-world transform using DA3's estimated extrinsics
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

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
    conf: np.ndarray

    # (H, W) bool — pixels that survived the confidence filter at build time
    conf_mask: np.ndarray

    # (4, 4) float32 — inverse camera intrinsics K⁻¹ padded to 4×4
    proj_mat: np.ndarray

    # (D,) float32 — DINOv2 retrieval descriptor; set by LoopClosureDetector
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
        stored depth map. Pixels that fail the optional threshold or have
        invalid depth are set to NaN so the spatial layout is preserved.
        """
        K = self.intrinsic
        H, W = self.depth.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

        mask = np.isfinite(self.depth) & (self.depth > 0.0)
        if conf_threshold is not None:
            mask &= (self.conf >= conf_threshold)

        z = np.where(mask, self.depth, np.nan)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.stack([x, y, z], axis=-1).astype(np.float32)


@dataclass
class Submap:
    """Local map built from a batch of keyframes."""

    # Position of this submap in the global sequence
    idx: int

    # Ordered list of frames in this submap
    frames: list[Frame] = field(default_factory=list)

    # Loop closure metadata
    is_lc_submap: bool = False
    last_non_loop_frame_index: int | None = None

    # Original file paths and parsed numeric frame IDs
    img_names: list[str] = field(default_factory=list)
    frame_ids: list[float] = field(default_factory=list)

    # Per-frame CLIP semantic embeddings (set by SemanticEmbedder)
    semantic_vectors: list = field(default_factory=list)

    # Global confidence threshold used at build time; enables re-filtering
    conf_threshold: float | None = None

    def __post_init__(self):
        # Cached Open3D PointCloud for voxelization — not serialised
        self._voxelized_points = None

    # ── basic properties ──────────────────────────────────────────────────────

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

    @property
    def conf(self) -> np.ndarray:
        """(N_frames, H, W) float32 raw confidence maps stacked from all frames."""
        return np.stack([f.conf for f in self.frames], axis=0)

    @property
    def conf_masks(self) -> np.ndarray:
        """(N_frames, H, W) bool build-time confidence masks stacked from all frames."""
        return np.stack([f.conf_mask for f in self.frames], axis=0)

    @property
    def retrieval_vectors(self) -> np.ndarray | None:
        """(N_frames, D) per-frame DINOv2 descriptors, or None if not set."""
        vecs = [f.retrieval_vector for f in self.frames]
        if any(v is None for v in vecs):
            return None
        return np.stack(vecs, axis=0)

    # ── loop closure metadata ─────────────────────────────────────────────────

    def set_lc_status(self, is_lc_submap: bool) -> None:
        self.is_lc_submap = is_lc_submap

    def get_lc_status(self) -> bool:
        return self.is_lc_submap

    def set_last_non_loop_frame_index(self, idx: int) -> None:
        self.last_non_loop_frame_index = idx

    def get_last_non_loop_frame_index(self) -> int | None:
        return self.last_non_loop_frame_index

    # ── image provenance ──────────────────────────────────────────────────────

    def set_img_names(self, img_names: list[str]) -> None:
        self.img_names = list(img_names)

    def get_img_names_at_index(self, index: int) -> str:
        return self.img_names[index]

    def set_frame_ids(self, file_paths: list[str]) -> None:
        """Parse integer/decimal frame numbers from file paths; fall back to index."""
        ids = []
        for i, path in enumerate(file_paths):
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)
            ids.append(float(match.group()) if match else float(i))
        self.frame_ids = ids

    def get_frame_ids(self) -> list[float]:
        return self.frame_ids

    # ── retrieval vectors ─────────────────────────────────────────────────────

    def set_all_retrieval_vectors(self, vectors: list[np.ndarray]) -> None:
        """Set per-frame DINOv2 retrieval descriptors."""
        for frame, vec in zip(self.frames, vectors):
            frame.retrieval_vector = vec

    def get_all_retrieval_vectors(self) -> np.ndarray | None:
        return self.retrieval_vectors

    # ── semantic vectors ──────────────────────────────────────────────────────

    def set_all_semantic_vectors(self, vectors: list[np.ndarray]) -> None:
        """Set per-frame CLIP semantic embeddings."""
        self.semantic_vectors = list(vectors)
        for frame, vec in zip(self.frames, vectors):
            frame.semantic_vector = vec

    def get_all_semantic_vectors(self) -> list:
        return self.semantic_vectors

    # ── confidence utilities ──────────────────────────────────────────────────

    def get_conf_threshold(self) -> float | None:
        return self.conf_threshold

    def get_conf_masks_frame(self, index: int) -> np.ndarray:
        """(H, W) bool confidence mask for the frame at `index`."""
        return self.frames[index].conf_mask

    def filter_data_by_confidence(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the global confidence mask to dense (N, H, W, ...) data.
        Returns the subset of elements where confidence >= conf_threshold.
        """
        if self.conf_threshold is None:
            return data
        mask = self.conf >= self.conf_threshold  # (N, H, W)
        return data[mask]

    # ── world-frame point access ──────────────────────────────────────────────

    def get_points_in_world_frame(self, opt_result) -> np.ndarray:
        """
        All global-world-space points using per-frame optimised poses.

        Each frame's camera-space points are projected to global world via
        the frame's optimised cam-to-world pose: global_pts = c2w @ pts_cam.

        Returns (N_total, 3) float32.
        """
        all_pts: list[np.ndarray] = []
        for frame in self.frames:
            pts = frame.points_cam  # (M, 3) in camera space
            if len(pts) == 0:
                continue
            global_c2w = opt_result.pose(frame.seq_idx).astype(np.float64)
            homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            all_pts.append((global_c2w @ homo.T).T[:, :3].astype(np.float32))
        if not all_pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(all_pts, axis=0)

    def get_points_list_in_world_frame(
        self,
        opt_result,
    ) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
        """
        Per-frame global-world-space points, frame IDs, and confidence masks.

        Returns:
            point_list:      list of (M_i, 3) float32 arrays (one per frame)
            frame_id_list:   list of float frame IDs
            frame_conf_mask: list of (H, W) bool masks
        """
        point_list, frame_id_list, frame_conf_mask = [], [], []
        for i, frame in enumerate(self.frames):
            pts = frame.points_cam  # (M, 3)
            if len(pts) > 0:
                global_c2w = opt_result.pose(frame.seq_idx).astype(np.float64)
                homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
                pts_global = (global_c2w @ homo.T).T[:, :3].astype(np.float32)
            else:
                pts_global = np.empty((0, 3), dtype=np.float32)
            point_list.append(pts_global)
            fid = self.frame_ids[i] if i < len(self.frame_ids) else float(frame.seq_idx)
            frame_id_list.append(fid)
            frame_conf_mask.append(frame.conf_mask)
        return point_list, frame_id_list, frame_conf_mask

    def get_all_poses_world(self, opt_result) -> np.ndarray:
        """(N_frames, 4, 4) optimised cam-to-world poses in global frame."""
        return np.stack(
            [opt_result.pose(f.seq_idx).astype(np.float32) for f in self.frames],
            axis=0,
        )

    def get_first_pose_world(self, opt_result) -> np.ndarray:
        """(4, 4) cam-to-world for the first frame in global space."""
        return opt_result.pose(self.frames[0].seq_idx).astype(np.float32)

    def get_last_pose_world(self, opt_result) -> np.ndarray:
        """(4, 4) cam-to-world for the last non-LC frame in global space."""
        last_idx = (
            self.last_non_loop_frame_index
            if self.last_non_loop_frame_index is not None
            else len(self.frames) - 1
        )
        return opt_result.pose(self.frames[last_idx].seq_idx).astype(np.float32)

    # ── voxelization ──────────────────────────────────────────────────────────

    def get_voxel_points_in_world_frame(
        self,
        opt_result,
        voxel_size: float,
        nb_points: int = 8,
        outlier_radius_factor: float = 2.0,
    ):
        """
        Voxel-downsample and radius-filter the submap's global point cloud.

        Returns an Open3D PointCloud in global world coordinates.
        Global points are computed per-frame using the optimised SL(4) poses,
        then merged before downsampling (no caching — poses change each call).
        """
        import open3d as o3d

        if voxel_size <= 0.0:
            raise ValueError("`voxel_size` must be > 0.0")

        global_pts    = self.get_points_in_world_frame(opt_result)
        global_colors = self.colors.astype(np.float64) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(global_pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(global_colors)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if nb_points > 0:
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=nb_points,
                radius=voxel_size * outlier_radius_factor,
            )
        return pcd

    # ── semantic mask extraction ──────────────────────────────────────────────

    def get_points_in_mask(
        self,
        frame_index: int,
        mask: np.ndarray,
        opt_result,
    ) -> np.ndarray:
        """
        World-space points that fall within a 2D segmentation mask.

        Reconstructs the dense point cloud from the stored depth map for the
        requested frame, then transforms to global world via opt_result.

        Args:
            frame_index: index within this submap's frame list
            mask:        (H, W) bool — pixels to include
            opt_result:  OptimizationResult providing the global Sim3

        Returns:
            (N_mask, 3) float32 global world-space points
        """
        frame = self.frames[frame_index]
        K     = frame.intrinsic
        depth = frame.depth  # (H, W)
        H, W  = depth.shape

        u, v  = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        if not valid.any():
            return np.empty((0, 3), dtype=np.float32)

        z = depth[valid]
        x = (u[valid] - K[0, 2]) * z / K[0, 0]
        y = (v[valid] - K[1, 2]) * z / K[1, 1]
        points_cam = np.stack([x, y, z], axis=-1).astype(np.float32)

        # Camera space → global world via per-frame optimised pose
        global_c2w = opt_result.pose(frame.seq_idx).astype(np.float64)
        homo = np.hstack([points_cam, np.ones((len(points_cam), 1), dtype=np.float32)])
        return (global_c2w @ homo.T).T[:, :3].astype(np.float32)


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

    def build_from_prediction(
        self,
        prediction: DepthPrediction,
        submap_idx: int,
    ) -> Submap:
        """
        Build a Submap directly from an already-computed DepthPrediction.

        Used by LoopClosureDetector to build 2-frame LC submaps from DA3
        re-inference without hitting the filesystem again.
        """
        conf_threshold = float(np.percentile(prediction.confidence, self.confidence_percentile))
        submap = Submap(idx=submap_idx, conf_threshold=conf_threshold)

        for i in range(len(prediction.extrinsics)):
            points_cam, mask = prediction.to_pointcloud(i, self.confidence_percentile)
            points_world = _transform_to_world(points_cam, prediction.extrinsics[i])
            colors = _extract_colors(prediction.processed_images[i], mask)

            K = prediction.intrinsics[i]
            K_inv = np.eye(4, dtype=np.float32)
            K_inv[:3, :3] = np.linalg.inv(K)

            frame = Frame(
                seq_idx=i,
                image=prediction.processed_images[i],
                points_cam=points_cam,
                points_world=points_world,
                colors=colors,
                extrinsic=prediction.extrinsics[i],
                intrinsic=K,
                depth=prediction.depth[i],
                conf=prediction.confidence[i],
                conf_mask=mask,
                proj_mat=K_inv,
            )
            submap.frames.append(frame)

        submap.set_last_non_loop_frame_index(len(submap.frames) - 1)
        return submap

    def build(
        self,
        image_paths: list[str],
        images: list[np.ndarray],
        seq_indices: list[int],
        submap_idx: int = 0,
    ) -> Submap:
        """
        Build a Submap from a batch of keyframes.

        In addition to sparse filtered point clouds, stores per-frame raw depth
        and confidence maps (for re-thresholding), inverse intrinsics, and
        image provenance metadata.

        Args:
            image_paths:  ordered list of keyframe file paths (used for metadata only)
            images:       pre-loaded HxWx3 uint8 RGB arrays, one per keyframe
            seq_indices:  corresponding indices in the original full sequence
            submap_idx:   position of this submap in the global sequence

        Returns:
            Submap with per-frame point clouds, raw maps, and metadata
        """
        assert len(image_paths) == len(seq_indices) == len(images)

        prediction: DepthPrediction = self.estimator.infer(images)

        # Global confidence threshold computed across all frames in this batch
        conf_threshold = float(np.percentile(prediction.confidence, self.confidence_percentile))

        submap = Submap(idx=submap_idx, conf_threshold=conf_threshold)

        for i, seq_idx in enumerate(seq_indices):
            points_cam, mask = prediction.to_pointcloud(i, self.confidence_percentile)
            points_world = _transform_to_world(points_cam, prediction.extrinsics[i])
            colors = _extract_colors(prediction.processed_images[i], mask)

            K = prediction.intrinsics[i]       # (3, 3)
            K_inv = np.eye(4, dtype=np.float32)
            K_inv[:3, :3] = np.linalg.inv(K)  # inverse intrinsics padded to (4, 4)

            frame = Frame(
                seq_idx=seq_idx,
                image=prediction.processed_images[i],
                points_cam=points_cam,
                points_world=points_world,
                colors=colors,
                extrinsic=prediction.extrinsics[i],
                intrinsic=K,
                depth=prediction.depth[i],
                conf=prediction.confidence[i],
                conf_mask=mask,
                proj_mat=K_inv,
            )
            submap.frames.append(frame)

        # Provenance metadata
        submap.set_img_names(image_paths)
        submap.set_frame_ids(image_paths)
        submap.set_last_non_loop_frame_index(len(image_paths) - 1)

        return submap


# ── internal helpers ──────────────────────────────────────────────────────────

def _transform_to_world(
    points_cam: np.ndarray, extrinsic: np.ndarray
) -> np.ndarray:
    cam_to_world = np.linalg.inv(extrinsic)
    return (points_cam @ cam_to_world[:3, :3].T) + cam_to_world[:3, 3]


def _extract_colors(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image[mask].astype(np.uint8)
