"""
Loop closure detection and transform estimation.

Detection:
    DINOv2 (ViT-B/14) CLS token as a global descriptor per submap.
    Cosine similarity between descriptors identifies revisited places.

Transform estimation:
    Point-to-point ICP between the two submaps' world-frame point clouds,
    using the graph's accumulated pose as the initial alignment guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from da3_slam.backend.inference.submap import Submap
from da3_slam.backend.processing.alignment import AlignmentResult


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class LoopClosureConfig:
    # Canonical values: config/default.yaml → loop_closure.*

    # Minimum cosine similarity to flag a candidate
    similarity_threshold: float

    # Submaps must be this far apart in the sequence to be a loop closure
    min_submaps_apart: int

    # DINOv2 model variant
    dinov2_model: str

    # ICP: max number of iterations
    icp_max_iterations: int

    # ICP: convergence tolerance
    icp_tolerance: float

    # ICP: max correspondence distance (metres)
    icp_max_distance: float

    # Number of points to subsample per submap for ICP
    icp_num_points: int

    # ICP RMSE above which a loop closure is rejected (metres)
    icp_max_rmse: float

    # Maximum |log(scale)| deviation from 1.0 before the estimated Sim3 scale
    # is considered unreliable and the loop closure is rejected.
    # log(1.20) ≈ 0.18 → rejects scale changes larger than ~20%.
    icp_max_scale_deviation: float

    # Number of evenly-spaced frames to average for the submap descriptor.
    # More frames → more robust but slower. 1 = middle frame only (old behaviour).
    n_descriptor_frames: int


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class LoopCandidate:
    """A potential loop closure detected by descriptor matching."""
    submap_idx_a: int
    submap_idx_b: int
    similarity: float


@dataclass
class LoopClosure:
    """A verified loop closure with an estimated relative transform."""
    candidate: LoopCandidate
    alignment: AlignmentResult
    icp_rmse: float  # point cloud fit quality (lower = better)

    @property
    def submap_idx_a(self) -> int:
        return self.candidate.submap_idx_a

    @property
    def submap_idx_b(self) -> int:
        return self.candidate.submap_idx_b


# ── detector ──────────────────────────────────────────────────────────────────

class LoopClosureDetector:
    """
    Detects loop closures and estimates relative transforms.

    Usage:
        detector = LoopClosureDetector()
        for submap in submaps:
            closures = detector.process(submap, graph_pose)
            for closure in closures:
                pose_graph.add_loop_closure(closure.submap_idx_a,
                                            closure.submap_idx_b,
                                            closure.alignment)
    """

    def __init__(
        self,
        config: LoopClosureConfig | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or LoopClosureConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._descriptors: dict[int, np.ndarray] = {}
        self._submaps: dict[int, Submap] = {}
        self._poses: dict[int, np.ndarray] = {}      # raw accumulated poses
        self._opt_poses: dict[int, np.ndarray] = {}  # latest optimized poses

        print(f"[LoopClosure] Loading {self.config.dinov2_model}...")
        self._model: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2",
            self.config.dinov2_model,
            verbose=False,
        ).to(self.device).eval()
        print("[LoopClosure] Ready.")

    def update_optimized_poses(self, poses: dict[int, np.ndarray]) -> None:
        """
        Update the stored optimized poses for all known submaps.
        Called after each incremental optimization so ICP uses the best
        available initial transform rather than the raw accumulated pose.
        """
        self._opt_poses.update(poses)

    @torch.no_grad()
    def process(
        self,
        submap: Submap,
        graph_pose: np.ndarray,
    ) -> list[LoopClosure]:
        """
        Add a submap and return verified loop closures.

        Args:
            submap:     newly built submap
            graph_pose: (4, 4) initial pose estimate for this submap
                        in the global frame (from accumulated alignments)

        Returns:
            list of LoopClosure objects ready to add to the factor graph
        """
        self._submaps[submap.idx] = submap
        self._poses[submap.idx] = graph_pose
        self._descriptors[submap.idx] = self._extract_descriptor(submap)

        candidates = self._find_candidates(submap.idx)
        closures = []
        for candidate in candidates:
            closure = self._verify(candidate)
            if closure is not None:
                closures.append(closure)

        return closures

    # ── descriptor extraction ─────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_descriptor(self, submap: Submap) -> np.ndarray:
        """
        Extract a global L2-normalised descriptor by averaging DINOv2 CLS
        tokens from n_descriptor_frames evenly-spaced frames in the submap.
        Averaging suppresses per-frame noise and gives a view that is more
        representative of the submap's spatial extent.
        """
        import torch.nn.functional as F

        n_frames = len(submap.frames)
        n_sample = min(self.config.n_descriptor_frames, n_frames)
        if n_sample <= 1:
            frame_indices = [n_frames // 2]
        else:
            frame_indices = [
                int(round(i * (n_frames - 1) / (n_sample - 1)))
                for i in range(n_sample)
            ]

        mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        descriptors = []
        for frame_idx in frame_indices:
            img = submap.frames[frame_idx].image.astype(np.float32) / 255.0
            img = (img - mean_np) / std_np
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            _, _, H, W = tensor.shape
            H14 = (H // 14) * 14
            W14 = (W // 14) * 14
            if H14 != H or W14 != W:
                tensor = F.interpolate(tensor, size=(H14, W14), mode="bilinear",
                                       align_corners=False)
            feat = self._model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            descriptors.append(feat)

        descriptor = np.mean(descriptors, axis=0)
        descriptor /= np.linalg.norm(descriptor) + 1e-8
        return descriptor

    # ── candidate detection ───────────────────────────────────────────────────

    def _find_candidates(self, query_idx: int) -> list[LoopCandidate]:
        config = self.config
        query_descriptor = self._descriptors[query_idx]
        candidates = []

        for candidate_idx, stored_descriptor in self._descriptors.items():
            if abs(candidate_idx - query_idx) <= config.min_submaps_apart:
                continue
            similarity = float(np.dot(query_descriptor, stored_descriptor))
            if similarity >= config.similarity_threshold:
                candidates.append(LoopCandidate(
                    submap_idx_a=candidate_idx,
                    submap_idx_b=query_idx,
                    similarity=similarity,
                ))

        return sorted(candidates, key=lambda c: -c.similarity)

    # ── transform verification via ICP ────────────────────────────────────────

    def _verify(self, candidate: LoopCandidate) -> LoopClosure | None:
        """
        Estimate relative transform between two submaps using Sim3 ICP.

        Rejection gates (in order):
          1. ICP fails to find enough correspondences → None
          2. RMSE > icp_max_rmse → poor geometric fit
          3. |log(scale)| > icp_max_scale_deviation → unreliable scale estimate
        """
        submap_idx_a = candidate.submap_idx_a
        submap_idx_b = candidate.submap_idx_b
        submap_a = self._submaps[submap_idx_a]
        submap_b = self._submaps[submap_idx_b]

        # Use optimized pose if available (better initial alignment); else raw.
        global_pose_a = self._opt_poses.get(submap_idx_a, self._poses[submap_idx_a])
        global_pose_b = self._opt_poses.get(submap_idx_b, self._poses[submap_idx_b])
        initial_world_b_to_world_a = np.linalg.inv(global_pose_a) @ global_pose_b

        destination_points = _subsample(submap_a.points_world, self.config.icp_num_points)
        source_points      = _subsample(submap_b.points_world, self.config.icp_num_points)

        icp_world_b_to_world_a, rmse = _icp(source_points, destination_points,
                                             initial_world_b_to_world_a, self.config)

        tag = f"[LoopClosure] {submap_idx_a}↔{submap_idx_b}  sim={candidate.similarity:.3f}"

        if rmse is None:
            print(f"{tag}  REJECTED (no correspondences)")
            return None

        if rmse > self.config.icp_max_rmse:
            print(f"{tag}  icp_rmse={rmse:.4f}m  REJECTED (rmse > {self.config.icp_max_rmse}m)")
            return None

        icp_scale = float(np.cbrt(max(abs(np.linalg.det(icp_world_b_to_world_a[:3, :3])), 1e-12)))
        log_scale_deviation = abs(float(np.log(max(icp_scale, 1e-6))))
        if log_scale_deviation > self.config.icp_max_scale_deviation:
            print(f"{tag}  icp_rmse={rmse:.4f}m  scale={icp_scale:.3f}"
                  f"  REJECTED (|log(scale)|={log_scale_deviation:.3f} > {self.config.icp_max_scale_deviation})")
            return None

        print(f"{tag}  icp_rmse={rmse:.4f}m  scale={icp_scale:.4f}  ACCEPTED")
        alignment = AlignmentResult(
            world_b_to_world_a=icp_world_b_to_world_a.astype(np.float32),
            method="icp",
            scale=icp_scale,
        )
        return LoopClosure(candidate=candidate, alignment=alignment, icp_rmse=rmse)


# ── ICP implementation ────────────────────────────────────────────────────────

def _subsample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    selected_indices = np.random.choice(len(points), n, replace=False)
    return points[selected_indices]


def _icp(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    initial_transform: np.ndarray,
    config: LoopClosureConfig,
) -> tuple[np.ndarray, float | None]:
    """
    Point-to-point ICP.

    Args:
        source_points:      (M, 3) source points (submap B world-frame)
        destination_points: (N, 3) destination points (submap A world-frame)
        initial_transform:  (4, 4) initial transform (source → destination)
        config:             ICP hyperparameters

    Returns:
        (refined_transform, rmse) or (initial_transform, None) on failure
    """
    from scipy.spatial import KDTree

    transform = initial_transform.copy().astype(np.float64)
    destination_tree = KDTree(destination_points)
    prev_rmse = np.inf

    for _ in range(config.icp_max_iterations):
        scale_rotation, translation = transform[:3, :3], transform[:3, 3]
        source_transformed = (source_points @ scale_rotation.T) + translation

        distances, closest_indices = destination_tree.query(source_transformed, workers=-1)
        inlier_mask = distances < config.icp_max_distance
        if inlier_mask.sum() < 10:
            return initial_transform, None

        inlier_source      = source_transformed[inlier_mask]
        inlier_destination = destination_points[closest_indices[inlier_mask]]
        rmse = float(np.sqrt((distances[inlier_mask] ** 2).mean()))

        if abs(prev_rmse - rmse) < config.icp_tolerance:
            break
        prev_rmse = rmse

        # Estimate incremental Sim3 transform (Umeyama)
        incremental_scale, incremental_rotation, incremental_translation = \
            _umeyama_sim3(inlier_source, inlier_destination)
        incremental_transform = np.eye(4, dtype=np.float64)
        incremental_transform[:3, :3] = incremental_scale * incremental_rotation
        incremental_transform[:3, 3]  = incremental_translation
        transform = incremental_transform @ transform

    return transform.astype(np.float32), rmse


def _umeyama_sim3(
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Sim3 alignment via Umeyama (1991).
    Finds scale, rotation, translation minimising
    sum ||scale·rotation·source_i + translation − destination_i||².
    Returns (scale, rotation_3x3, translation_3).
    """
    num_points          = len(source_points)
    source_centroid     = source_points.mean(0)
    destination_centroid = destination_points.mean(0)
    centered_source      = source_points      - source_centroid
    centered_destination = destination_points - destination_centroid

    source_variance       = float(np.trace(centered_source.T @ centered_source) / num_points)
    cross_covariance      = (centered_destination.T @ centered_source) / num_points
    U, singular_values, Vt = np.linalg.svd(cross_covariance)

    determinant_product       = np.linalg.det(U) * np.linalg.det(Vt)
    reflection_correction_matrix = np.diag([1., 1., float(np.sign(determinant_product))])

    rotation    = U @ reflection_correction_matrix @ Vt
    scale       = float(max(
        np.trace(np.diag(singular_values) @ reflection_correction_matrix) / max(source_variance, 1e-12),
        1e-6,
    ))
    translation = destination_centroid - scale * rotation @ source_centroid
    return scale, rotation, translation
