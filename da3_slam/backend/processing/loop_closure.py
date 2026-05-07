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

from dataclasses import dataclass, field

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
        self._poses: dict[int, np.ndarray] = {}

        print(f"[LoopClosure] Loading {self.config.dinov2_model}...")
        self._model: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2",
            self.config.dinov2_model,
            verbose=False,
        ).to(self.device).eval()
        print("[LoopClosure] Ready.")

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
        Extract a global L2-normalised descriptor from the submap's middle frame.
        Uses DINOv2 CLS token — 768-dim for ViT-B/14.
        """
        mid = len(submap.frames) // 2
        img = submap.frames[mid].image.astype(np.float32) / 255.0  # (H, W, 3)

        # DINOv2 expects (B, 3, H, W), normalised with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Resize to multiple of patch size (14)
        _, _, H, W = tensor.shape
        H14 = (H // 14) * 14
        W14 = (W // 14) * 14
        if H14 != H or W14 != W:
            import torch.nn.functional as F
            tensor = F.interpolate(tensor, size=(H14, W14), mode="bilinear",
                                   align_corners=False)

        features = self._model(tensor)       # (1, D)
        descriptor = features.squeeze(0).cpu().numpy().astype(np.float32)
        descriptor /= np.linalg.norm(descriptor) + 1e-8
        return descriptor

    # ── candidate detection ───────────────────────────────────────────────────

    def _find_candidates(self, query_idx: int) -> list[LoopCandidate]:
        config = self.config
        query_descriptor = self._descriptors[query_idx]
        candidates = []

        for idx, descriptor in self._descriptors.items():
            if abs(idx - query_idx) <= config.min_submaps_apart:
                continue
            similarity = float(np.dot(query_descriptor, descriptor))
            if similarity >= config.similarity_threshold:
                candidates.append(LoopCandidate(
                    submap_idx_a=idx,
                    submap_idx_b=query_idx,
                    similarity=similarity,
                ))

        return sorted(candidates, key=lambda c: -c.similarity)

    # ── transform verification via ICP ────────────────────────────────────────

    def _verify(self, candidate: LoopCandidate) -> LoopClosure | None:
        """
        Estimate relative transform between two submaps using ICP.
        Returns None if ICP fails to converge.
        """
        submap_a = self._submaps[candidate.submap_idx_a]
        submap_b = self._submaps[candidate.submap_idx_b]
        pose_a = self._poses[candidate.submap_idx_a]
        pose_b = self._poses[candidate.submap_idx_b]

        # Initial guess: relative transform from accumulated poses
        # T_a_from_b_initial = inv(pose_a) @ pose_b
        initial_transform = np.linalg.inv(pose_a) @ pose_b

        points_a = _subsample(submap_a.points_world, self.config.icp_num_points)
        points_b = _subsample(submap_b.points_world, self.config.icp_num_points)

        refined_transform, rmse = _icp(points_b, points_a, initial_transform, self.config)

        if rmse is None:
            return None

        print(f"[LoopClosure] {candidate.submap_idx_a}↔{candidate.submap_idx_b}  "
              f"sim={candidate.similarity:.3f}  icp_rmse={rmse:.4f}m")

        alignment = AlignmentResult(
            T_a_from_b=refined_transform.astype(np.float32),
            method="icp",
        )
        return LoopClosure(candidate=candidate, alignment=alignment, icp_rmse=rmse)


# ── ICP implementation ────────────────────────────────────────────────────────

def _subsample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.random.choice(len(points), n, replace=False)
    return points[idx]


def _icp(
    src: np.ndarray,
    dst: np.ndarray,
    initial_transform: np.ndarray,
    config: LoopClosureConfig,
) -> tuple[np.ndarray, float | None]:
    """
    Point-to-point ICP.

    Args:
        src:               (M, 3) source points (submap B world-frame)
        dst:               (N, 3) destination points (submap A world-frame)
        initial_transform: (4, 4) initial transform (src → dst)
        config:            ICP hyperparameters

    Returns:
        (refined_transform, rmse) or (initial_transform, None) on failure
    """
    from scipy.spatial import KDTree

    transform = initial_transform.copy().astype(np.float64)
    dst_tree = KDTree(dst)
    prev_rmse = np.inf

    for _ in range(config.icp_max_iterations):
        # Transform src
        R, t = transform[:3, :3], transform[:3, 3]
        src_transformed = (src @ R.T) + t

        # Find correspondences
        dists, idx = dst_tree.query(src_transformed, workers=-1)
        valid = dists < config.icp_max_distance
        if valid.sum() < 10:
            return initial_transform, None

        src_valid = src_transformed[valid]
        dst_valid = dst[idx[valid]]
        rmse = float(np.sqrt((dists[valid] ** 2).mean()))

        if abs(prev_rmse - rmse) < config.icp_tolerance:
            break
        prev_rmse = rmse

        # Estimate incremental transform (SVD)
        delta_transform = _estimate_rigid(src_valid, dst_valid)
        transform = delta_transform @ transform

    return transform.astype(np.float32), rmse


def _estimate_rigid(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate rigid transform from matched point pairs using SVD."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    A = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    t = mu_d - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
