"""
Sim3 pose graph optimization over submaps.

One 7-DOF Similarity3 node per submap: rotation (3), translation (3),
log-scale (1). Optimized with scipy Levenberg-Marquardt — GTSAM's Python
bindings expose Similarity3 only as a data class with no factor types.

The scale DOF allows the optimizer to absorb per-submap metric scale drift
that arises when DA3 processes different frame batches independently.
Between-factors from anchor alignment are encoded as Sim3 with scale=1;
loop closure factors carry a scale estimated by Sim3 ICP (Umeyama), providing
the gradient needed to move scales away from 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import least_squares

from da3_slam.backend.inference.submap import Submap
from da3_slam.backend.processing.alignment import AlignmentResult


# ── Sim3 Lie group utilities ──────────────────────────────────────────────────

def _skew(vector: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([[ 0.,           -vector[2],  vector[1]],
                     [ vector[2],    0.,          -vector[0]],
                     [-vector[1],    vector[0],   0.        ]], dtype=np.float64)


def _sim3_exp(tangent_vector: np.ndarray) -> np.ndarray:
    """
    Sim3 exponential map: tangent 7-vector → 4×4 matrix [s·R | t; 0 | 1].

    tangent_vector = [rotation_axis_angle (3), tangent_translation (3), log_scale (1)]
    where scale = exp(log_scale), R = Rodrigues(rotation_axis_angle), t = left_jacobian @ tangent_translation
    """
    rotation_axis_angle = tangent_vector[:3]
    tangent_translation = tangent_vector[3:6]
    log_scale           = float(tangent_vector[6])

    scale           = np.exp(log_scale)
    rotation_angle  = np.linalg.norm(rotation_axis_angle)

    if rotation_angle < 1e-9:
        rotation    = np.eye(3, dtype=np.float64)
        translation = tangent_translation.copy().astype(np.float64)
    else:
        rotation_unit_axis = rotation_axis_angle / rotation_angle
        axis_skew     = _skew(rotation_unit_axis)
        rotation      = (np.eye(3)
                         + np.sin(rotation_angle) * axis_skew
                         + (1. - np.cos(rotation_angle)) * (axis_skew @ axis_skew))
        left_jacobian = (np.eye(3)
                         + (1. - np.cos(rotation_angle)) / rotation_angle * axis_skew
                         + (rotation_angle - np.sin(rotation_angle)) / rotation_angle * (axis_skew @ axis_skew))
        translation   = left_jacobian @ tangent_translation

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = scale * rotation
    result[:3, 3]  = translation
    return result


def _sim3_log(sim3_matrix: np.ndarray) -> np.ndarray:
    """Sim3 logarithm map: 4×4 matrix [s·R | t; 0 | 1] → tangent 7-vector."""
    scale_rotation = sim3_matrix[:3, :3].astype(np.float64)
    translation    = sim3_matrix[:3, 3].astype(np.float64)

    scale     = float(np.cbrt(max(abs(np.linalg.det(scale_rotation)), 1e-12)))
    rotation  = scale_rotation / scale
    log_scale = np.log(max(scale, 1e-12))

    cos_angle      = float(np.clip((np.trace(rotation) - 1.) / 2., -1., 1.))
    rotation_angle = float(np.arccos(cos_angle))

    if rotation_angle < 1e-9:
        rotation_axis_angle   = np.zeros(3)
        left_jacobian_inverse = np.eye(3, dtype=np.float64)
    else:
        # Extract the rotation axis from the skew-symmetric part of R
        rotation_skew         = (rotation - rotation.T) / (2. * np.sin(rotation_angle))
        rotation_axis_angle   = rotation_angle * np.array([
            rotation_skew[2, 1],
            rotation_skew[0, 2],
            rotation_skew[1, 0],
        ])
        # V_inv closes the Sim3 log: tangent_translation = V_inv @ translation
        half_angle            = rotation_angle / 2.
        rotation_unit_axis    = rotation_axis_angle / rotation_angle
        axis_skew             = _skew(rotation_unit_axis)
        left_jacobian_inverse = (np.eye(3)
                                 - half_angle * axis_skew
                                 + (1. - half_angle / np.tan(half_angle)) * (axis_skew @ axis_skew))

    tangent_translation = left_jacobian_inverse @ translation
    return np.array([*rotation_axis_angle, *tangent_translation, log_scale], dtype=np.float64)


def _sim3_inv(sim3_matrix: np.ndarray) -> np.ndarray:
    """Inverse of a 4×4 Sim3 matrix [s·R | t; 0 | 1]."""
    scale_rotation = sim3_matrix[:3, :3].astype(np.float64)
    translation    = sim3_matrix[:3, 3].astype(np.float64)

    scale    = float(np.cbrt(max(abs(np.linalg.det(scale_rotation)), 1e-12)))
    rotation = scale_rotation / scale

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T / scale
    inverse[:3, 3]  = -(rotation.T @ translation) / scale
    return inverse


# ── residual function ─────────────────────────────────────────────────────────

def _compute_residuals(
    parameters: np.ndarray,
    num_nodes: int,
    node_to_index: dict[int, int],
    sequential_factors: list[tuple[int, int, np.ndarray]],
    loop_closure_factors: list[tuple[int, int, np.ndarray]],
    prior_weights: np.ndarray,
    sequential_weights: np.ndarray,
    loop_closure_weights: np.ndarray,
) -> np.ndarray:
    tangent_vectors = [parameters[7 * i: 7 * i + 7] for i in range(num_nodes)]
    pose_matrices   = [_sim3_exp(tangent_vectors[i]) for i in range(num_nodes)]
    residual_blocks = []

    # Prior: anchor the first submap at the global origin (identity Sim3)
    residual_blocks.append(tangent_vectors[0] * prior_weights)

    # Sequential between-factors (consecutive submaps, SE3 measurement, scale=1)
    for idx_a, idx_b, measured_transform in sequential_factors:
        predicted_transform = _sim3_inv(pose_matrices[node_to_index[idx_a]]) @ pose_matrices[node_to_index[idx_b]]
        residual_blocks.append(_sim3_log(_sim3_inv(measured_transform) @ predicted_transform) * sequential_weights)

    # Loop closure between-factors (Sim3 ICP measurement with estimated scale)
    for idx_a, idx_b, measured_transform in loop_closure_factors:
        predicted_transform = _sim3_inv(pose_matrices[node_to_index[idx_a]]) @ pose_matrices[node_to_index[idx_b]]
        residual_blocks.append(_sim3_log(_sim3_inv(measured_transform) @ predicted_transform) * loop_closure_weights)

    return np.concatenate(residual_blocks)


# ── noise models ──────────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    # Canonical values: config/default.yaml → noise.*

    # Prior on submap 0: pins it at the global origin
    prior_rotation_sigma: float     # rad
    prior_translation_sigma: float  # m
    # Note: the prior's scale weight reuses between_scale_sigma so the first
    # submap's scale is held softly rather than pinned hard to 1.

    # Between-factors from anchor-frame alignment (consecutive submaps)
    between_rotation_sigma: float     # rad
    between_translation_sigma: float  # m
    between_scale_sigma: float        # log(scale) — allows per-submap scale to drift

    # Loop closure between-factors (Sim3 ICP provides a scale estimate)
    loop_rotation_sigma: float     # rad
    loop_translation_sigma: float  # m
    loop_scale_sigma: float        # log(scale) — tighter, ICP gives a real estimate


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    # Optimized 4×4 Sim3 matrices [s·R | t; 0 | 1] per submap, keyed by submap idx
    poses: dict[int, np.ndarray]

    # Optimized scale per submap (s = exp(log_scale))
    scales: dict[int, float]

    # Final 0.5·||r||² cost from scipy (not the same units as GTSAM error)
    final_error: float

    # Number of function evaluations (njev is None for scipy LM; nfev is the proxy)
    iterations: int

    def pose(self, submap_idx: int) -> np.ndarray:
        return self.poses[submap_idx]

    def scale(self, submap_idx: int) -> float:
        return self.scales[submap_idx]


# ── graph ─────────────────────────────────────────────────────────────────────

class PoseGraph:
    """
    Sim3 pose graph: one 7-DOF node per submap (rotation, translation, log-scale).

    Usage:
        graph = PoseGraph(noise_config)
        graph.add_submap(submap_0)
        graph.add_submap(submap_1, alignment_01)
        graph.add_loop_closure(0, 5, loop_alignment)
        result = graph.optimize()
    """

    def __init__(self, noise: NoiseConfig | None = None):
        self.noise = noise or NoiseConfig()

        self._submap_indices:          list[int]                        = []
        self._sequential_factors:      list[tuple[int, int, np.ndarray]] = []  # (idx_a, idx_b, sim3_4x4)
        self._loop_closure_factors:    list[tuple[int, int, np.ndarray]] = []  # (idx_a, idx_b, sim3_4x4)
        self._initial_tangent_vectors: dict[int, np.ndarray]             = {}  # seeded from composition
        self._optimized_tangent_vectors: dict[int, np.ndarray] | None   = None  # warm-start cache
        self._accumulated_pose = np.eye(4, dtype=np.float64)  # running pose for warm-start seeding

    # ── building ──────────────────────────────────────────────────────────────

    def add_submap(
        self,
        submap: Submap,
        alignment: AlignmentResult | None = None,
    ) -> None:
        """
        Add a submap node.

        Args:
            submap:    submap to add
            alignment: alignment from the previous submap (None for first)
        """
        node_idx = submap.idx

        if not self._submap_indices:
            self._accumulated_pose = np.eye(4, dtype=np.float64)
        else:
            assert alignment is not None, \
                "alignment required for all submaps after the first"
            se3_transform = alignment.world_b_to_world_a.astype(np.float64)
            # Encode SE3 measurement as Sim3 with scale=1 (scale will be optimized freely)
            sim3_measurement = np.eye(4, dtype=np.float64)
            sim3_measurement[:3, :3] = se3_transform[:3, :3]
            sim3_measurement[:3, 3]  = se3_transform[:3, 3]
            self._sequential_factors.append((self._submap_indices[-1], node_idx, sim3_measurement))
            self._accumulated_pose = self._accumulated_pose @ sim3_measurement

        self._initial_tangent_vectors[node_idx] = _sim3_log(self._accumulated_pose)
        self._submap_indices.append(node_idx)

    def add_loop_closure(
        self,
        submap_idx_a: int,
        submap_idx_b: int,
        alignment: AlignmentResult,
    ) -> None:
        """Add a loop closure factor. alignment.scale carries the Sim3 ICP scale."""
        sim3_measurement = alignment.world_b_to_world_a.astype(np.float64)
        self._loop_closure_factors.append((submap_idx_a, submap_idx_b, sim3_measurement))

    # ── optimization ──────────────────────────────────────────────────────────

    def optimize(self, verbose: bool = False) -> OptimizationResult:
        """Run Levenberg-Marquardt optimization over the Sim3 pose graph."""
        node_indices  = self._submap_indices
        num_nodes     = len(node_indices)
        node_to_index = {node_idx: position for position, node_idx in enumerate(node_indices)}
        noise         = self.noise

        # Build initial parameter vector; reuse last result for warm-starting
        cached_vectors = self._optimized_tangent_vectors or {}
        initial_parameters = np.concatenate([
            cached_vectors.get(node_idx, self._initial_tangent_vectors[node_idx])
            for node_idx in node_indices
        ])

        # Inverse-sigma weights: weighted residual r = error / sigma → minimise ||r||²
        prior_weights = np.array(
            [1. / noise.prior_rotation_sigma]     * 3 +
            [1. / noise.prior_translation_sigma]  * 3 +
            [1. / noise.between_scale_sigma]           # soft scale prior (reuses between sigma)
        )
        sequential_weights = np.array(
            [1. / noise.between_rotation_sigma]    * 3 +
            [1. / noise.between_translation_sigma] * 3 +
            [1. / noise.between_scale_sigma]
        )
        loop_closure_weights = np.array(
            [1. / noise.loop_rotation_sigma]    * 3 +
            [1. / noise.loop_translation_sigma] * 3 +
            [1. / noise.loop_scale_sigma]
        )

        residual_fn = partial(
            _compute_residuals,
            num_nodes=num_nodes,
            node_to_index=node_to_index,
            sequential_factors=self._sequential_factors,
            loop_closure_factors=self._loop_closure_factors,
            prior_weights=prior_weights,
            sequential_weights=sequential_weights,
            loop_closure_weights=loop_closure_weights,
        )

        lm_result = least_squares(
            residual_fn, initial_parameters, method="lm",
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            verbose=2 if verbose else 0,
        )
        optimized_parameters = lm_result.x

        # Cache for warm-starting the next optimize() call
        self._optimized_tangent_vectors = {
            node_indices[i]: optimized_parameters[7 * i: 7 * i + 7]
            for i in range(num_nodes)
        }
        self._accumulated_pose = _sim3_exp(self._optimized_tangent_vectors[node_indices[-1]])

        poses  = {}
        scales = {}
        for i, submap_idx in enumerate(node_indices):
            pose_matrix = _sim3_exp(optimized_parameters[7 * i: 7 * i + 7])
            submap_scale = float(np.cbrt(max(abs(np.linalg.det(pose_matrix[:3, :3])), 1e-12)))
            poses[submap_idx]  = pose_matrix.astype(np.float32)
            scales[submap_idx] = submap_scale

        return OptimizationResult(
            poses=poses,
            scales=scales,
            final_error=float(lm_result.cost),
            iterations=int(lm_result.nfev),
        )

    # ── queries ───────────────────────────────────────────────────────────────

    @property
    def n_factors(self) -> int:
        return 1 + len(self._sequential_factors) + len(self._loop_closure_factors)

    @property
    def n_nodes(self) -> int:
        return len(self._submap_indices)
