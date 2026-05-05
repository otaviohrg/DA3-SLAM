"""
Pose graph optimization over submaps using GTSAM.

One Pose3 node per submap represents that submap's coordinate frame
in the global world frame. Between-factors encode the relative transforms
computed by SubmapAligner. A prior on the first node fixes the gauge.

GTSAM convention used here:
    Pose3(R, t) = cam-to-world transform
    P_world = R @ P_local + t

Between-factor T_{i,j} satisfies:  X_j = X_i * T_{i,j}
so the measurement is:             T_{i,j} = X_i^{-1} * X_j

For consecutive submaps aligned with T_a_from_b:
    X_0 = Identity
    X_1 = T_0_from_1
    T_{0,1} = X_0^{-1} * X_1 = T_0_from_1
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import gtsam

from da3_slam.submap import Submap
from da3_slam.alignment import AlignmentResult


# ── noise models ──────────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    # Sigmas for the prior on submap 0 — [rot (rad), trans (m)] x3
    # Canonical values: config/default.yaml → noise.*
    prior_rot_sigma: float
    prior_trans_sigma: float

    # Sigmas for between-factors from anchor alignment
    between_rot_sigma: float
    between_trans_sigma: float

    # Sigmas for loop closure between-factors (looser)
    loop_rot_sigma: float
    loop_trans_sigma: float


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    # Optimized (4, 4) world-frame poses for each submap, indexed by submap idx
    poses: dict[int, np.ndarray]

    # Final error after optimization
    final_error: float

    # Number of iterations taken
    iterations: int

    def pose(self, submap_idx: int) -> np.ndarray:
        return self.poses[submap_idx]


# ── graph ─────────────────────────────────────────────────────────────────────

class PoseGraph:
    """
    Builds and optimizes a pose graph over submaps.

    Usage:
        graph = PoseGraph()
        graph.add_submap(submap_0)                          # first submap
        graph.add_submap(submap_1, alignment_01)            # + between-factor
        graph.add_submap(submap_2, alignment_12)
        graph.add_loop_closure(0, 5, loop_alignment)        # optional
        result = graph.optimize()
    """

    def __init__(self, noise: NoiseConfig | None = None):
        self.noise = noise or NoiseConfig()
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial = gtsam.Values()
        self._submap_ids: list[int] = []           # submap.idx in insertion order
        self._current_pose = gtsam.Pose3()         # accumulated global pose
        self._result: gtsam.Values | None = None

    # ── building ──────────────────────────────────────────────────────────────

    def add_submap(
        self,
        submap: Submap,
        alignment: AlignmentResult | None = None,
    ) -> None:
        """
        Add a submap node to the graph.

        Args:
            submap:    the submap to add
            alignment: alignment from the previous submap to this one.
                       Must be None for the first submap.
        """
        key = submap.idx
        n = noise = self.noise

        if not self._submap_ids:
            # First submap — add prior and seed at identity
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([n.prior_rot_sigma] * 3 + [n.prior_trans_sigma] * 3)
            )
            self._graph.add(
                gtsam.PriorFactorPose3(key, gtsam.Pose3(), prior_noise)
            )
            self._current_pose = gtsam.Pose3()
        else:
            assert alignment is not None, \
                "alignment required for all submaps after the first"

            prev_key = self._submap_ids[-1]
            between_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([n.between_rot_sigma] * 3 + [n.between_trans_sigma] * 3)
            )
            T = alignment.T_a_from_b
            relative = gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3])

            self._graph.add(
                gtsam.BetweenFactorPose3(prev_key, key, relative, between_noise)
            )
            # Accumulate initial estimate
            self._current_pose = self._current_pose.compose(relative)

        self._initial.insert(key, self._current_pose)
        self._submap_ids.append(key)

    def add_loop_closure(
        self,
        submap_idx_a: int,
        submap_idx_b: int,
        alignment: AlignmentResult,
    ) -> None:
        """
        Add a loop closure between-factor between two non-consecutive submaps.
        """
        n = self.noise
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([n.loop_rot_sigma] * 3 + [n.loop_trans_sigma] * 3)
        )
        T = alignment.T_a_from_b
        relative = gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3])
        self._graph.add(
            gtsam.BetweenFactorPose3(submap_idx_a, submap_idx_b, relative, loop_noise)
        )

    # ── optimization ──────────────────────────────────────────────────────────

    def optimize(self, verbose: bool = False) -> OptimizationResult:
        """Run Levenberg-Marquardt optimization."""
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")

        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self._graph, self._initial, params
        )
        self._result = optimizer.optimize()

        poses = {
            idx: _pose3_to_matrix(self._result.atPose3(idx))
            for idx in self._submap_ids
        }

        return OptimizationResult(
            poses=poses,
            final_error=self._graph.error(self._result),
            iterations=optimizer.iterations(),
        )

    # ── queries ───────────────────────────────────────────────────────────────

    def initial_error(self) -> float:
        return self._graph.error(self._initial)

    @property
    def n_factors(self) -> int:
        return self._graph.size()

    @property
    def n_nodes(self) -> int:
        return len(self._submap_ids)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pose3_to_matrix(pose: gtsam.Pose3) -> np.ndarray:
    """Convert a gtsam.Pose3 to a (4, 4) float32 numpy matrix."""
    R = pose.rotation().matrix()
    t = pose.translation()
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
