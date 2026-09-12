"""
Per-frame SL(4) pose graph using GTSAM — identical architecture to VGGT-SLAM.

One SL(4) node per keyframe (identified by seq_idx). Factor types:
  - Prior:        anchors frame 0 at identity (very tight, 15-dim isotropic)
  - Between:      consecutive frames within a submap (inner-submap)
  - Loop closure: frame pair from DA3 re-inference

The shared anchor frame (last of submap N = first of submap N+1) implicitly
bridges consecutive submaps — no explicit inter-submap factor is needed.
Scale differences between submap batches are resolved by scaling the
translation component of between-factor measurements before insertion,
using a depth-ratio estimate at each submap boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gtsam import (
    NonlinearFactorGraph,
    Values,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    noiseModel,
    SL4,
    PriorFactorSL4,
    BetweenFactorSL4,
    Similarity3,
    PriorFactorSimilarity3,
    BetweenFactorSimilarity3,
    Rot3,
    Point3,
)
from gtsam.symbol_shorthand import X

# Re-exported so callers can import the config next to the component it tunes.
from da3_slam.config import NoiseConfig


# ── pose parameterisation ─────────────────────────────────────────────────────
#
# SL(4) is inherited from VGGT-SLAM, where it is justified by the PROJECTIVE
# ambiguity of uncalibrated monocular reconstruction: submaps there are genuinely
# related by homographies.  DA3 is a different case — it predicts metric depth
# AND intrinsics per frame, which collapses most of that ambiguity before the
# graph sees it.  Measured inter-submap disagreement is 0.245 deg of rotation
# and 4.8% of scale, i.e. Sim(3) (7 DOF), not the 15 DOF of SL(4).
#
# Sim(3) convention (verified against GTSAM, do not assume): a Similarity3 acts
# as transformFrom(p) = s * (R p + t), and `matrix()` returns [R, t; 0, 1/s].
# So the camera CENTRE in world is scale() * translation() — reading
# translation() alone silently discards the scale correction, which is the
# entire reason for using Sim(3).

class _Sl4Ops:
    """15-DOF projective parameterisation (the shipped default)."""

    name, dim = "sl4", 15
    PriorFactor, BetweenFactor = PriorFactorSL4, BetweenFactorSL4

    @staticmethod
    def make(matrix: np.ndarray):
        return SL4(matrix.astype(np.float64))

    @staticmethod
    def at(values, key):
        return values.atSL4(key)

    @staticmethod
    def to_matrix(element) -> np.ndarray:
        return element.matrix()


class _Sim3Ops:
    """7-DOF similarity parameterisation (rigid + uniform scale)."""

    name, dim = "sim3", 7
    PriorFactor, BetweenFactor = PriorFactorSimilarity3, BetweenFactorSimilarity3

    @staticmethod
    def make(matrix: np.ndarray):
        matrix = matrix.astype(np.float64)
        # Incoming measurements are rigid, so scale enters as 1.0 and the
        # optimiser is free to move it only where constraints conflict.
        return Similarity3(Rot3(matrix[:3, :3]), Point3(*matrix[:3, 3]), 1.0)

    @staticmethod
    def at(values, key):
        return values.atSimilarity3(key)

    @staticmethod
    def to_matrix(element) -> np.ndarray:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = element.rotation().matrix()
        # scale() * translation(), per the convention noted above.
        out[:3, 3] = element.scale() * np.asarray(element.translation())
        return out


PARAMETERISATIONS = {"sl4": _Sl4Ops, "sim3": _Sim3Ops}

__all__ = ["NoiseConfig", "OptimizationResult", "PoseGraph"]


def _isotropic_noise(sigma: float, huber_k: float | None, dim: int = 15):
    """Isotropic noise model of the given dimension, optionally Huber-robustified.

    With a Huber kernel an outlier measurement is down-weighted instead of
    warping the whole map; huber_k=None keeps plain Gaussian noise.

    NOTE the isotropy is a modelling compromise, not a principled choice: the
    dimensions are not commensurable (rotation in radians, translation in
    metres, and for SL(4) the shear/projective components are dimensionless).
    One sigma across all of them is dimensionally incoherent; it is kept
    because it mirrors VGGT-SLAM and because a chain-only graph is exactly
    determined, where the noise model has no effect at all.
    """
    noise = noiseModel.Diagonal.Sigmas(np.full(dim, sigma))
    if huber_k:
        noise = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(float(huber_k)), noise)
    return noise


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    # Optimized (4, 4) cam-to-world matrices per keyframe, keyed by seq_idx
    frame_poses: dict[int, np.ndarray]

    # Final 0.5·||r||² graph error from GTSAM
    final_error: float

    # Number of LM iterations
    iterations: int

    def pose(self, seq_idx: int) -> np.ndarray:
        """(4, 4) cam-to-world for the frame with the given seq_idx."""
        return self.frame_poses[seq_idx]


# ── graph ─────────────────────────────────────────────────────────────────────

class PoseGraph:
    """
    SL(4) pose graph: one node per keyframe (seq_idx), optimised with GTSAM LM.

    Usage:
        graph = PoseGraph(noise_config)

        # First submap
        for frame in submap_0.frames:
            graph.add_frame(frame.seq_idx, frame.cam_to_world)
        graph.add_prior(submap_0.frames[0].seq_idx)
        for i in range(1, len(submap_0.frames)):
            graph.add_between(frames[i-1].seq_idx, frames[i].seq_idx, relative)

        # Subsequent submaps — anchor frame already in graph
        for frame in submap_1.frames[1:]:
            graph.add_frame(frame.seq_idx, global_c2w)
        for i in range(1, len(submap_1.frames)):
            graph.add_between(...)

        # Loop closure
        graph.add_between(frame_b.seq_idx, frame_a.seq_idx, relative_lc, loop=True)

        result = graph.optimize()
    """

    def __init__(self, noise: NoiseConfig | None = None,
                 parameterisation: str = "sl4"):
        self.noise = noise or NoiseConfig(prior_sigma=1e-6, between_sigma=0.05, loop_sigma=0.05)
        try:
            self._ops = PARAMETERISATIONS[str(parameterisation).lower()]
        except KeyError:
            raise ValueError(
                f"unknown pose parameterisation {parameterisation!r}; "
                f"expected one of {sorted(PARAMETERISATIONS)}") from None

        self._graph  = NonlinearFactorGraph()
        self._values = Values()
        self._initialized: set[int] = set()   # seq_idx values currently in graph

        self._prior_noise = _isotropic_noise(self.noise.prior_sigma, None,
                                             self._ops.dim)
        # Huber on odometry factors acts only where redundancy exists
        # (overlap>=2 duplicate boundary factors, loop-closure cycles) — a
        # broken boundary measurement then absorbs its own error instead of
        # deforming the whole cycle into offset ghost copies.  Huber on loop
        # factors cushions an aliased closure that survived the gates.
        self._between_noise = _isotropic_noise(
            self.noise.between_sigma, self.noise.between_huber_k, self._ops.dim)
        self._loop_noise = _isotropic_noise(
            self.noise.loop_sigma, self.noise.loop_huber_k, self._ops.dim)

    # ── building ──────────────────────────────────────────────────────────────

    def add_frame(self, seq_idx: int, cam_to_world: np.ndarray) -> None:
        """Insert a new frame node. Silently skips if seq_idx is already in the graph."""
        if seq_idx in self._initialized:
            return
        self._values.insert(X(seq_idx), self._ops.make(cam_to_world))
        self._initialized.add(seq_idx)

    def add_prior(self, seq_idx: int) -> None:
        """Add a prior factor anchoring the given frame at its current value."""
        pose = self._ops.at(self._values, X(seq_idx))
        self._graph.add(self._ops.PriorFactor(X(seq_idx), pose, self._prior_noise))

    def add_between(
        self,
        seq_idx_a: int,
        seq_idx_b: int,
        relative_cam_to_world: np.ndarray,
        loop: bool = False,
    ) -> None:
        """
        Add a between factor encoding: node_a.inverse() ⊗ node_b ≈ relative_cam_to_world.

        For nodes representing cam-to-world, relative_cam_to_world = w2c_a @ c2w_b.

        Args:
            seq_idx_a:    source frame
            seq_idx_b:    target frame
            relative_cam_to_world: (4, 4) measured relative transform in global scale
            loop:         True → use loop closure noise model (looser)
        """
        noise = self._loop_noise if loop else self._between_noise
        self._graph.add(
            self._ops.BetweenFactor(
                X(seq_idx_a), X(seq_idx_b),
                self._ops.make(relative_cam_to_world),
                noise,
            )
        )

    def get_pose(self, seq_idx: int) -> np.ndarray:
        """(4, 4) current cam-to-world estimate for the given frame."""
        return self._ops.to_matrix(self._ops.at(self._values, X(seq_idx)))

    # ── optimization ──────────────────────────────────────────────────────────

    def optimize(self, verbose: bool = False) -> OptimizationResult:
        """Run Levenberg-Marquardt optimisation and update internal values."""
        params = LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        initial_error = self._graph.error(self._values)
        if verbose:
            print(f"[PoseGraph] Initial error: {initial_error:.6f}")

        optimizer = LevenbergMarquardtOptimizer(self._graph, self._values, params)
        result    = optimizer.optimize()

        final_error = self._graph.error(result)
        iterations  = optimizer.iterations()

        # Update stored values for warm-starting the next call
        self._values = result

        frame_poses: dict[int, np.ndarray] = {}
        for seq_idx in self._initialized:
            frame_poses[seq_idx] = self._ops.to_matrix(
                self._ops.at(result, X(seq_idx))).astype(np.float32)

        return OptimizationResult(
            frame_poses=frame_poses,
            final_error=float(final_error),
            iterations=int(iterations),
        )

    # ── queries ───────────────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return len(self._initialized)

    @property
    def n_factors(self) -> int:
        return self._graph.size()
