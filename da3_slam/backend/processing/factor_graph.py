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
)
from gtsam.symbol_shorthand import X

# Re-exported so callers can import the config next to the component it tunes.
from da3_slam.config import NoiseConfig

__all__ = ["NoiseConfig", "OptimizationResult", "PoseGraph"]


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

    def __init__(self, noise: NoiseConfig | None = None):
        self.noise = noise or NoiseConfig(prior_sigma=1e-6, between_sigma=0.05, loop_sigma=0.05)

        self._graph  = NonlinearFactorGraph()
        self._values = Values()
        self._initialized: set[int] = set()   # seq_idx values currently in graph

        self._prior_noise   = noiseModel.Diagonal.Sigmas(np.full(15, self.noise.prior_sigma))
        self._between_noise = noiseModel.Diagonal.Sigmas(np.full(15, self.noise.between_sigma))
        self._loop_noise    = noiseModel.Diagonal.Sigmas(np.full(15, self.noise.loop_sigma))

    # ── building ──────────────────────────────────────────────────────────────

    def add_frame(self, seq_idx: int, cam_to_world: np.ndarray) -> None:
        """Insert a new frame node. Silently skips if seq_idx is already in the graph."""
        if seq_idx in self._initialized:
            return
        self._values.insert(X(seq_idx), SL4(cam_to_world.astype(np.float64)))
        self._initialized.add(seq_idx)

    def add_prior(self, seq_idx: int) -> None:
        """Add a prior factor anchoring the given frame at its current value."""
        pose = self._values.atSL4(X(seq_idx))
        self._graph.add(PriorFactorSL4(X(seq_idx), pose, self._prior_noise))

    def add_between(
        self,
        seq_idx_a: int,
        seq_idx_b: int,
        relative_c2w: np.ndarray,
        loop: bool = False,
    ) -> None:
        """
        Add a between factor encoding: node_a.inverse() ⊗ node_b ≈ relative_c2w.

        For nodes representing cam-to-world, relative_c2w = w2c_a @ c2w_b.

        Args:
            seq_idx_a:    source frame
            seq_idx_b:    target frame
            relative_c2w: (4, 4) measured relative transform in global scale
            loop:         True → use loop closure noise model (looser)
        """
        noise = self._loop_noise if loop else self._between_noise
        self._graph.add(
            BetweenFactorSL4(
                X(seq_idx_a), X(seq_idx_b),
                SL4(relative_c2w.astype(np.float64)),
                noise,
            )
        )

    def get_pose(self, seq_idx: int) -> np.ndarray:
        """(4, 4) current cam-to-world estimate for the given frame."""
        return self._values.atSL4(X(seq_idx)).matrix()

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
            frame_poses[seq_idx] = result.atSL4(X(seq_idx)).matrix().astype(np.float32)

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
