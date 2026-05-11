"""
Cross-submap alignment.

Each submap is processed independently by DA3, which picks its own reference
frame. This module computes the rigid transform that maps submap B's world
frame into submap A's world frame, enabling a globally consistent map.

Strategy — anchor frame:
    Consecutive submaps share one physical frame (last of A = first of B).
    The anchor frame has identical camera-space coordinates in both world
    systems, so the transform from world_B to world_A is exact:

        world_b_to_world_a = inv(anchor_world_a_to_cam) @ anchor_world_b_to_cam

    Proof: for any point P at the anchor frame,
        anchor_world_a_to_cam @ P_in_world_a = P_in_cam
        anchor_world_b_to_cam @ P_in_world_b = P_in_cam
        => P_in_world_a = inv(anchor_world_a_to_cam) @ anchor_world_b_to_cam @ P_in_world_b
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_slam.backend.inference.submap import Submap


@dataclass
class AlignmentResult:
    # (4, 4) float32 — transforms points expressed in world_B into world_A.
    # For SE3 anchor alignment: [:3,:3] = R.
    # For Sim3 ICP alignment:   [:3,:3] = scale * R.
    world_b_to_world_a: np.ndarray

    # Alignment method used
    method: str  # "anchor" | "icp"

    # Sim3 scale factor (1.0 for pure SE3 anchor alignment).
    # For ICP results this is the Umeyama scale: s such that [:3,:3] = s·R.
    scale: float = 1.0

    @property
    def rotation(self) -> np.ndarray:
        """(3, 3) pure SO3 rotation (scale-normalised for Sim3 ICP results)."""
        return self.world_b_to_world_a[:3, :3] / self.scale

    @property
    def translation(self) -> np.ndarray:
        """(3,) translation component."""
        return self.world_b_to_world_a[:3, 3]

    @property
    def rotation_angle_deg(self) -> float:
        """Rotation magnitude in degrees (axis-angle)."""
        cos = np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))


class SubmapAligner:
    """
    Aligns consecutive submaps into a common world frame.

    Requires that consecutive submaps share one anchor frame:
    the last frame of submap A is also the first frame of submap B.
    This is guaranteed by the 1-frame overlap maintained in DA3SLAM.run().
    """

    def align(self, submap_a: Submap, submap_b: Submap) -> AlignmentResult:
        """
        Compute the transform that maps submap B's world frame to submap A's.

        Uses the anchor frame (last of A / first of B) for an exact solution.
        Scale is left at 1.0; the Sim3 pose graph optimizer finds per-submap
        scales from loop closure constraints.
        """
        assert submap_a.frames[-1].seq_idx == submap_b.frames[0].seq_idx, (
            f"Submaps {submap_a.idx} and {submap_b.idx} do not share an anchor frame "
            f"(last seq_idx={submap_a.frames[-1].seq_idx}, "
            f"first seq_idx={submap_b.frames[0].seq_idx})"
        )

        # Cast to float64 for the inversion; extrinsics from DA3 are float32
        # and LU decomposition accumulates more error at single precision.
        anchor_world_a_to_cam = submap_a.frames[-1].extrinsic.astype(np.float64)
        anchor_world_b_to_cam = submap_b.frames[0].extrinsic.astype(np.float64)

        world_b_to_world_a = np.linalg.inv(anchor_world_a_to_cam) @ anchor_world_b_to_cam

        return AlignmentResult(
            world_b_to_world_a=world_b_to_world_a.astype(np.float32),
            method="anchor",
        )
