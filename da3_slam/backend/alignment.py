"""
Cross-submap alignment.

Each submap is processed independently by DA3, which picks its own reference
frame. This module computes the rigid transform that maps submap B's world
frame into submap A's world frame, enabling a globally consistent map.

Strategy — anchor frame:
    When consecutive submaps share one frame (last of A = first of B), the
    exact transform is:

        T_AB = inv(E_A_last) @ E_B_first

    Proof: for the anchor frame, camera coords are identical in both systems:
        E_A_last @ P_worldA = E_B_first @ P_worldB
        => P_worldA = inv(E_A_last) @ E_B_first @ P_worldB
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_slam.frontend.submap import Submap


@dataclass
class AlignmentResult:
    # (4, 4) float32 — transforms world_B points into world_A coordinates
    T_a_from_b: np.ndarray

    # Alignment method used
    method: str  # "anchor" | "icp"

    @property
    def rotation(self) -> np.ndarray:
        """(3, 3) rotation component."""
        return self.T_a_from_b[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """(3,) translation component."""
        return self.T_a_from_b[:3, 3]

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
        Compute T that maps submap_b's world frame to submap_a's world frame.

        Uses the anchor frame (last of A / first of B) for an exact solution.
        """
        extrinsic_a = submap_a.frames[-1].extrinsic  # (4, 4) world_A-to-cam
        extrinsic_b = submap_b.frames[0].extrinsic   # (4, 4) world_B-to-cam
        transform = np.linalg.inv(extrinsic_a) @ extrinsic_b
        return AlignmentResult(T_a_from_b=transform.astype(np.float32), method="anchor")
