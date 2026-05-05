"""
DA3-SLAM: full pipeline runner.

Wires together:
  KeyframeSelector → SubmapBuilder → SubmapAligner
  → PoseGraph → LoopClosureDetector → optimization
  → trajectory export (KITTI / TUM)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from da3_slam.depth_estimator import DepthEstimator
from da3_slam.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
from da3_slam.submap import Submap, SubmapBuilder
from da3_slam.alignment import SubmapAligner
from da3_slam.factor_graph import PoseGraph, NoiseConfig, OptimizationResult
from da3_slam.loop_closure import LoopClosureDetector, LoopClosureConfig, LoopClosure


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class SLAMConfig:
    # Keyframe selection
    keyframe: KeyframeSelectorConfig = field(
        default_factory=KeyframeSelectorConfig
    )
    # Pose graph noise
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    # Loop closure
    loop_closure: LoopClosureConfig = field(
        default_factory=LoopClosureConfig
    )

    # Frames per submap (including the 1-frame anchor overlap)
    submap_size: int = 8

    # Confidence percentile for point cloud filtering
    conf_percentile: float = 40.0

    # DA3 model
    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    da3_process_res: int = 504

    # Enable loop closure (can disable for speed during debugging)
    enable_loop_closure: bool = True


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class SLAMResult:
    # Optimized (4, 4) cam-to-world poses per keyframe, keyed by seq_idx
    keyframe_poses: dict[int, np.ndarray]

    # All submaps (for point cloud access)
    submaps: list[Submap]

    # Factor graph optimization output
    optimization: OptimizationResult

    # Detected loop closures
    loop_closures: list[LoopClosure]

    # Wall-clock timing breakdown
    timings: dict[str, float]

    @property
    def n_keyframes(self) -> int:
        return len(self.keyframe_poses)

    @property
    def trajectory(self) -> np.ndarray:
        """(N, 4, 4) cam-to-world poses sorted by seq_idx."""
        items = sorted(self.keyframe_poses.items())
        return np.stack([pose for _, pose in items])

    def save_kitti(self, path: str) -> None:
        """
        Save trajectory in KITTI format.
        Each line: 12 space-separated values — the top 3 rows of the
        cam-to-world 4×4 matrix, flattened row-major.
        """
        with open(path, "w") as f:
            for _, pose in sorted(self.keyframe_poses.items()):
                row = pose[:3, :].flatten()
                f.write(" ".join(f"{v:.9e}" for v in row) + "\n")

    def save_tum(self, path: str, fps: float = 30.0) -> None:
        """
        Save trajectory in TUM RGB-D format.
        Each line: timestamp tx ty tz qx qy qz qw
        Timestamps are synthesised from seq_idx / fps.
        """
        from scipy.spatial.transform import Rotation

        with open(path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for seq_idx, pose in sorted(self.keyframe_poses.items()):
                t = pose[:3, 3]
                R = Rotation.from_matrix(pose[:3, :3])
                q = R.as_quat()  # (qx, qy, qz, qw)
                ts = seq_idx / fps
                f.write(
                    f"{ts:.6f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )

    def save_ply(self, path: str) -> None:
        """
        Save the full merged coloured point cloud as PLY.
        Each submap's points are in the submap's local frame; we apply the
        optimized global transform before writing so the cloud aligns with
        the trajectory.
        """
        all_pts = []
        all_col = []
        for sm in self.submaps:
            T_opt = self.optimization.pose(sm.idx)   # (4,4) submap-local → global
            pts = sm.points_world                    # (M, 3) submap-local
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            pts_global = (T_opt @ pts_h.T).T[:, :3]
            all_pts.append(pts_global.astype(np.float32))
            all_col.append(sm.colors)
        all_pts = np.concatenate(all_pts)
        all_col = np.concatenate(all_col)
        n = len(all_pts)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for pt, col in zip(all_pts, all_col):
                f.write(
                    f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{col[0]} {col[1]} {col[2]}\n"
                )


# ── runner ────────────────────────────────────────────────────────────────────

class DA3SLAM:
    """
    Full DA3-SLAM pipeline.

    Usage:
        slam = DA3SLAM(config)
        result = slam.run(image_paths)
        result.save_kitti("trajectory.txt")
        result.save_ply("map.ply")
    """

    def __init__(self, config: SLAMConfig | None = None):
        self.config = config or SLAMConfig()
        cfg = self.config

        self.selector = KeyframeSelector(cfg.keyframe)
        self.estimator = DepthEstimator(
            model_id=cfg.da3_model,
            process_res=cfg.da3_process_res,
        )
        self.builder = SubmapBuilder(
            self.estimator,
            conf_percentile=cfg.conf_percentile,
        )
        self.aligner = SubmapAligner()
        self.detector = LoopClosureDetector(cfg.loop_closure) \
            if cfg.enable_loop_closure else None

    def run(self, image_paths: list[str]) -> SLAMResult:
        timings: dict[str, float] = {}
        cfg = self.config

        # ── 1. Keyframe selection ──────────────────────────────────────────
        t0 = time.time()
        kf_result = self.selector.select_paths(image_paths)
        kf_indices = kf_result.indices
        timings["keyframe_selection"] = time.time() - t0
        print(f"[SLAM] Keyframe selection: {len(kf_indices)}/{len(image_paths)} frames "
              f"({timings['keyframe_selection']:.1f}s)")

        # ── 2. Build submaps ───────────────────────────────────────────────
        t0 = time.time()
        submaps = self.builder.build_sequence(
            image_paths, kf_indices, submap_size=cfg.submap_size
        )
        timings["submap_building"] = time.time() - t0
        total_pts = sum(len(sm.points_world) for sm in submaps)
        print(f"[SLAM] Built {len(submaps)} submaps, "
              f"{total_pts:,} points ({timings['submap_building']:.1f}s)")

        # ── 3. Align + build factor graph ──────────────────────────────────
        t0 = time.time()
        graph = PoseGraph(cfg.noise)
        alignments: list = []
        accumulated_pose = np.eye(4, dtype=np.float32)

        graph.add_submap(submaps[0])
        for i in range(1, len(submaps)):
            alignment = self.aligner.align(submaps[i - 1], submaps[i])
            alignments.append(alignment)
            graph.add_submap(submaps[i], alignment)
            accumulated_pose = accumulated_pose @ alignment.T_a_from_b

            print(f"[SLAM] Submap {i}: "
                  f"rot={alignment.rotation_angle_deg:.2f}°  "
                  f"|t|={np.linalg.norm(alignment.translation):.3f}m")

        timings["graph_building"] = time.time() - t0

        # ── 4. Loop closure ────────────────────────────────────────────────
        loop_closures: list[LoopClosure] = []
        if self.detector is not None:
            t0 = time.time()
            # Reset accumulated pose and replay for loop closure
            acc = np.eye(4, dtype=np.float32)
            self.detector.process(submaps[0], acc)
            for i, (sm, alignment) in enumerate(zip(submaps[1:], alignments)):
                acc = acc @ alignment.T_a_from_b
                lcs = self.detector.process(sm, acc)
                for lc in lcs:
                    loop_closures.append(lc)
                    graph.add_loop_closure(
                        lc.submap_idx_a, lc.submap_idx_b, lc.alignment
                    )
            timings["loop_closure"] = time.time() - t0
            print(f"[SLAM] Loop closures: {len(loop_closures)} "
                  f"({timings['loop_closure']:.1f}s)")

        # ── 5. Optimize ────────────────────────────────────────────────────
        t0 = time.time()
        print(f"[SLAM] Optimizing graph "
              f"({graph.n_nodes} nodes, {graph.n_factors} factors) ...")
        opt_result = graph.optimize(verbose=True)
        timings["optimization"] = time.time() - t0
        print(f"[SLAM] Optimization done: "
              f"error {opt_result.final_error:.4f}, "
              f"{opt_result.iterations} iters "
              f"({timings['optimization']:.1f}s)")

        # ── 6. Build per-keyframe trajectory ───────────────────────────────
        keyframe_poses = _build_keyframe_poses(submaps, opt_result)

        return SLAMResult(
            keyframe_poses=keyframe_poses,
            submaps=submaps,
            optimization=opt_result,
            loop_closures=loop_closures,
            timings=timings,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_keyframe_poses(
    submaps: list[Submap],
    opt: OptimizationResult,
) -> dict[int, np.ndarray]:
    """
    Compute global cam-to-world pose for every keyframe.

    For each frame in each submap:
        c2w_global = T_opt_submap @ inv(frame.extrinsic)

    Where T_opt_submap is the optimized pose of the submap's local frame
    in the global frame.
    """
    poses: dict[int, np.ndarray] = {}
    for submap in submaps:
        T_opt = opt.pose(submap.idx)   # (4,4) submap frame → global
        for frame in submap.frames:
            c2w_local = frame.cam_to_world   # (4,4) cam → submap local
            c2w_global = T_opt @ c2w_local
            # Anchor frame appears in two submaps — first wins
            if frame.seq_idx not in poses:
                poses[frame.seq_idx] = c2w_global.astype(np.float32)
    return poses
