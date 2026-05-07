"""
DA3-SLAM: full pipeline runner.

Wires together:
  OnlineKeyframeSelector → SubmapBuilder → SubmapAligner
  → PoseGraph → LoopClosureDetector → optimization
  → trajectory export (KITTI / TUM)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from da3_slam.frontend.depth_estimator import DepthEstimator
from da3_slam.frontend.keyframe_selector import OnlineKeyframeSelector, KeyframeSelectorConfig
from da3_slam.frontend.submap import Submap, SubmapBuilder
from da3_slam.backend.alignment import SubmapAligner
from da3_slam.backend.factor_graph import PoseGraph, NoiseConfig, OptimizationResult
from da3_slam.backend.loop_closure import LoopClosureDetector, LoopClosureConfig, LoopClosure


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class SLAMConfig:
    # Canonical values: config/default.yaml
    # Use da3_slam.config.load_slam_config() to construct from YAML.

    keyframe: KeyframeSelectorConfig
    noise: NoiseConfig
    loop_closure: LoopClosureConfig

    # Frames per submap (including the 1-frame anchor overlap)
    submap_size: int

    # Global confidence percentile threshold for point cloud filtering
    confidence_percentile: float

    # DA3 model
    depth_model: str
    depth_model_resolution: int

    # Enable loop closure (can disable for speed during debugging)
    enable_loop_closure: bool


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

    def save_tum(
        self,
        path: str,
        fps: float = 30.0,
        timestamps: dict[int, float] | None = None,
    ) -> None:
        """
        Save trajectory in TUM RGB-D format.
        Each line: timestamp tx ty tz qx qy qz qw

        Args:
            path:       output file path
            fps:        fallback frame rate used to synthesise timestamps when
                        `timestamps` is not provided
            timestamps: optional mapping {seq_idx: real_timestamp_seconds}.
                        When provided these are used instead of seq_idx / fps.
                        Frames with no entry fall back to seq_idx / fps.
        """
        from scipy.spatial.transform import Rotation

        with open(path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for seq_idx, pose in sorted(self.keyframe_poses.items()):
                t = pose[:3, 3]
                R = Rotation.from_matrix(pose[:3, :3])
                q = R.as_quat()  # (qx, qy, qz, qw)
                if timestamps is not None and seq_idx in timestamps:
                    ts = timestamps[seq_idx]
                else:
                    ts = seq_idx / fps
                f.write(
                    f"{ts:.6f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )

    def save_ply(self, path: str) -> None:
        """
        Save the full merged coloured point cloud as binary PLY.
        Each submap's points are in the submap's local frame; we apply the
        optimized global transform before writing so the cloud aligns with
        the trajectory.
        """
        all_pts = []
        all_col = []
        for submap in self.submaps:
            T_opt = self.optimization.pose(submap.idx)   # (4,4) submap-local → global
            pts = submap.points_world                    # (M, 3) float32, submap-local
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            all_pts.append((T_opt @ pts_h.T).T[:, :3].astype(np.float32))
            all_col.append(submap.colors)

        all_pts = np.concatenate(all_pts)  # (N, 3) float32
        all_col = np.concatenate(all_col)  # (N, 3) uint8
        n = len(all_pts)

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        # Pack each vertex as 12 bytes xyz (float32) + 3 bytes rgb (uint8).
        # view(uint8) reinterprets the float32 memory; hstack interleaves them.
        xyz_bytes = all_pts.view(np.uint8).reshape(n, 12)
        vertex_data = np.hstack([xyz_bytes, all_col])  # (N, 15)

        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(vertex_data.tobytes())


# ── run context ───────────────────────────────────────────────────────────────

@dataclass
class _RunContext:
    """Shared mutable state passed between the frontend and backend threads."""
    config:            SLAMConfig
    batch_queue:       queue.Queue
    submaps:           list[Submap]
    loop_closures:     list[LoopClosure]
    frontend_timings:  dict[str, float]
    backend_timings:   dict[str, float]
    opt_result:        OptimizationResult | None = None
    backend_error:     BaseException | None = None


def _enqueue(ctx: _RunContext, paths: list[str], indices: list[int]) -> None:
    """Put a keyframe batch on the queue; give up silently if the backend crashed."""
    while True:
        try:
            ctx.batch_queue.put((paths, indices), timeout=0.5)
            return
        except queue.Full:
            if ctx.backend_error is not None:
                return


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

        self.estimator = DepthEstimator(
            model_id=cfg.depth_model,
            process_resolution=cfg.depth_model_resolution,
        )
        self.builder = SubmapBuilder(
            self.estimator,
            confidence_percentile=cfg.confidence_percentile,
        )
        self.aligner = SubmapAligner()
        self.detector = LoopClosureDetector(cfg.loop_closure) \
            if cfg.enable_loop_closure else None

    def run(self, image_paths: list[str]) -> SLAMResult:
        ctx = _RunContext(
            config=self.config,
            batch_queue=queue.Queue(maxsize=2),
            submaps=[],
            loop_closures=[],
            frontend_timings={"keyframe_selection": 0.0},
            backend_timings={
                "submap_building": 0.0,
                "graph_building":  0.0,
                "loop_closure":    0.0,
                "optimization":    0.0,
            },
        )

        # Start backend first so it is ready before the frontend produces anything.
        backend_thread = threading.Thread(target=self._backend, args=(ctx,),
                                          name="da3-backend", daemon=True)
        frontend_thread = threading.Thread(target=self._frontend, args=(image_paths, ctx),
                                           name="da3-frontend", daemon=True)
        wall_start = time.time()
        backend_thread.start()
        frontend_thread.start()
        frontend_thread.join()
        backend_thread.join()
        wall_elapsed = time.time() - wall_start

        if ctx.backend_error is not None:
            raise ctx.backend_error

        timings = {**ctx.frontend_timings, **ctx.backend_timings}
        col = max(len(k) for k in timings)
        print("[SLAM] Timing breakdown (per-module compute time, threads overlap):")
        for module, seconds in timings.items():
            print(f"  {module:<{col}}  {seconds:6.1f}s")
        print(f"  {'':-<{col+9}}")
        print(f"  {'compute total':<{col}}  {sum(timings.values()):6.1f}s")
        print(f"  {'wall-clock':<{col}}  {wall_elapsed:6.1f}s")

        keyframe_poses = _build_keyframe_poses(ctx.submaps, ctx.opt_result)

        return SLAMResult(
            keyframe_poses=keyframe_poses,
            submaps=ctx.submaps,
            optimization=ctx.opt_result,
            loop_closures=ctx.loop_closures,
            timings=timings,
        )

    def _frontend(self, image_paths: list[str], ctx: _RunContext) -> None:
        selector = OnlineKeyframeSelector(ctx.config.keyframe)
        keyframe_paths:   list[str] = []
        keyframe_indices: list[int] = []
        try:
            for i, path in enumerate(image_paths):
                if ctx.backend_error is not None:
                    break

                t0 = time.time()
                is_keyframe = selector.step_path(path)
                ctx.frontend_timings["keyframe_selection"] += time.time() - t0

                if is_keyframe:
                    keyframe_paths.append(path)
                    keyframe_indices.append(i)

                if len(keyframe_paths) >= ctx.config.submap_size:
                    _enqueue(ctx, list(keyframe_paths), list(keyframe_indices))
                    # 1-frame overlap: anchor next submap on the last keyframe
                    keyframe_paths[:] = [keyframe_paths[-1]]
                    keyframe_indices[:] = [keyframe_indices[-1]]

            if len(keyframe_paths) >= 2 and ctx.backend_error is None:
                _enqueue(ctx, list(keyframe_paths), list(keyframe_indices))
        finally:
            ctx.batch_queue.put(None)  # sentinel — always sent, even on error

    def _backend(self, ctx: _RunContext) -> None:
        pose_graph       = PoseGraph(ctx.config.noise)
        accumulated_pose = np.eye(4, dtype=np.float32)
        try:
            while True:
                item = ctx.batch_queue.get()
                if item is None:
                    break
                paths, indices = item

                # ── build submap ───────────────────────────────────────────
                t0 = time.time()
                submap = self.builder.build(paths, indices, len(ctx.submaps))
                ctx.backend_timings["submap_building"] += time.time() - t0

                # ── align + add to graph ───────────────────────────────────
                t0 = time.time()
                if ctx.submaps:
                    alignment = self.aligner.align(ctx.submaps[-1], submap)
                    pose_graph.add_submap(submap, alignment)
                    accumulated_pose = accumulated_pose @ alignment.T_a_from_b
                    print(f"[SLAM] Submap {submap.idx}: "
                          f"rot={alignment.rotation_angle_deg:.2f}°  "
                          f"|t|={np.linalg.norm(alignment.translation):.3f}m")
                else:
                    pose_graph.add_submap(submap)
                ctx.submaps.append(submap)
                ctx.backend_timings["graph_building"] += time.time() - t0

                # ── loop closure ───────────────────────────────────────────
                if self.detector is not None:
                    t0 = time.time()
                    closures = self.detector.process(submap, accumulated_pose)
                    for closure in closures:
                        ctx.loop_closures.append(closure)
                        pose_graph.add_loop_closure(
                            closure.submap_idx_a, closure.submap_idx_b, closure.alignment
                        )
                    ctx.backend_timings["loop_closure"] += time.time() - t0

                # ── incremental optimization ───────────────────────────────
                t0 = time.time()
                pose_graph.optimize()
                ctx.backend_timings["optimization"] += time.time() - t0

            if not ctx.submaps:
                raise RuntimeError(
                    "No submaps built — sequence too short or no keyframes detected."
                )

            # ── final optimization ─────────────────────────────────────────
            print(f"[SLAM] Final optimization "
                  f"({pose_graph.n_nodes} nodes, {pose_graph.n_factors} factors) ...")
            t0 = time.time()
            ctx.opt_result = pose_graph.optimize(verbose=True)
            ctx.backend_timings["optimization"] += time.time() - t0
            print(f"[SLAM] Done: error {ctx.opt_result.final_error:.4f}, "
                  f"{ctx.opt_result.iterations} iters, "
                  f"{len(ctx.submaps)} submaps, {len(ctx.loop_closures)} loop closures")

        except Exception as exc:
            ctx.backend_error = exc


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
            cam_to_world_local  = frame.cam_to_world   # (4,4) cam → submap local
            cam_to_world_global = T_opt @ cam_to_world_local
            # Anchor frame appears in two submaps — first wins
            if frame.seq_idx not in poses:
                poses[frame.seq_idx] = cam_to_world_global.astype(np.float32)
    return poses
