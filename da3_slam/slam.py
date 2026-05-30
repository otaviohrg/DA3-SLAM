"""
DA3-SLAM: full pipeline runner.

Wires together:
  OnlineKeyframeSelector → SubmapBuilder
  → PoseGraph → LoopClosureDetector → optimization
  → trajectory export (KITTI / TUM)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from da3_slam.frontend.keyframe_selector import OnlineKeyframeSelector, KeyframeSelectorConfig
from da3_slam.backend.inference.depth_estimator import DepthEstimator
from da3_slam.backend.inference.submap import Submap, SubmapBuilder
from da3_slam.backend.processing.factor_graph import PoseGraph, NoiseConfig, OptimizationResult
from da3_slam.backend.processing.loop_closure import LoopClosureDetector, LoopClosureConfig, LoopClosure


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

    # Fields with defaults must come after all non-default fields
    use_ray_pose: bool = False

    # HuggingFace CLIP model ID for semantic embeddings (None = disabled)
    semantic_model: str | None = None


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
        return np.stack([pose for _, pose in sorted(self.keyframe_poses.items())])

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
                translation = pose[:3, 3]
                quaternion  = Rotation.from_matrix(pose[:3, :3]).as_quat()  # (qx, qy, qz, qw)
                ts = timestamps[seq_idx] if timestamps is not None and seq_idx in timestamps else seq_idx / fps
                f.write(
                    f"{ts:.6f} "
                    f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                    f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n"
                )

    def save_ply(self, path: str) -> None:
        """
        Save the full merged coloured point cloud as binary PLY.
        Each frame's camera-space points are projected to global world via the
        per-frame optimised cam-to-world pose from GTSAM.
        """
        all_points = []
        all_colors = []
        seen_seq_idx: set[int] = set()
        for submap in self.submaps:
            if submap.is_lc_submap:
                continue
            for frame in submap.frames:
                if frame.seq_idx in seen_seq_idx:
                    continue
                seen_seq_idx.add(frame.seq_idx)
                pts = frame.points_cam              # (M, 3) in camera space
                if len(pts) == 0:
                    continue
                global_c2w = self.optimization.pose(frame.seq_idx).astype(np.float64)
                homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
                all_points.append((global_c2w @ homo.T).T[:, :3].astype(np.float32))
                all_colors.append(frame.colors)

        all_points = np.concatenate(all_points)  # (N, 3) float32
        all_colors = np.concatenate(all_colors)  # (N, 3) uint8
        n = len(all_points)

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
        xyz_bytes   = np.ascontiguousarray(all_points).view(np.uint8).reshape(n, 12)
        vertex_data = np.hstack([xyz_bytes, all_colors])  # (N, 15)

        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(vertex_data.tobytes())

    def retrieve_best_semantic_frame(
        self,
        text_embedding: np.ndarray,
    ) -> tuple[int, int, float] | None:
        """
        Find the frame whose CLIP embedding best matches a text embedding.

        Args:
            text_embedding: (D,) float32 L2-normalised CLIP text vector,
                            produced by SemanticEmbedder.encode_text()

        Returns:
            (submap_idx, frame_index_in_submap, cosine_similarity) of the
            best-matching frame, or None if no semantic embeddings are stored.
        """
        best_submap_idx = None
        best_frame_idx  = None
        best_sim        = -np.inf

        for submap in self.submaps:
            for frame_idx, frame in enumerate(submap.frames):
                if frame.semantic_vector is None:
                    continue
                sim = float(np.dot(text_embedding, frame.semantic_vector))
                if sim > best_sim:
                    best_sim        = sim
                    best_submap_idx = submap.idx
                    best_frame_idx  = frame_idx

        if best_submap_idx is None:
            return None
        return best_submap_idx, best_frame_idx, best_sim


# ── run context ───────────────────────────────────────────────────────────────

@dataclass
class _RunContext:
    """Shared mutable state passed between the frontend, inference, processing, and LC threads."""
    config:          SLAMConfig
    batch_queue:     queue.Queue   # frontend   → inference  (paths, images, indices)
    submap_queue:    queue.Queue   # inference  → processing (Submap)
    lc_queue:        queue.Queue   # processing → lc         (submap, dict snaps)
    lc_result_queue: queue.Queue   # lc         → processing (factor tuples)
    lc_done:         threading.Event
    submaps:         list[Submap]
    loop_closures:   list[LoopClosure]
    timings:         dict[str, float]
    opt_result:      OptimizationResult | None = None
    backend_error:   BaseException | None = None


def _drain_lc_results(ctx: _RunContext, pose_graph: PoseGraph) -> None:
    """Add all completed LC results from lc_result_queue into the pose graph."""
    tag = f"[{threading.current_thread().name}]"
    while True:
        try:
            seq_b, seq_a, relative_lc, closure = ctx.lc_result_queue.get_nowait()
            ctx.loop_closures.append(closure)
            pose_graph.add_between(seq_b, seq_a, relative_lc, loop=True)
            print(f"{tag} LC {closure.candidate.submap_idx_b}"
                  f"[f{closure.candidate.frame_idx_b}]"
                  f" ↔ {closure.candidate.submap_idx_a}"
                  f"[f{closure.candidate.frame_idx_a}]")
        except queue.Empty:
            break


def _blocking_put(ctx: _RunContext, q: queue.Queue, item) -> bool:
    """Put item on q, retrying every 0.5 s until space is available or a thread error is set.

    Returns False if a thread error was set before the item could be placed.
    """
    while True:
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            if ctx.backend_error is not None:
                return False


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
            use_ray_pose=cfg.use_ray_pose,
        )
        self.builder = SubmapBuilder(
            self.estimator,
            confidence_percentile=cfg.confidence_percentile,
        )
        self.detector = LoopClosureDetector(cfg.loop_closure, builder=self.builder) \
            if cfg.enable_loop_closure else None

        self.semantic_embedder = None
        if cfg.semantic_model:
            from da3_slam.backend.inference.semantic_embedder import SemanticEmbedder
            self.semantic_embedder = SemanticEmbedder(cfg.semantic_model)

    def run(self, image_paths: list[str]) -> SLAMResult:
        lc_done = threading.Event()
        if self.detector is None:
            lc_done.set()  # no LC thread — event is immediately done

        ctx = _RunContext(
            config=self.config,
            batch_queue=queue.Queue(maxsize=2),
            submap_queue=queue.Queue(maxsize=2),
            lc_queue=queue.Queue(),
            lc_result_queue=queue.Queue(),
            lc_done=lc_done,
            submaps=[],
            loop_closures=[],
            timings={
                "keyframe_selection": 0.0,
                "submap_building":    0.0,
                "graph_building":     0.0,
                "loop_closure":       0.0,
                "optimization":       0.0,
            },
        )

        # Start consumers before producers so they are ready immediately.
        processing_thread = threading.Thread(target=self._processing, args=(ctx,),
                                             name="da3-processing", daemon=True)
        inference_thread  = threading.Thread(target=self._inference,  args=(ctx,),
                                             name="da3-inference",   daemon=True)
        frontend_thread   = threading.Thread(target=self._frontend, args=(image_paths, ctx),
                                             name="da3-frontend",    daemon=True)
        wall_start = time.time()
        processing_thread.start()
        inference_thread.start()
        if self.detector is not None:
            lc_thread = threading.Thread(target=self._lc, args=(ctx,),
                                         name="da3-lc", daemon=True)
            lc_thread.start()
        frontend_thread.start()
        frontend_thread.join()
        inference_thread.join()
        processing_thread.join()
        if self.detector is not None:
            lc_thread.join()
        wall_elapsed = time.time() - wall_start

        if ctx.backend_error is not None:
            raise ctx.backend_error

        timings = ctx.timings
        col = max(len(k) for k in timings)
        tag = f"[{threading.current_thread().name}]"
        print(f"{tag} Timing breakdown (per-module compute time, threads overlap):")
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
        keyframe_paths:   list[str]        = []
        keyframe_images:  list[np.ndarray] = []
        keyframe_indices: list[int]        = []
        try:
            for i, path in enumerate(image_paths):
                if ctx.backend_error is not None:
                    break

                bgr = cv2.imread(path)
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                t0 = time.time()
                is_keyframe = selector.step(image)
                ctx.timings["keyframe_selection"] += time.time() - t0

                if is_keyframe:
                    keyframe_paths.append(path)
                    keyframe_images.append(image)
                    keyframe_indices.append(i)

                if len(keyframe_paths) >= ctx.config.submap_size:
                    _blocking_put(ctx, ctx.batch_queue, (
                        list(keyframe_paths),
                        list(keyframe_images),
                        list(keyframe_indices),
                    ))
                    # 1-frame overlap: anchor next submap on the last keyframe
                    keyframe_paths[:]   = [keyframe_paths[-1]]
                    keyframe_images[:]  = [keyframe_images[-1]]
                    keyframe_indices[:] = [keyframe_indices[-1]]

            if len(keyframe_paths) >= 2 and ctx.backend_error is None:
                _blocking_put(ctx, ctx.batch_queue, (
                    list(keyframe_paths),
                    list(keyframe_images),
                    list(keyframe_indices),
                ))
        finally:
            ctx.batch_queue.put(None)  # sentinel — always sent, even on error

    def _inference(self, ctx: _RunContext) -> None:
        """DA3 inference: pops batches from batch_queue, pushes built Submaps to submap_queue."""
        submap_idx = 0
        try:
            while True:
                item = ctx.batch_queue.get()
                if item is None:
                    break
                if ctx.backend_error is not None:
                    break
                paths, images, indices = item

                t0 = time.time()
                submap = self.builder.build(paths, images, indices, submap_idx)
                if self.semantic_embedder is not None:
                    semantic_vecs = self.semantic_embedder.encode_frames(submap)
                    submap.set_all_semantic_vectors(semantic_vecs)
                ctx.timings["submap_building"] += time.time() - t0
                submap_idx += 1

                if not _blocking_put(ctx, ctx.submap_queue, submap):
                    return

        except Exception as exc:
            ctx.backend_error = exc
        finally:
            try:
                ctx.submap_queue.put(None, timeout=1.0)
            except queue.Full:
                pass  # processing is already dead; sentinel is not needed

    def _processing(self, ctx: _RunContext) -> None:
        """Graph building and incremental optimisation. LC is dispatched to _lc thread."""
        pose_graph = PoseGraph(ctx.config.noise)

        accumulated_scale: float = 1.0
        submap_scales: dict[int, float] = {}
        submap_dict:   dict[int, Submap] = {}

        try:
            while True:
                submap = ctx.submap_queue.get()
                if submap is None:
                    break

                # ── build per-frame graph nodes ────────────────────────────
                t0  = time.time()
                tag = f"[{threading.current_thread().name}]"
                prev_submap = ctx.submaps[-1] if ctx.submaps else None

                if prev_submap is None:
                    for frame in submap.frames:
                        pose_graph.add_frame(
                            frame.seq_idx,
                            frame.cam_to_world.astype(np.float64),
                        )
                    pose_graph.add_prior(submap.frames[0].seq_idx)
                else:
                    delta_scale = _estimate_boundary_scale(prev_submap, submap)
                    accumulated_scale *= delta_scale
                    anchor_global_c2w = pose_graph.get_pose(submap.frames[0].seq_idx)
                    anchor_local_w2c  = submap.frames[0].extrinsic.astype(np.float64)
                    for frame in submap.frames[1:]:
                        local_c2w       = frame.cam_to_world.astype(np.float64)
                        relative        = anchor_local_w2c @ local_c2w
                        relative_scaled = relative.copy()
                        relative_scaled[:3, 3] *= accumulated_scale
                        pose_graph.add_frame(
                            frame.seq_idx,
                            anchor_global_c2w @ relative_scaled,
                        )
                    print(f"{tag} Submap {submap.idx}: scale={accumulated_scale:.4f} "
                          f"(Δ={delta_scale:.4f})")

                for i in range(1, len(submap.frames)):
                    f_prev = submap.frames[i - 1]
                    f_curr = submap.frames[i]
                    relative        = f_prev.extrinsic.astype(np.float64) @ f_curr.cam_to_world.astype(np.float64)
                    relative_scaled = relative.copy()
                    relative_scaled[:3, 3] *= accumulated_scale
                    pose_graph.add_between(f_prev.seq_idx, f_curr.seq_idx, relative_scaled)

                ctx.submaps.append(submap)
                submap_scales[submap.idx] = accumulated_scale
                submap_dict[submap.idx]   = submap
                ctx.timings["graph_building"] += time.time() - t0

                # ── dispatch submap to LC thread ───────────────────────────
                if self.detector is not None:
                    ctx.lc_queue.put((submap, dict(submap_dict), dict(submap_scales)))

                # ── drain any completed LC results into the graph ──────────
                _drain_lc_results(ctx, pose_graph)

                # ── incremental optimisation ───────────────────────────────
                t0  = time.time()
                opt = pose_graph.optimize()
                ctx.timings["optimization"] += time.time() - t0
                print(f"{tag} Submap {submap.idx}: "
                      f"{pose_graph.n_nodes} nodes  "
                      f"{pose_graph.n_factors} factors  "
                      f"error={opt.final_error:.4f}")

            if ctx.backend_error is not None:
                return

            if not ctx.submaps:
                raise RuntimeError(
                    "No submaps built — sequence too short or no keyframes detected."
                )

            # ── wait for LC thread to finish, then incorporate final results ──
            if self.detector is not None:
                ctx.lc_queue.put(None)  # sentinel
                ctx.lc_done.wait()
            _drain_lc_results(ctx, pose_graph)

            # ── final optimisation ─────────────────────────────────────────
            tag = f"[{threading.current_thread().name}]"
            print(f"{tag} Final optimisation "
                  f"({pose_graph.n_nodes} nodes, {pose_graph.n_factors} factors) ...")
            t0 = time.time()
            ctx.opt_result = pose_graph.optimize(verbose=True)
            ctx.timings["optimization"] += time.time() - t0
            print(f"{tag} Done: error={ctx.opt_result.final_error:.4f}  "
                  f"iters={ctx.opt_result.iterations}  "
                  f"submaps={len(ctx.submaps)}  "
                  f"loop_closures={len(ctx.loop_closures)}")

        except Exception as exc:
            ctx.backend_error = exc

    def _lc(self, ctx: _RunContext) -> None:
        """Loop closure thread: detects and scores candidates, posts results for _processing."""
        try:
            while True:
                item = ctx.lc_queue.get()
                if item is None:
                    break
                if ctx.backend_error is not None:
                    break

                submap, submap_dict_snap, submap_scales_snap = item

                t0 = time.time()
                closures = self.detector.process(submap, None)
                ctx.timings["loop_closure"] += time.time() - t0

                for closure in closures:
                    lc_e0 = closure.lc_submap.frames[0].extrinsic.astype(np.float64)
                    lc_e1 = closure.lc_submap.frames[1].extrinsic.astype(np.float64)
                    relative_lc = lc_e0 @ np.linalg.inv(lc_e1)

                    submap_b = submap_dict_snap[closure.candidate.submap_idx_b]
                    frame_b  = submap_b.frames[closure.candidate.frame_idx_b]

                    s_lc = _estimate_depth_scale(
                        frame_b.depth,
                        closure.lc_submap.frames[0].depth,
                    )
                    relative_lc[:3, 3] *= s_lc * submap_scales_snap[closure.candidate.submap_idx_b]

                    ctx.lc_result_queue.put((
                        frame_b.seq_idx,
                        submap_dict_snap[closure.candidate.submap_idx_a]
                            .frames[closure.candidate.frame_idx_a].seq_idx,
                        relative_lc,
                        closure,
                    ))

        except Exception as exc:
            ctx.backend_error = exc
        finally:
            ctx.lc_done.set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_keyframe_poses(
    submaps: list[Submap],
    opt: OptimizationResult,
) -> dict[int, np.ndarray]:
    """
    Return the per-frame cam-to-world poses from the GTSAM optimisation result.

    The anchor frame (shared between consecutive submaps) is included once —
    the first submap that contributed it wins.  LC submaps are skipped.
    """
    poses: dict[int, np.ndarray] = {}
    for submap in submaps:
        if submap.is_lc_submap:
            continue
        for frame in submap.frames:
            if frame.seq_idx not in poses:
                poses[frame.seq_idx] = opt.pose(frame.seq_idx).astype(np.float32)
    return poses


def _estimate_boundary_scale(prev_submap: Submap, curr_submap: Submap) -> float:
    """
    Estimate the metric scale of curr_submap relative to prev_submap.

    The anchor frame (last of prev, first of curr) observed the same scene in
    both DA3 batches.  The median depth ratio gives the scale factor needed to
    bring curr_submap's translations into the same metric unit as prev_submap.
    """
    return _estimate_depth_scale(
        prev_submap.frames[-1].depth,
        curr_submap.frames[0].depth,
    )


def _estimate_depth_scale(depth_ref: np.ndarray, depth_new: np.ndarray) -> float:
    """
    Median ratio depth_ref / depth_new over valid pixels.

    If shapes differ (different DA3 resolutions), depth_new is resized to
    match depth_ref before comparison.  Returns 1.0 if no valid pixels exist.
    """
    if depth_ref.shape != depth_new.shape:
        import cv2
        depth_new = cv2.resize(
            depth_new, (depth_ref.shape[1], depth_ref.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    valid = (
        (depth_ref > 0) & (depth_new > 0) &
        np.isfinite(depth_ref) & np.isfinite(depth_new)
    )
    if not valid.any():
        return 1.0
    return float(np.median(depth_ref[valid] / depth_new[valid]))
