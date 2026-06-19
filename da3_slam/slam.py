"""
DA3-SLAM: full pipeline runner.

Wires together:
  OnlineKeyframeSelector → SubmapBuilder
  → PoseGraph → LoopClosureDetector → optimization
  → trajectory export (KITTI / TUM)

Threading model (DA3SLAM.run):
  Four daemon threads communicate over bounded queues; None is the
  end-of-stream sentinel on every queue.

    _frontend  →(batch_queue)→  _inference  →(submap_queue)→  _processing
                                                  ⇅
                                  (lc_queue / lc_result_queue)
                                                  ⇅
                                          _loop_closure_worker

  Errors in any worker are recorded in _RunContext.backend_error; the other
  threads poll it and shut down, and run() re-raises it at the end.

Two structural invariants that are easy to break (mirrors VGGT-SLAM):

  1. Anchor-frame submap bridging.  Consecutive submaps share one physical
     keyframe (last of submap N = first of submap N+1, same seq_idx).  That
     shared node is the *only* connection between submaps in the pose graph —
     there is no explicit inter-submap factor.  If the batching in _frontend
     changes, the 1-frame overlap must be preserved or the graph disconnects.

  2. Per-submap metric scale accumulation.  Each DA3 batch has its own
     arbitrary metric scale.  The median depth ratio at the shared anchor
     frame gives the scale change between consecutive submaps; the running
     product (accumulated_scale) converts every measurement's *translation*
     into the metric unit of submap 0 before it enters the graph.  Rotation
     is never scaled.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from da3_slam.config import SLAMConfig, load_slam_config
from da3_slam.frontend.keyframe_selector import OnlineKeyframeSelector
from da3_slam.backend.inference.depth_estimator import DepthEstimator
from da3_slam.backend.inference.submap import Submap, SubmapBuilder, transform_points
from da3_slam.backend.processing.factor_graph import PoseGraph, OptimizationResult
from da3_slam.backend.processing.loop_closure import LoopClosureDetector, LoopClosure

__all__ = ["SLAMConfig", "SLAMResult", "DA3SLAM", "FrameItem"]

# A streamed input frame: (RGB image HxWx3 uint8, seq_idx, label).
#   seq_idx must be unique and strictly increasing in arrival order — it keys
#           the pose-graph node and orders the output trajectory.
#   label   is provenance metadata only (file path, topic+timestamp, …); it is
#           stored on the submap but never used for computation.
FrameItem = tuple[np.ndarray, int, str]


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
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()  # (qx, qy, qz, qw)
                ts = timestamps[seq_idx] if timestamps is not None and seq_idx in timestamps \
                    else seq_idx / fps
                f.write(
                    f"{ts:.6f} "
                    f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                    f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} "
                    f"{quaternion[3]:.9f}\n"
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
                # The anchor frame appears in two submaps; keep it once.
                if frame.seq_idx in seen_seq_idx:
                    continue
                seen_seq_idx.add(frame.seq_idx)
                if len(frame.points_cam) == 0:
                    continue
                global_c2w = self.optimization.pose(frame.seq_idx).astype(np.float64)
                all_points.append(transform_points(frame.points_cam, global_c2w))
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
        xyz_bytes = np.ascontiguousarray(all_points).view(np.uint8).reshape(n, 12)
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
        best: tuple[int, int, float] | None = None
        for submap in self.submaps:
            for frame_idx, frame in enumerate(submap.frames):
                if frame.semantic_vector is None:
                    continue
                similarity = float(np.dot(text_embedding, frame.semantic_vector))
                if best is None or similarity > best[2]:
                    best = (submap.idx, frame_idx, similarity)
        return best


# ── run context ───────────────────────────────────────────────────────────────

@dataclass
class _RunContext:
    """Shared mutable state passed between the four pipeline threads."""
    config: SLAMConfig
    batch_queue: queue.Queue       # frontend   → inference  (paths, images, indices)
    submap_queue: queue.Queue      # inference  → processing (Submap)
    lc_queue: queue.Queue          # processing → lc worker  (submap + dict snapshots)
    lc_result_queue: queue.Queue   # lc worker  → processing (seq_b, seq_a, relative, closure)
    lc_done: threading.Event
    submaps: list[Submap]
    loop_closures: list[LoopClosure]
    timings: dict[str, float]
    opt_result: OptimizationResult | None = None

    # First exception raised by any worker thread; checked by the others to
    # shut down early, and re-raised by run().
    backend_error: BaseException | None = None


def _drain_lc_results(ctx: _RunContext, pose_graph: PoseGraph) -> None:
    """Move all completed loop-closure results from lc_result_queue into the graph."""
    tag = f"[{threading.current_thread().name}]"
    while True:
        try:
            seq_b, seq_a, relative_b_to_a, closure = ctx.lc_result_queue.get_nowait()
        except queue.Empty:
            return
        ctx.loop_closures.append(closure)
        pose_graph.add_between(seq_b, seq_a, relative_b_to_a, loop=True)
        print(f"{tag} LC {closure.candidate.submap_idx_b}"
              f"[f{closure.candidate.frame_idx_b}]"
              f" ↔ {closure.candidate.submap_idx_a}"
              f"[f{closure.candidate.frame_idx_a}]")


def _blocking_put(ctx: _RunContext, q: queue.Queue, item) -> bool:
    """Put item on q, retrying every 0.5 s until space is available.

    Returns False if another thread recorded an error before the item could
    be placed (the pipeline is shutting down).
    """
    while True:
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            if ctx.backend_error is not None:
                return False


def _scaled_translation(transform: np.ndarray, scale: float) -> np.ndarray:
    """Copy of a (4, 4) transform with the translation multiplied by `scale`.

    Rotation is never scaled — only translation carries the metric unit.
    """
    scaled = transform.copy()
    scaled[:3, 3] *= scale
    return scaled


def _adjust_boundary_scale(
    delta_scale: float,
    damping: float,
    clamp: float | None,
) -> float:
    """Damp and/or clamp a raw inter-submap boundary scale ratio.

    Boundary ratios are chained multiplicatively (see module docstring), so
    estimation noise compounds into scale drift over long sequences.  DA3
    depth is nominally metric, so the true ratio should be near 1.0:

      - damping g in [0, 1] applies delta^(1-g): 0 keeps the raw ratio
        (full chaining), 1 forces 1.0 (trust DA3's metric consistency).
      - clamp c (> 1) bounds the applied ratio to [1/c, c], guarding
        against a single bad depth estimate at one boundary.
    """
    adjusted = float(delta_scale) ** (1.0 - damping)
    if clamp is not None and clamp > 1.0:
        adjusted = min(max(adjusted, 1.0 / clamp), clamp)
    return adjusted


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
        # Default to the canonical YAML config.  (Constructing SLAMConfig()
        # directly is not possible — it has required fields.)
        self.config = config or load_slam_config()
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
        self.detector = (
            LoopClosureDetector(cfg.loop_closure, builder=self.builder)
            if cfg.enable_loop_closure else None
        )

        self.semantic_embedder = None
        if cfg.semantic_model:
            # Imported lazily: requires the optional `transformers` package.
            from da3_slam.backend.inference.semantic_embedder import SemanticEmbedder
            self.semantic_embedder = SemanticEmbedder(cfg.semantic_model)

    def run(self, image_paths: list[str]) -> SLAMResult:
        """Offline entry point: run SLAM over a fixed list of image files.

        Thin wrapper over run_stream() — images are read lazily inside the
        frontend thread (preserving I/O / inference overlap) by the disk
        frame source below.
        """
        return self.run_stream(self._disk_frame_source(image_paths))

    @staticmethod
    def _disk_frame_source(image_paths: list[str]) -> Iterator[FrameItem]:
        """Yield (RGB image, seq_idx, path) for each image on disk."""
        for i, path in enumerate(image_paths):
            bgr = cv2.imread(path)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i, path

    def run_stream(self, frame_source: Iterable[FrameItem]) -> SLAMResult:
        """Streaming entry point: run SLAM over an iterable of input frames.

        `frame_source` yields (RGB image, seq_idx, label) tuples in arrival
        order — see FrameItem.  It may block between items (e.g. a live ROS
        stream); the frontend thread consumes it one frame at a time and the
        pipeline runs incrementally, so this works for both offline lists and
        unbounded real-time streams.  The iterator is fully consumed (or an
        end-of-stream is signalled by it simply terminating).
        """
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
                "submap_building": 0.0,
                "graph_building": 0.0,
                "loop_closure": 0.0,
                "optimization": 0.0,
            },
        )

        # Start consumers before producers so they are ready immediately.
        threads = [
            threading.Thread(target=self._processing, args=(ctx,),
                             name="da3-processing", daemon=True),
            threading.Thread(target=self._inference, args=(ctx,),
                             name="da3-inference", daemon=True),
        ]
        if self.detector is not None:
            threads.append(threading.Thread(target=self._loop_closure_worker,
                                            args=(ctx,), name="da3-lc", daemon=True))
        threads.append(threading.Thread(target=self._frontend, args=(frame_source, ctx),
                                        name="da3-frontend", daemon=True))

        wall_start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_elapsed = time.time() - wall_start

        if ctx.backend_error is not None:
            raise ctx.backend_error

        self._print_timing_breakdown(ctx.timings, wall_elapsed)

        return SLAMResult(
            keyframe_poses=_build_keyframe_poses(ctx.submaps, ctx.opt_result),
            submaps=ctx.submaps,
            optimization=ctx.opt_result,
            loop_closures=ctx.loop_closures,
            timings=ctx.timings,
        )

    @staticmethod
    def _print_timing_breakdown(timings: dict[str, float], wall_elapsed: float) -> None:
        col = max(len(k) for k in timings)
        tag = f"[{threading.current_thread().name}]"
        print(f"{tag} Timing breakdown (per-module compute time, threads overlap):")
        for module, seconds in timings.items():
            print(f"  {module:<{col}}  {seconds:6.1f}s")
        print(f"  {'':-<{col + 9}}")
        print(f"  {'compute total':<{col}}  {sum(timings.values()):6.1f}s")
        print(f"  {'wall-clock':<{col}}  {wall_elapsed:6.1f}s")

    # ── frontend thread ───────────────────────────────────────────────────────

    def _frontend(self, frame_source: Iterable[FrameItem], ctx: _RunContext) -> None:
        """Keyframe selection: pulls input frames, batches keyframes into batch_queue.

        Consumes `frame_source` one (image, seq_idx, label) at a time — the
        same code path serves an offline file list and a live stream.  Each
        batch shares its last keyframe with the next batch (the anchor frame)
        — see the module docstring.
        """
        selector = OnlineKeyframeSelector(ctx.config.keyframe)
        keyframe_paths: list[str] = []
        keyframe_images: list[np.ndarray] = []
        keyframe_indices: list[int] = []
        try:
            for image, seq_idx, label in frame_source:
                if ctx.backend_error is not None:
                    break

                t0 = time.time()
                is_keyframe = selector.step(image)
                ctx.timings["keyframe_selection"] += time.time() - t0

                if is_keyframe:
                    keyframe_paths.append(label)
                    keyframe_images.append(image)
                    keyframe_indices.append(seq_idx)

                if len(keyframe_paths) >= ctx.config.submap_size:
                    _blocking_put(ctx, ctx.batch_queue, (
                        list(keyframe_paths),
                        list(keyframe_images),
                        list(keyframe_indices),
                    ))
                    # 1-frame overlap: anchor next submap on the last keyframe
                    keyframe_paths[:] = [keyframe_paths[-1]]
                    keyframe_images[:] = [keyframe_images[-1]]
                    keyframe_indices[:] = [keyframe_indices[-1]]

            # Flush the final partial batch (a single leftover frame is only
            # the anchor copy of the previous batch — nothing new to add).
            if len(keyframe_paths) >= 2 and ctx.backend_error is None:
                _blocking_put(ctx, ctx.batch_queue, (
                    list(keyframe_paths),
                    list(keyframe_images),
                    list(keyframe_indices),
                ))
        except Exception as exc:
            ctx.backend_error = exc
        finally:
            ctx.batch_queue.put(None)  # sentinel — always sent, even on error

    # ── inference thread ──────────────────────────────────────────────────────

    def _inference(self, ctx: _RunContext) -> None:
        """DA3 inference: pops keyframe batches, pushes built Submaps."""
        submap_idx = 0
        try:
            while True:
                item = ctx.batch_queue.get()
                if item is None or ctx.backend_error is not None:
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

    # ── processing thread ─────────────────────────────────────────────────────

    def _processing(self, ctx: _RunContext) -> None:
        """Graph building and incremental optimisation.

        Loop closure detection is dispatched to _loop_closure_worker; its
        results are drained back into the graph between optimisations.
        """
        pose_graph = PoseGraph(ctx.config.noise)

        # Running product of inter-submap scale ratios: converts translations
        # from the current submap's metric unit to submap 0's unit.
        accumulated_scale: float = 1.0
        # Snapshots handed to the LC worker (it runs concurrently and must
        # not observe later mutations).
        submap_scales: dict[int, float] = {}
        submaps_by_idx: dict[int, Submap] = {}

        tag = f"[{threading.current_thread().name}]"
        try:
            while True:
                submap = ctx.submap_queue.get()
                if submap is None:
                    break

                t0 = time.time()
                prev_submap = ctx.submaps[-1] if ctx.submaps else None
                if prev_submap is None:
                    self._add_first_submap_to_graph(pose_graph, submap)
                else:
                    raw_delta_scale = _estimate_boundary_scale(prev_submap, submap)
                    delta_scale = _adjust_boundary_scale(
                        raw_delta_scale,
                        ctx.config.boundary_scale_damping,
                        ctx.config.boundary_scale_clamp,
                    )
                    accumulated_scale *= delta_scale
                    self._add_submap_to_graph(pose_graph, submap, accumulated_scale)
                    print(f"{tag} Submap {submap.idx}: scale={accumulated_scale:.4f} "
                          f"(Δ raw={raw_delta_scale:.4f} applied={delta_scale:.4f})")
                self._add_consecutive_frame_factors(pose_graph, submap, accumulated_scale)

                ctx.submaps.append(submap)
                submap_scales[submap.idx] = accumulated_scale
                submaps_by_idx[submap.idx] = submap
                ctx.timings["graph_building"] += time.time() - t0

                if self.detector is not None:
                    ctx.lc_queue.put((submap, dict(submaps_by_idx), dict(submap_scales)))

                _drain_lc_results(ctx, pose_graph)

                t0 = time.time()
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

            # Wait for the LC worker to finish, then incorporate its last results.
            if self.detector is not None:
                ctx.lc_queue.put(None)  # sentinel
                ctx.lc_done.wait()
            _drain_lc_results(ctx, pose_graph)

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

    @staticmethod
    def _add_first_submap_to_graph(pose_graph: PoseGraph, submap: Submap) -> None:
        """Insert the first submap's frames at their DA3 poses and pin frame 0."""
        for frame in submap.frames:
            pose_graph.add_frame(frame.seq_idx, frame.cam_to_world.astype(np.float64))
        pose_graph.add_prior(submap.frames[0].seq_idx)

    @staticmethod
    def _add_submap_to_graph(
        pose_graph: PoseGraph,
        submap: Submap,
        accumulated_scale: float,
    ) -> None:
        """Insert a later submap's frames, placed via the shared anchor frame.

        The anchor (frames[0]) already exists in the graph with an optimised
        global pose; each new frame's initial global pose is that anchor pose
        composed with DA3's local anchor-to-frame transform (translation
        rescaled to the global metric unit).
        """
        anchor_global_c2w = pose_graph.get_pose(submap.frames[0].seq_idx)
        anchor_local_w2c = submap.frames[0].extrinsic.astype(np.float64)
        for frame in submap.frames[1:]:
            local_c2w = frame.cam_to_world.astype(np.float64)
            relative = anchor_local_w2c @ local_c2w
            pose_graph.add_frame(
                frame.seq_idx,
                anchor_global_c2w @ _scaled_translation(relative, accumulated_scale),
            )

    @staticmethod
    def _add_consecutive_frame_factors(
        pose_graph: PoseGraph,
        submap: Submap,
        accumulated_scale: float,
    ) -> None:
        """Add a between-factor for each consecutive frame pair in the submap."""
        for previous, current in zip(submap.frames, submap.frames[1:]):
            relative = (
                previous.extrinsic.astype(np.float64)
                @ current.cam_to_world.astype(np.float64)
            )
            pose_graph.add_between(
                previous.seq_idx,
                current.seq_idx,
                _scaled_translation(relative, accumulated_scale),
            )

    # ── loop closure thread ───────────────────────────────────────────────────

    def _loop_closure_worker(self, ctx: _RunContext) -> None:
        """Detects and verifies loop closures; posts scaled factors for _processing.

        Receives (submap, submaps_by_idx, submap_scales) snapshots so it can
        run concurrently with graph building.
        """
        try:
            while True:
                item = ctx.lc_queue.get()
                if item is None or ctx.backend_error is not None:
                    break
                submap, submaps_by_idx, submap_scales = item

                t0 = time.time()
                closures = self.detector.process(submap)
                ctx.timings["loop_closure"] += time.time() - t0

                for closure in closures:
                    candidate = closure.candidate
                    frame_b = (submaps_by_idx[candidate.submap_idx_b]
                               .frames[candidate.frame_idx_b])
                    frame_a = (submaps_by_idx[candidate.submap_idx_a]
                               .frames[candidate.frame_idx_a])

                    # The LC re-inference has its own arbitrary metric scale.
                    # Convert its translation to the global unit in two steps:
                    # LC unit → query submap unit (depth ratio at the shared
                    # query frame), then → global unit (the query submap's
                    # accumulated scale).
                    lc_to_query_scale = _estimate_depth_scale(
                        frame_b.depth,
                        closure.lc_submap.frames[0].depth,
                    )
                    global_scale = (
                        lc_to_query_scale * submap_scales[candidate.submap_idx_b]
                    )
                    relative_b_to_a = _scaled_translation(
                        closure.relative_b_to_a, global_scale
                    )

                    ctx.lc_result_queue.put(
                        (frame_b.seq_idx, frame_a.seq_idx, relative_b_to_a, closure)
                    )

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
