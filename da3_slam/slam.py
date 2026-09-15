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
                                  (loop_closure_queue / loop_closure_result_queue)
                                                  ⇅
                                          _loop_closure_worker

  Errors in any worker are recorded in _RunContext.backend_error; the other
  threads poll it and shut down, and run() re-raises it at the end.

Two structural invariants that are easy to break (mirrors VGGT-SLAM):

  1. Anchor-frame submap bridging.  Consecutive submaps share the last
     `submap_overlap` physical keyframes of submap N as the first frames of
     submap N+1 (same seq_idx).  Those shared nodes are the *only* connection
     between submaps in the pose graph — there is no explicit inter-submap
     factor.  If the batching in _frontend changes, the overlap must be
     preserved or the graph disconnects.  With overlap >= 2 the shared frame
     pair is measured by both DA3 batches: the duplicate between-factor adds
     boundary redundancy and _check_boundary_consistency flags odometry
     breaks (a single bad anchor pose can otherwise displace a whole submap).

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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from da3_slam.config import SLAMConfig, load_slam_config
from da3_slam.frontend.keyframe_selector import (
    OnlineKeyframeSelector,
    ReplayKeyframeSelector,
    SegmentKeyframeSelector,
    load_keyframe_list,
    save_keyframe_list,
)
from da3_slam.backend.inference.depth_estimator import DepthEstimator
from da3_slam.backend.inference.submap import Submap, SubmapBuilder, transform_points
from da3_slam.backend.processing.factor_graph import PoseGraph, OptimizationResult
from da3_slam.backend.processing.loop_closure import LoopClosureDetector, LoopClosure

__all__ = ["SLAMConfig", "SLAMResult", "SLAMUpdate", "DA3SLAM", "FrameItem"]

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

    # Backbone (DA3 forward) instrumentation for the whole run, read from the
    # DepthEstimator: cumulative forward wall-clock (s), number of forward
    # calls, and the peak GPU memory of a single forward (bytes).  Backbone-
    # only latency + peak memory are two of the sweep's headline axes.
    backbone_seconds: float = 0.0
    backbone_calls: int = 0
    peak_gpu_mem_bytes: int = 0

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
            if submap.is_loop_closure_submap:
                continue
            for frame in submap.frames:
                # The anchor frame appears in two submaps; keep it once.
                if frame.seq_idx in seen_seq_idx:
                    continue
                seen_seq_idx.add(frame.seq_idx)
                if len(frame.points_cam) == 0:
                    continue
                points_to_world = self.optimization.point_transform(frame.seq_idx)
                all_points.append(transform_points(frame.points_cam, points_to_world))
                all_colors.append(frame.colors)

        if not all_points:
            raise ValueError(
                "No point clouds stored — the run was built with "
                "build_pointclouds=False (lean mode); PLY export needs a run "
                "with point clouds enabled."
            )
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


# ── live update ─────────────────────────────────────────────────────────────

@dataclass
class SLAMUpdate:
    """Incremental snapshot emitted after each submap is optimised.

    Passed to the optional ``on_update`` callback of :meth:`DA3SLAM.run_stream`
    so a live viewer can render the trajectory and map as they grow.  The poses
    are the *current* best global cam-to-world estimates — they may still shift
    on later optimisation (e.g. when a loop closure is incorporated).

    The callback runs inside the processing thread; keep it cheap (a slow
    callback backpressures the whole pipeline).  Exceptions raised by the
    callback are caught and logged, never propagated into the SLAM run.
    """
    # Index of the submap that was just optimised.
    submap_idx: int

    # seq_idx → (4, 4) current global cam-to-world pose for every keyframe.
    keyframe_poses: dict[int, np.ndarray]

    # (M, 3) float32 world-space points contributed by this submap (anchor
    # frame excluded for submaps after the first, so points are not duplicated).
    new_points_world: np.ndarray

    # (M, 3) uint8 colours aligned with new_points_world.
    new_colors: np.ndarray

    # Running counts (loop-closure re-inference submaps are excluded).
    n_submaps: int
    n_loop_closures: int

    # Per-frame camera-space points for the new submap: (seq_idx, points_cam
    # (M, 3) float32, colors (M, 3) uint8) per contributing frame.  Lets the
    # viewer cache and *re-project* earlier submaps when a loop closure or
    # later optimisation shifts their poses (keyframe_poses always carries the
    # current estimate for every keyframe).
    frame_points_cam: list[tuple[int, np.ndarray, np.ndarray]] = field(
        default_factory=list)

    # seq_idx → similarity scale (Sim(3); 1.0 under SL(4)).  keyframe_poses are
    # rigid, so re-projecting points must scale them: world = s·R·p + t.
    keyframe_scales: dict[int, float] = field(default_factory=dict)


# ── run context ───────────────────────────────────────────────────────────────

@dataclass
class _RunContext:
    """Shared mutable state passed between the four pipeline threads."""
    config: SLAMConfig
    batch_queue: queue.Queue       # frontend   → inference  (paths, images, indices)
    submap_queue: queue.Queue      # inference  → processing (Submap)
    # processing → loop-closure worker (submap + dict snapshots)
    loop_closure_queue: queue.Queue
    # loop-closure worker → processing (seq_b, seq_a, relative, closure)
    loop_closure_result_queue: queue.Queue
    loop_closure_done: threading.Event
    submaps: list[Submap]
    loop_closures: list[LoopClosure]
    timings: dict[str, float]
    optimization_result: OptimizationResult | None = None

    # Optional live-update callback (fired per submap from the processing
    # thread); None disables incremental emission.
    on_update: Callable[[SLAMUpdate], None] | None = None

    # Optional loop-closure callback (processing thread): fired twice per
    # drain that inserted loop factors, with the keyframe poses (seq_idx →
    # 4x4 cam-to-world), the inserted (query_seq, detected_seq) pairs, and a
    # phase tag.  phase="pre": factors are in the graph but the optimisation
    # has not run — the endpoint poses still carry the accumulated drift the
    # closure is about to correct, so a chord drawn between them visualises
    # the detected loop.  phase="post": right after that optimisation, same
    # pairs — showing what the correction did (converged endpoints coincide).
    # Same error contract as on_update.
    on_loop_closure: (Callable[[dict[int, np.ndarray],
                                list[tuple[int, int]], str], None] | None) = None

    # First exception raised by any worker thread; checked by the others to
    # shut down early, and re-raised by run().
    backend_error: BaseException | None = None


def _normalized_rotation(transform: np.ndarray) -> np.ndarray | None:
    """Det-normalised 3×3 block of a transform (SL(4)-safe rotation
    approximation), or None if degenerate."""
    block = transform[:3, :3].astype(np.float64)
    det = np.linalg.det(block)
    if not np.isfinite(det) or det <= 0:
        return None
    return block / det ** (1.0 / 3.0)


def _rotation_angle_deg(transform: np.ndarray) -> float:
    """Rotation angle (degrees) of a transform's 3×3 block.  NaN if degenerate."""
    rotation = _normalized_rotation(transform)
    if rotation is None:
        return float("nan")
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ── loop-closure geometric gate + corroboration pool ─────────────────────────
#
# A closure whose measurement disagrees wildly with the graph's current
# prediction is *held*, not dropped: the gate compares against the graph, and
# after an odometry break (a bad DA3 pose at a submap boundary) the graph
# itself is wrong by exactly the disagreement the gate measures.  A single
# aliased match and a single break-healing match look identical — but when
# two independent held closures imply the *same* correction, the measurements
# corroborate each other and the group is inserted (this is what merges a
# duplicated map back together).  Held closures are re-evaluated against the
# updated graph on every drain; one that stays held (uncorroborated but also
# uncontradicted) for _HELD_ACCEPT_AFTER_DRAINS drains is accepted on its own
# — it already passed the descriptor + confidence gates, and the Huber loop
# noise cushions the graph if it turns out to be aliased after all.  The pool
# expires the oldest entries when it overflows.

_PENDING_POOL_MAX = 8           # held closures kept for corroboration
_HELD_ACCEPT_AFTER_DRAINS = 5   # drains (≈ submaps) before a lone held closure is accepted
_CONSISTENCY_TRANS_TOL = 1.0    # min tolerance between implied corrections (global units)
_CONSISTENCY_TRANS_FRAC = 0.25  # ...or this fraction of the correction magnitude
_CONSISTENCY_ROT_TOL_DEG = 30.0


@dataclass
class _HeldLoop:
    """A gate-failing loop closure awaiting corroboration."""
    seq_b: int
    seq_a: int
    relative_b_to_a: np.ndarray
    closure: LoopClosure
    announced: bool = False   # "held" message printed once, not every drain
    drains_held: int = 0      # consecutive drains spent in the pool


def _loop_disagreement(
    pose_graph: PoseGraph,
    seq_b: int,
    seq_a: int,
    measured: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Disagreement between a loop measurement and the graph's prediction.

    Returns (d_world, translation_error, rotation_error_deg): the translation
    disagreement rotated into world axes — so disagreements of closures at
    different query frames are comparable — plus its norm and the rotation
    disagreement angle.
    """
    pose_b = pose_graph.get_pose(seq_b)
    predicted = np.linalg.inv(pose_b) @ pose_graph.get_pose(seq_a)
    d_local = measured[:3, 3] - predicted[:3, 3]
    rotation_wb = _normalized_rotation(pose_b)
    d_world = rotation_wb @ d_local if rotation_wb is not None else d_local
    rotation_error = _rotation_angle_deg(np.linalg.inv(predicted) @ measured)
    return d_world, float(np.linalg.norm(d_local)), rotation_error


def _corrections_consistent(
    d1: np.ndarray, t1: float, r1: float,
    d2: np.ndarray, t2: float, r2: float,
) -> bool:
    """True if two held closures imply the same graph correction."""
    trans_tol = max(_CONSISTENCY_TRANS_TOL, _CONSISTENCY_TRANS_FRAC * max(t1, t2))
    if float(np.linalg.norm(d1 - d2)) > trans_tol:
        return False
    if np.isfinite(r1) and np.isfinite(r2) and abs(r1 - r2) > _CONSISTENCY_ROT_TOL_DEG:
        return False
    return True


def _insert_loop_factor(
    ctx: _RunContext,
    pose_graph: PoseGraph,
    inserted: list[tuple[int, int]],
    held: _HeldLoop,
    tag: str,
    note: str = "",
) -> None:
    """Insert one accepted loop closure into the graph and record it."""
    ctx.loop_closures.append(held.closure)
    pose_graph.add_between(held.seq_b, held.seq_a, held.relative_b_to_a, loop=True)
    inserted.append((held.seq_b, held.seq_a))
    candidate = held.closure.candidate
    print(f"{tag} Loop closure {candidate.submap_idx_b}"
          f"[f{candidate.frame_idx_b}]"
          f" ↔ {candidate.submap_idx_a}"
          f"[f{candidate.frame_idx_a}]{note}")


def _drain_loop_closure_results(
    ctx: _RunContext,
    pose_graph: PoseGraph,
    pending: list[_HeldLoop],
) -> list[tuple[int, int]]:
    """Move completed loop-closure results into the graph.

    Each result passes the geometric gate (measurement vs. current graph
    prediction; thresholds from LoopClosureConfig, each disableable with
    None).  Gate failures are held in `pending` and re-evaluated on every
    drain; a mutually consistent group of held closures is inserted (see the
    corroboration-pool comment above).

    Returns the (query_seq, detected_seq) pairs of the factors inserted by
    this drain — the graph has not yet been re-optimised at that point (see
    _RunContext.on_loop_closure).
    """
    tag = f"[{threading.current_thread().name}]"
    inserted: list[tuple[int, int]] = []
    trusted_items: list[_HeldLoop] = []
    while True:
        try:
            seq_b, seq_a, relative_b_to_a, closure, trusted = (
                ctx.loop_closure_result_queue.get_nowait())
        except queue.Empty:
            break
        held = _HeldLoop(seq_b, seq_a, relative_b_to_a, closure)
        (trusted_items if trusted else pending).append(held)
    if not pending and not trusted_items:
        return inserted

    config = ctx.config.loop_closure
    max_rotation = config.max_rotation_error_deg
    max_translation = config.max_translation_error

    # Boundary repairs bypass the gate: the matched frames are known-adjacent
    # (no aliasing risk) and the graph is known-wrong exactly there.
    for held in trusted_items:
        _insert_loop_factor(ctx, pose_graph, inserted, held, tag,
                            note="  (boundary repair)")

    # Evaluate everything (new + previously held) against the current graph.
    held_evaluated: list[tuple[_HeldLoop, np.ndarray, float, float]] = []
    for held in pending:
        try:
            d_world, translation_error, rotation_error = _loop_disagreement(
                pose_graph, held.seq_b, held.seq_a, held.relative_b_to_a)
        except Exception as exc:  # missing node / singular matrix — do not gate
            print(f"{tag} loop geometry gate skipped ({exc!r})")
            _insert_loop_factor(ctx, pose_graph, inserted, held, tag)
            continue
        rotation_bad = (max_rotation is not None and np.isfinite(rotation_error)
                        and rotation_error > max_rotation)
        translation_bad = (max_translation is not None
                           and translation_error > max_translation)
        if rotation_bad or translation_bad:
            held_evaluated.append((held, d_world, translation_error, rotation_error))
        else:
            _insert_loop_factor(ctx, pose_graph, inserted, held, tag)
    pending.clear()

    # Corroboration among the gate failures.
    accepted: set[int] = set()
    for i in range(len(held_evaluated)):
        for j in range(i + 1, len(held_evaluated)):
            _, d1, t1, r1 = held_evaluated[i]
            _, d2, t2, r2 = held_evaluated[j]
            if _corrections_consistent(d1, t1, r1, d2, t2, r2):
                accepted.update((i, j))
    for i in sorted(accepted):
        held, _, translation_error, rotation_error = held_evaluated[i]
        _insert_loop_factor(
            ctx, pose_graph, inserted, held, tag,
            note=f"  (corroborated — graph was off by "
                 f"{translation_error:.2f} / {rotation_error:.0f}°)")

    for i, (held, _, translation_error, rotation_error) in enumerate(held_evaluated):
        if i in accepted:
            continue
        held.drains_held += 1
        if held.drains_held >= _HELD_ACCEPT_AFTER_DRAINS:
            _insert_loop_factor(
                ctx, pose_graph, inserted, held, tag,
                note=f"  (accepted after {held.drains_held} submaps "
                     f"held uncontradicted — graph off by "
                     f"{translation_error:.2f} / {rotation_error:.0f}°)")
            continue
        if not held.announced:
            held.announced = True
            print(f"{tag} Loop closure HELD (geometric: trans {translation_error:.2f}, "
                  f"rot {rotation_error:.0f}° vs graph) — awaiting corroboration")
        pending.append(held)
    del pending[:-_PENDING_POOL_MAX]  # cap the pool, dropping the oldest
    return inserted


def _blocking_put(ctx: _RunContext, target_queue: queue.Queue, item) -> bool:
    """Put item on target_queue, retrying every 0.5 s until space is available.

    Returns False if another thread recorded an error before the item could
    be placed (the pipeline is shutting down).
    """
    while True:
        try:
            target_queue.put(item, timeout=0.5)
            return True
        except queue.Full:
            if ctx.backend_error is not None:
                return False


def _effective_overlap(config: SLAMConfig) -> int:
    """Anchor overlap clamped to [1, submap_size - 1]."""
    return max(1, min(int(config.submap_overlap), config.submap_size - 1))


# A keyframe batch handed from the frontend to the inference thread:
# (labels, images, seq_indices), index-aligned.
_KeyframeBatch = tuple[list[str], list[np.ndarray], list[int]]


class _KeyframeBatcher:
    """Groups keyframes into submap batches with anchor-frame overlap.

    Each emitted batch keeps its last `overlap` keyframes as the start of the
    next batch — the shared anchors that bridge consecutive submaps in the
    pose graph (module docstring, invariant 1).
    """

    def __init__(self, submap_size: int, overlap: int,
                 warmup_size: int = 0, warmup_submaps: int = 0,
                 flow_budget: float = 0.0, keyframe_config=None):
        """Batch keyframes into submaps, optionally with a WARMUP RAMP.

        With warmup_size == 0 the behaviour is what it has always been: every
        submap holds `submap_size` keyframes.

        With a warmup, the first `warmup_submaps` submaps hold `warmup_size`
        keyframes and the rest hold `submap_size`.  This exists because a SHORT
        sequence at a large submap_size can yield a single submap, which
        silently disables the whole SLAM layer: no boundary to chain, a one-node
        SL(4) graph with nothing to optimise, and loop closure structurally
        impossible (min_submaps_apart >= 1 cannot be met).  Measured on
        7-Scenes/stairs (500 frames -> 32 keyframes -> 1 submap at size 32):
        ATE 0.1149 with scale 1.140, versus 0.0305 and scale 1.009 at size 16,
        where the same sequence yields 3 submaps — a 3.8x difference produced
        entirely by whether the pose graph exists.

        The ramp is CAUSAL: it depends only on how many submaps have already
        been emitted, never on the total sequence length, so it works online
        where the length is unknown.  Submaps of differing sizes need no special
        handling downstream — graph nodes are submaps and factors are relative
        poses at shared anchors, so a [16, 16, 32, 32, ...] sequence is a
        perfectly ordinary graph and no re-inference is required.

        Trade-off to be aware of: DA3's cross-view attention has less context in
        a smaller batch, so warmup submaps have somewhat weaker internal
        geometry (at size 8 stairs scored 0.0997, worse than size 16's 0.0305),
        and they sit at the START of the trajectory where errors propagate
        furthest.  Keep warmup_size at the smallest value that still gives good
        geometry rather than the smallest that gives many submaps.
        """
        self._submap_size = submap_size
        self._overlap = overlap
        # FLOW BUDGET: close a submap once the view has moved far enough,
        # instead of only after a fixed KEYFRAME COUNT.
        #
        # A submap is one DA3 batch, and DA3 estimates its geometry jointly by
        # attending BETWEEN views -- which requires the views to still see
        # overlapping scene.  Counting keyframes does not measure that.  In
        # segment mode keyframes are picked at a fixed FRAME stride, so a fixed
        # count spans whatever distance the camera happened to travel:
        # measured, 2.8 m per submap on TUM fr1/teddy and 3.3 m on Replica
        # room2, against 51 m on UAS fyllingsdalen and 65 m on UAS hornbill.
        # At 65 m in a tunnel the first and last frames of a batch share no
        # visible surface, so cross-view attention has nothing to attend to --
        # and those two sequences are exactly the ones that regressed 2.2-2.4x
        # when submap_size went 16 -> 32, while campus_fog (8 m/submap) barely
        # moved.
        #
        # Accumulated optical flow is a scale-free proxy for view change: it
        # needs no metric estimate and no intrinsics, so ONE budget serves
        # indoor and aerial alike.  submap_size remains as a hard cap.
        # 0 disables it and the behaviour is exactly as before.
        self._flow_budget = float(flow_budget or 0.0)
        self._kf_config = keyframe_config
        self._flow_accum = 0.0
        self._last_image = None
        self._warmup_size = int(warmup_size)
        self._warmup_submaps = int(warmup_submaps)
        self._emitted = 0
        self._labels: list[str] = []
        self._images: list[np.ndarray] = []
        self._indices: list[int] = []

    def _target_size(self) -> int:
        """Keyframes for the submap currently being filled.

        GEOMETRIC RAMP: start at `warmup_size` and DOUBLE every
        `warmup_submaps` submaps, capped at `submap_size`.  With
        warmup_size=16, warmup_submaps=2 and submap_size=32 that is
        16, 16, 32, 32, 32, ... — the first submaps are small so a short
        sequence still gets a pose graph, and the size climbs to the measured
        optimum for everything after.

        THE CAP IS NOT ARBITRARY.  A sweep of fixed sizes {8, 16, 32, 64} on
        TUM fr1 (keyframes frozen per sequence) gave mean Sim(3) ATE
        0.0324 / 0.0295 / 0.0267 / 0.0270: the curve is U-shaped with its
        minimum at 32, and 64 already ties while degrading fr1/room and
        fr1/teddy.  Ramping past the cap would also shrink the submap COUNT —
        loop closures already fall 3.8 -> 1.7 per sequence going from 16 to 32,
        since fewer submaps means fewer eligible retrieval pairs — and would
        recreate at the long end exactly the failure this ramp fixes at the
        short end, where one submap disables chaining, the graph and loop
        closure together.  DA3's cross-view attention is also quadratic in
        batch frames, so very large submaps are expensive and outside the
        regime the backbone was trained on.
        """
        if self._warmup_size <= 0:
            return self._submap_size
        step = max(1, self._warmup_submaps)
        size = self._warmup_size * (2 ** (self._emitted // step))
        size = min(size, self._submap_size)
        return max(size, self._overlap + 1)

    def add(self, label: str, image: np.ndarray, seq_idx: int) -> _KeyframeBatch | None:
        """Append a keyframe; return a full batch when one is ready."""
        if self._flow_budget > 0.0 and self._last_image is not None:
            try:
                from da3_slam.frontend.keyframe_selector import keyframe_flow
                self._flow_accum += keyframe_flow(
                    self._last_image, image, self._kf_config)[2]
            except Exception:
                pass          # never fail a run over the motion signal
        self._last_image = image

        self._labels.append(label)
        self._images.append(image)
        self._indices.append(seq_idx)
        over_budget = (self._flow_budget > 0.0
                       and self._flow_accum >= self._flow_budget
                       and len(self._labels) > self._overlap + 1)
        if len(self._labels) < self._target_size() and not over_budget:
            return None
        self._emitted += 1
        self._flow_accum = 0.0
        batch = (list(self._labels), list(self._images), list(self._indices))
        # Anchor the next submap on the last `overlap` keyframes.
        del self._labels[:-self._overlap]
        del self._images[:-self._overlap]
        del self._indices[:-self._overlap]
        return batch

    def tail(self) -> _KeyframeBatch | None:
        """The final partial batch, or None when only the anchor copies of
        the previous batch remain (they carry nothing new)."""
        if len(self._labels) <= self._overlap:
            return None
        return (list(self._labels), list(self._images), list(self._indices))


def _scaled_translation(transform: np.ndarray, scale: float) -> np.ndarray:
    """Copy of a (4, 4) transform with the translation multiplied by `scale`.

    Rotation is never scaled — only translation carries the metric unit.
    """
    scaled = transform.copy()
    scaled[:3, 3] *= scale
    return scaled


def _rolloff_damping(boundary_idx: int, config) -> float:
    """Effective damping g for the boundary entering submap `boundary_idx`.

    With boundary_scale_rolloff_tau <= 0 this returns the fixed
    boundary_scale_damping and nothing changes.

    Otherwise the per-boundary WEIGHT w = (1 - g) follows a smooth rolloff

        w_j = 1 / (1 + (j / tau)^p)

    so g_j = 1 - w_j.  Motivation: boundary ratios are chained
    multiplicatively, so with a constant weight the accumulated scale variance
    is Var(log S_k) = sigma^2 * sum_j w_j^2 = sigma^2 * k -- it grows without
    bound in the number of boundaries.  Any w_j with sum w_j^2 < infinity keeps
    it bounded; this form converges for p > 1/2 (sum = 7.56 at tau=10, p=3).

    WHY NOT A SINGLE EXPONENTIAL: e^(-j/tau) is also summable, but one
    parameter controls both where the transition sits and how gradual it is, so
    it cannot be flat early and steep later.  Measured against the damping
    sweep, tau=30 leaves w=0.46 at j=23 (UAS median, which wants w~0) while
    tau=10 already drops to w=0.67 at j=4 (TUM median, where damping costs ~5%).
    The rolloff separates the two: tau sets the transition, p its sharpness, so
    tau=10 p=3 gives w=0.94 at j=4 and w=0.08 at j=23.

    This is causal -- it depends only on how many boundaries have been chained
    so far, never on total sequence length -- so it works online.
    """
    tau = float(getattr(config, "boundary_scale_rolloff_tau", 0.0) or 0.0)
    if tau <= 0.0:
        return float(config.boundary_scale_damping)
    p = float(getattr(config, "boundary_scale_rolloff_p", 3.0) or 3.0)
    j = max(int(boundary_idx), 0)
    w = 1.0 / (1.0 + (j / tau) ** p)
    # Never trust a boundary MORE than the configured floor allows.
    w = min(w, 1.0 - float(config.boundary_scale_damping))
    return 1.0 - w


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


# A boundary repair enters the graph as a *trusted* factor (bypasses the
# geometric gate), so its measurement gets one independent sanity check: the
# relative rotation of the repair pair composed through each batch's own
# chain.  The two chains agree on that rotation to within the boundary
# disagreement (~tens of degrees even at a break); a repair re-inference
# deviating far beyond that is a garbage wide-baseline estimate (observed:
# conf ≈ 0.21-0.26 repairs carrying ~120° rotation errors that corrupted the
# whole downstream map orientation).
_REPAIR_MAX_ROTATION_DEV_DEG = 45.0


def _repair_rotation_deviation(
    prev_frames: list,
    frame_idx_a: int,
    curr_frames: list,
    frame_idx_b: int,
    overlap: int,
    measured_b_to_a: np.ndarray,
) -> float:
    """Rotation angle (deg) between a boundary-repair measurement and the
    same relative pose composed through the two batches' chains via the
    shared anchor frame (prev.frames[-1] == curr.frames[overlap-1]).
    Rotation only — the two chains' translation units differ at a break."""
    expected_b_to_a = (
        curr_frames[frame_idx_b].extrinsic.astype(np.float64)
        @ curr_frames[overlap - 1].cam_to_world.astype(np.float64)
        @ prev_frames[-1].extrinsic.astype(np.float64)
        @ prev_frames[frame_idx_a].cam_to_world.astype(np.float64)
    )
    return _rotation_angle_deg(np.linalg.inv(expected_b_to_a) @ measured_b_to_a)


def _boundary_delta_scale(
    prev_submap: Submap,
    curr_submap: Submap,
    overlap: int,
    config: SLAMConfig,
    tag: str,
) -> tuple[float, float]:
    """(raw, applied) boundary scale ratio under the configured policy.

    boundary_scale_deadband > 0 splits the two regimes DA3 actually exhibits:
    ratios inside the band are noise around metric consistency and are forced
    to 1.0 (the sweep-validated behaviour); ratios outside the band are a
    genuine per-batch metric scale *break* (live D455 runs showed 2-6x
    translation-unit jumps) and are applied in full — damping is ignored,
    only the clamp still applies.  With deadband == 0 the legacy
    damping/clamp behaviour is used unchanged.
    """
    deadband = config.boundary_scale_deadband
    if (deadband <= 0 and config.boundary_scale_damping >= 1.0
            and float(getattr(config, "boundary_scale_rolloff_tau", 0.0) or 0.0) <= 0.0):
        # Nothing would use the ratio — skip the median depth-ratio estimate.
        return 1.0, 1.0
    raw = _estimate_boundary_scale(prev_submap, curr_submap, overlap)
    if deadband > 0:
        if not np.isfinite(raw) or raw <= 0:
            return raw, 1.0
        if max(raw, 1.0 / raw) - 1.0 <= deadband:
            return raw, 1.0
        applied = _adjust_boundary_scale(raw, 0.0, config.boundary_scale_clamp)
        print(f"{tag} WARNING: metric scale break at boundary "
              f"{prev_submap.idx}→{curr_submap.idx} (depth ratio {raw:.3f}) — "
              f"applying full correction {applied:.3f}")
        return raw, applied
    return raw, _adjust_boundary_scale(
        raw, _rolloff_damping(curr_submap.idx, config),
        config.boundary_scale_clamp)


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
        config = self.config

        self.estimator = DepthEstimator(
            model_id=config.depth_model,
            process_resolution=config.depth_model_resolution,
            use_ray_pose=config.use_ray_pose,
            token_merging=config.token_merging,
            backbone_dtype=config.backbone_dtype,
        )
        self.builder = SubmapBuilder(
            self.estimator,
            confidence_percentile=config.confidence_percentile,
            build_pointclouds=config.build_pointclouds,
        )
        self.detector = (
            LoopClosureDetector(config.loop_closure, builder=self.builder)
            if config.enable_loop_closure else None
        )

        self.semantic_embedder = None
        if config.semantic_model:
            # Imported lazily: requires the optional `transformers` package.
            from da3_slam.backend.inference.semantic_embedder import SemanticEmbedder
            self.semantic_embedder = SemanticEmbedder(config.semantic_model)

    def run(
        self,
        image_paths: list[str],
        on_update: Callable[[SLAMUpdate], None] | None = None,
        on_loop_closure: (Callable[[dict[int, np.ndarray],
                                    list[tuple[int, int]], str], None]
                          | None) = None,
    ) -> SLAMResult:
        """Offline entry point: run SLAM over a fixed list of image files.

        Thin wrapper over run_stream() — images are read lazily inside the
        frontend thread (preserving I/O / inference overlap) by the disk
        frame source below.  `on_update` / `on_loop_closure` are forwarded to
        run_stream() so a live viewer can watch offline replays too (see
        SLAMUpdate and _RunContext.on_loop_closure).
        """
        return self.run_stream(self._disk_frame_source(image_paths),
                               on_update=on_update,
                               on_loop_closure=on_loop_closure)

    @staticmethod
    def _disk_frame_source(image_paths: list[str]) -> Iterator[FrameItem]:
        """Yield (RGB image, seq_idx, path) for each image on disk."""
        for i, path in enumerate(image_paths):
            bgr = cv2.imread(path)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i, path

    def run_stream(
        self,
        frame_source: Iterable[FrameItem],
        on_update: Callable[[SLAMUpdate], None] | None = None,
        on_loop_closure: (Callable[[dict[int, np.ndarray],
                                    list[tuple[int, int]], str], None]
                          | None) = None,
    ) -> SLAMResult:
        """Streaming entry point: run SLAM over an iterable of input frames.

        `frame_source` yields (RGB image, seq_idx, label) tuples in arrival
        order — see FrameItem.  It may block between items (e.g. a live ROS
        or RealSense stream); the frontend thread consumes it one frame at a
        time and the pipeline runs incrementally, so this works for both
        offline lists and unbounded real-time streams.  The iterator is fully
        consumed (or an end-of-stream is signalled by it simply terminating).

        `on_update`, if given, is called once per submap (from the processing
        thread) with a SLAMUpdate snapshot of the growing trajectory and map —
        used to drive a live viewer.  See SLAMUpdate for the threading and
        error-handling contract.
        """
        # Zero the backbone timing / peak-memory counters for this run (the
        # model — and thus the estimator — is reused across runs).
        self.estimator.reset_stats()

        loop_closure_done = threading.Event()
        if self.detector is None:
            loop_closure_done.set()  # no loop-closure thread — event is immediately done

        ctx = _RunContext(
            config=self.config,
            batch_queue=queue.Queue(maxsize=2),
            submap_queue=queue.Queue(maxsize=2),
            loop_closure_queue=queue.Queue(),
            loop_closure_result_queue=queue.Queue(),
            loop_closure_done=loop_closure_done,
            submaps=[],
            loop_closures=[],
            timings={
                "keyframe_selection": 0.0,
                "submap_building": 0.0,
                "graph_building": 0.0,
                "loop_closure": 0.0,
                "optimization": 0.0,
            },
            on_update=on_update,
            on_loop_closure=on_loop_closure,
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
                                            args=(ctx,), name="da3-loop-closure", daemon=True))
        threads.append(threading.Thread(target=self._frontend, args=(frame_source, ctx),
                                        name="da3-frontend", daemon=True))

        wall_start = time.time()
        for thread in threads:
            thread.start()
        # Join with periodic wakeups rather than a bare join(): an infinite
        # join() parks the main thread in an uninterruptible lock acquire, so
        # signals are never delivered — a Ctrl-C stop hooked to the frame
        # source (see run_realsense.py) would be ignored.  Waking every 0.2 s
        # keeps the main thread able to run its signal handler.
        pending = list(threads)
        while pending:
            for thread in pending:
                thread.join(timeout=0.2)
            pending = [t for t in pending if t.is_alive()]
        wall_elapsed = time.time() - wall_start

        if ctx.backend_error is not None:
            raise ctx.backend_error

        self._print_timing_breakdown(
            ctx.timings, wall_elapsed,
            backbone_seconds=self.estimator.backbone_seconds,
            backbone_calls=self.estimator.n_infer_calls,
            peak_gpu_mem_bytes=self.estimator.peak_memory_bytes,
        )

        return SLAMResult(
            keyframe_poses=_build_keyframe_poses(
                ctx.submaps, ctx.optimization_result.pose),
            submaps=ctx.submaps,
            optimization=ctx.optimization_result,
            loop_closures=ctx.loop_closures,
            timings=ctx.timings,
            backbone_seconds=self.estimator.backbone_seconds,
            backbone_calls=self.estimator.n_infer_calls,
            peak_gpu_mem_bytes=self.estimator.peak_memory_bytes,
        )

    @staticmethod
    def _print_timing_breakdown(
        timings: dict[str, float],
        wall_elapsed: float,
        backbone_seconds: float = 0.0,
        backbone_calls: int = 0,
        peak_gpu_mem_bytes: int = 0,
    ) -> None:
        col = max(len(k) for k in timings)
        tag = f"[{threading.current_thread().name}]"
        print(f"{tag} Timing breakdown (per-module compute time, threads overlap):")
        for module, seconds in timings.items():
            print(f"  {module:<{col}}  {seconds:6.1f}s")
        print(f"  {'':-<{col + 9}}")
        print(f"  {'compute total':<{col}}  {sum(timings.values()):6.1f}s")
        print(f"  {'wall-clock':<{col}}  {wall_elapsed:6.1f}s")
        # Backbone-only figures: the isolated DA3 forward cost (a subset of
        # submap_building / loop_closure) plus the peak GPU memory of one call.
        print(f"  {'backbone fwd':<{col}}  {backbone_seconds:6.1f}s "
              f"({backbone_calls} calls, "
              f"peak {peak_gpu_mem_bytes / 1e6:.0f} MB)")

    # ── frontend thread ───────────────────────────────────────────────────────

    def _frontend(self, frame_source: Iterable[FrameItem], ctx: _RunContext) -> None:
        """Keyframe selection: pulls input frames, batches keyframes into batch_queue.

        Consumes `frame_source` one (image, seq_idx, label) at a time — the
        same code path serves an offline file list and a live stream.  Each
        batch shares its last keyframe with the next batch (the anchor frame)
        — see the module docstring.
        """
        # Replay mode (frozen keyframes) bypasses optical-flow selection and
        # emits exactly the recorded seq_idxs, so two configs are compared on
        # byte-identical frames.  It shares segment mode's list-returning
        # step()/flush() interface, so both take the `list_mode` path.
        if ctx.config.keyframes_from:
            selector = ReplayKeyframeSelector(
                load_keyframe_list(ctx.config.keyframes_from))
            list_mode = True
        elif ctx.config.keyframe.selection_mode == "segment":
            selector = SegmentKeyframeSelector(ctx.config.keyframe)
            list_mode = True
        else:
            selector = OnlineKeyframeSelector(ctx.config.keyframe)
            list_mode = False
        batcher = _KeyframeBatcher(
            ctx.config.submap_size, _effective_overlap(ctx.config),
            warmup_size=getattr(ctx.config, "submap_warmup_size", 0),
            warmup_submaps=getattr(ctx.config, "submap_warmup_submaps", 2),
            flow_budget=getattr(ctx.config, "submap_flow_budget", 0.0),
            keyframe_config=ctx.config.keyframe)

        # (seq_idx, label) of every selected keyframe, accumulated only when
        # --dump_keyframes is set (frozen-keyframe capture).
        keyframe_log: list[tuple[int, str]] | None = (
            [] if ctx.config.dump_keyframes else None)

        def emit(new_keyframes: list[tuple[str, np.ndarray, int]]) -> None:
            for label, image, seq_idx in new_keyframes:
                if keyframe_log is not None:
                    keyframe_log.append((seq_idx, label))
                batch = batcher.add(label, image, seq_idx)
                if batch is not None:
                    _blocking_put(ctx, ctx.batch_queue, batch)

        try:
            for image, seq_idx, label in frame_source:
                if ctx.backend_error is not None:
                    break

                t0 = time.time()
                if list_mode:
                    new_keyframes = selector.step(image, seq_idx, label)
                else:
                    new_keyframes = (
                        [(label, image, seq_idx)] if selector.step(image) else []
                    )
                ctx.timings["keyframe_selection"] += time.time() - t0
                emit(new_keyframes)

            # Drain the final partial segment (list-mode selectors only).
            if list_mode and ctx.backend_error is None:
                t0 = time.time()
                tail_keyframes = selector.flush()
                ctx.timings["keyframe_selection"] += time.time() - t0
                emit(tail_keyframes)

            # Flush the final partial batch.
            tail_batch = batcher.tail()
            if tail_batch is not None and ctx.backend_error is None:
                _blocking_put(ctx, ctx.batch_queue, tail_batch)

            # Persist the selected keyframe list once the stream is fully
            # consumed (skip on error — the list would be truncated).
            if keyframe_log is not None and ctx.backend_error is None:
                save_keyframe_list(ctx.config.dump_keyframes, keyframe_log)
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
                    semantic_vectors = self.semantic_embedder.encode_frames(submap)
                    submap.set_all_semantic_vectors(semantic_vectors)
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
        pose_graph = PoseGraph(ctx.config.noise,
                               ctx.config.pose_parameterisation)
        overlap = _effective_overlap(ctx.config)

        # Running product of inter-submap scale ratios: converts translations
        # from the current submap's metric unit to submap 0's unit.
        accumulated_scale: float = 1.0
        # Snapshots handed to the loop-closure worker (it runs concurrently and must
        # not observe later mutations).
        submap_scales: dict[int, float] = {}
        submaps_by_idx: dict[int, Submap] = {}
        # Gate-failing loop closures awaiting corroboration (see the
        # corroboration-pool comment above _drain_loop_closure_results).
        pending_loops: list[_HeldLoop] = []

        tag = f"[{threading.current_thread().name}]"
        try:
            while True:
                submap = ctx.submap_queue.get()
                if submap is None:
                    break

                t0 = time.time()
                prev_submap = ctx.submaps[-1] if ctx.submaps else None
                boundary_broken = False
                if prev_submap is None:
                    self._add_first_submap_to_graph(pose_graph, submap)
                else:
                    raw_delta_scale, delta_scale = _boundary_delta_scale(
                        prev_submap, submap, overlap, ctx.config, tag)
                    accumulated_scale *= delta_scale
                    # A break was detected either by the scale dead-band
                    # (delta applied != 1) or by the shared-pair pose check.
                    boundary_broken = (
                        ctx.config.boundary_scale_deadband > 0
                        and delta_scale != 1.0)
                    if overlap >= 2:
                        boundary_broken |= _check_boundary_consistency(
                            prev_submap, submap, overlap, tag)
                    self._add_submap_to_graph(pose_graph, submap, accumulated_scale)
                    print(f"{tag} Submap {submap.idx}: scale={accumulated_scale:.4f} "
                          f"(Δ raw={raw_delta_scale:.4f} applied={delta_scale:.4f})")
                self._add_consecutive_frame_factors(
                    pose_graph, submap, accumulated_scale,
                    tuple(ctx.config.submap_skip_strides))

                ctx.submaps.append(submap)
                submap_scales[submap.idx] = accumulated_scale
                submaps_by_idx[submap.idx] = submap
                ctx.timings["graph_building"] += time.time() - t0

                if self.detector is not None:
                    # A broken boundary triggers a repair re-inference in the
                    # worker: a fresh DA3 pass over a frame pair spanning the
                    # boundary arbitrates the two contradictory measurements.
                    repair = ((prev_submap.idx, submap.idx)
                              if boundary_broken and prev_submap is not None else None)
                    ctx.loop_closure_queue.put(
                        (submap, dict(submaps_by_idx), dict(submap_scales), repair))

                inserted_loops = _drain_loop_closure_results(
                    ctx, pose_graph, pending_loops)
                if inserted_loops and ctx.on_loop_closure is not None:
                    self._emit_loop_closure_snapshot(
                        ctx, pose_graph.get_pose, inserted_loops, phase="pre")

                t0 = time.time()
                optimization = pose_graph.optimize()
                ctx.timings["optimization"] += time.time() - t0
                print(f"{tag} Submap {submap.idx}: "
                      f"{pose_graph.n_nodes} nodes  "
                      f"{pose_graph.n_factors} factors  "
                      f"error={optimization.final_error:.4f}")

                if inserted_loops and ctx.on_loop_closure is not None:
                    self._emit_loop_closure_snapshot(
                        ctx, optimization.pose, inserted_loops, phase="post")

                if ctx.on_update is not None:
                    self._emit_update(ctx, submap, optimization)

            if ctx.backend_error is not None:
                return

            if not ctx.submaps:
                raise RuntimeError(
                    "No submaps built — sequence too short or no keyframes detected."
                )

            # Wait for the loop-closure worker to finish, then incorporate its last results.
            if self.detector is not None:
                ctx.loop_closure_queue.put(None)  # sentinel
                ctx.loop_closure_done.wait()
            inserted_loops = _drain_loop_closure_results(
                ctx, pose_graph, pending_loops)
            if inserted_loops and ctx.on_loop_closure is not None:
                self._emit_loop_closure_snapshot(
                    ctx, pose_graph.get_pose, inserted_loops, phase="pre")

            print(f"{tag} Final optimisation "
                  f"({pose_graph.n_nodes} nodes, {pose_graph.n_factors} factors) ...")
            t0 = time.time()
            ctx.optimization_result = pose_graph.optimize(verbose=True)
            ctx.timings["optimization"] += time.time() - t0

            if inserted_loops and ctx.on_loop_closure is not None:
                self._emit_loop_closure_snapshot(
                    ctx, ctx.optimization_result.pose, inserted_loops, phase="post")
            print(f"{tag} Done: error={ctx.optimization_result.final_error:.4f}  "
                  f"iters={ctx.optimization_result.iterations}  "
                  f"submaps={len(ctx.submaps)}  "
                  f"loop_closures={len(ctx.loop_closures)}")

            # Final refresh so the live view reflects the globally-optimised
            # trajectory (incl. the last loop closures), not just the last
            # incremental step.
            if ctx.on_update is not None:
                self._emit_update(ctx, ctx.submaps[-1], ctx.optimization_result)

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
        anchor_global_cam_to_world = pose_graph.get_pose(submap.frames[0].seq_idx)
        anchor_local_world_to_cam = submap.frames[0].extrinsic.astype(np.float64)
        for frame in submap.frames[1:]:
            local_cam_to_world = frame.cam_to_world.astype(np.float64)
            relative = anchor_local_world_to_cam @ local_cam_to_world
            pose_graph.add_frame(
                frame.seq_idx,
                anchor_global_cam_to_world @ _scaled_translation(relative, accumulated_scale),
            )

    @staticmethod
    def _add_consecutive_frame_factors(
        pose_graph: PoseGraph,
        submap: Submap,
        accumulated_scale: float,
        skip_strides: tuple[int, ...] = (),
    ) -> None:
        """Add a between-factor for each consecutive frame pair in the submap.

        With submap_overlap >= 2, pairs inside the shared anchor block were
        already measured by the previous submap — the second factor is an
        *independent* DA3 measurement of the same pair and is added on
        purpose: the redundancy stops one broken batch from silently
        displacing everything after the boundary.
        `skip_strides` additionally links frames that are k apart *within the
        same batch* (k = 2, 4, 8, ...).  This is not extra computation: DA3
        estimates every frame of a batch JOINTLY, so the relative pose between
        frame i and frame i+k is a single direct measurement — not the
        composition of k consecutive ones.  Adding it is free information that
        the consecutive-only chain throws away.

        Why it matters: measured per-keyframe error has a floor that does not
        shrink with baseline (RPE/step triples from 0.089 to 0.312 as keyframe
        spacing shrinks 9x), so a chain of k short steps accumulates ~sqrt(k)
        floors where one stride-k measurement carries just one.  It also makes
        the graph OVER-determined — a consecutive-only chain has exactly
        n_nodes factors for n_nodes nodes, so it is exactly determined and the
        noise model cannot influence the solution at all.  Redundancy is what
        lets the optimiser average the floor down.
        """
        strides = (1, *sorted({int(k) for k in (skip_strides or ()) if int(k) > 1}))
        frames = submap.frames
        for stride in strides:
            for previous, current in zip(frames, frames[stride:]):
                relative = (
                    previous.extrinsic.astype(np.float64)
                    @ current.cam_to_world.astype(np.float64)
                )
                pose_graph.add_between(
                    previous.seq_idx,
                    current.seq_idx,
                    _scaled_translation(relative, accumulated_scale),
                )

    @staticmethod
    def _emit_loop_closure_snapshot(
        ctx: _RunContext,
        pose_fn: Callable[[int], np.ndarray],
        pairs: list[tuple[int, int]],
        phase: str,
    ) -> None:
        """Fire on_loop_closure with the keyframe poses read from `pose_fn`
        (seq_idx → 4x4: PoseGraph.get_pose for the live graph values
        pre-optimisation, or OptimizationResult.pose post)."""
        poses = _build_keyframe_poses(ctx.submaps, pose_fn)
        try:
            ctx.on_loop_closure(poses, pairs, phase)
        except Exception as exc:  # snapshot consumer must never kill the run
            print(f"[{threading.current_thread().name}] "
                  f"on_loop_closure callback error (ignored): {exc!r}")

    @staticmethod
    def _emit_update(ctx: _RunContext, submap: Submap, optimization: OptimizationResult) -> None:
        """Build a SLAMUpdate for the just-optimised submap and fire on_update.

        Points are projected with the *current* optimised poses.  The shared
        anchor frames are skipped for every submap after the first so the
        live cloud does not double-log them.  A failing viewer callback is
        logged and swallowed — it must never take down the SLAM run.
        """
        poses = _build_keyframe_poses(ctx.submaps, optimization.pose)

        overlap = _effective_overlap(ctx.config)
        frames = submap.frames if submap.idx == 0 else submap.frames[overlap:]
        frame_points_cam = []
        points_list, colors_list = [], []
        for frame in frames:
            if len(frame.points_cam) == 0:
                continue
            frame_points_cam.append((frame.seq_idx, frame.points_cam, frame.colors))
            points_to_world = optimization.point_transform(frame.seq_idx)
            points_list.append(transform_points(frame.points_cam, points_to_world))
            colors_list.append(frame.colors)
        new_points = (np.concatenate(points_list) if points_list
                      else np.empty((0, 3), dtype=np.float32))
        new_colors = (np.concatenate(colors_list) if colors_list
                      else np.empty((0, 3), dtype=np.uint8))

        update = SLAMUpdate(
            submap_idx=submap.idx,
            keyframe_poses=poses,
            new_points_world=new_points,
            new_colors=new_colors,
            n_submaps=len([s for s in ctx.submaps if not s.is_loop_closure_submap]),
            n_loop_closures=len(ctx.loop_closures),
            frame_points_cam=frame_points_cam,
            keyframe_scales={k: optimization.scale(k) for k in poses},
        )
        try:
            ctx.on_update(update)
        except Exception as exc:  # viewer must never kill the pipeline
            print(f"[{threading.current_thread().name}] "
                  f"on_update callback error (ignored): {exc!r}")

    # ── loop closure thread ───────────────────────────────────────────────────

    def _loop_closure_worker(self, ctx: _RunContext) -> None:
        """Detects and verifies loop closures; posts scaled factors for _processing.

        Receives (submap, submaps_by_idx, submap_scales, repair) snapshots so
        it can run concurrently with graph building.  `repair` names a broken
        boundary (prev_idx, curr_idx): a fresh DA3 pass over a frame pair
        spanning it produces a third, independent measurement that arbitrates
        the two contradictory boundary factors — including rotation, so the
        segments cannot settle at different angles.  Repair closures are
        posted `trusted` (the frames are known-adjacent, aliasing is not a
        concern) and bypass the geometric gate — in exchange they must pass
        the batch-chain rotation sanity check (_repair_rotation_deviation)
        on top of the confidence gate.
        """
        overlap = _effective_overlap(ctx.config)
        try:
            while True:
                item = ctx.loop_closure_queue.get()
                if item is None or ctx.backend_error is not None:
                    break
                submap, submaps_by_idx, submap_scales, repair = item

                t0 = time.time()
                verified = [(closure, False) for closure in self.detector.process(submap)]
                if repair is not None:
                    prev_idx, curr_idx = repair
                    prev_frames = submaps_by_idx[prev_idx].frames
                    curr_frames = submaps_by_idx[curr_idx].frames
                    # One frame off the shared anchor block on each side:
                    # independent of both disputed measurements, with as much
                    # visual overlap as possible (wide-baseline re-inference
                    # is what produces garbage repairs).
                    frame_idx_a = max(len(prev_frames) - overlap - 2, 0)
                    frame_idx_b = min(overlap + 1, len(curr_frames) - 1)
                    closure = self.detector.verify_boundary(
                        prev_idx, frame_idx_a, curr_idx, frame_idx_b)
                    reject_reason = "low re-inference confidence"
                    if closure is not None:
                        deviation = _repair_rotation_deviation(
                            prev_frames, frame_idx_a, curr_frames, frame_idx_b,
                            overlap, closure.relative_b_to_a)
                        if (np.isfinite(deviation)
                                and deviation > _REPAIR_MAX_ROTATION_DEV_DEG):
                            reject_reason = (
                                f"rotation {deviation:.0f}° off both batch "
                                f"chains (> {_REPAIR_MAX_ROTATION_DEV_DEG:g}°)")
                            closure = None
                    if closure is not None:
                        verified.append((closure, True))
                    else:
                        print(f"[{threading.current_thread().name}] boundary "
                              f"repair {prev_idx}→{curr_idx} rejected "
                              f"({reject_reason})")
                ctx.timings["loop_closure"] += time.time() - t0

                for closure, trusted in verified:
                    candidate = closure.candidate
                    frame_b = (submaps_by_idx[candidate.submap_idx_b]
                               .frames[candidate.frame_idx_b])
                    frame_a = (submaps_by_idx[candidate.submap_idx_a]
                               .frames[candidate.frame_idx_a])

                    # The loop-closure re-inference has its own arbitrary metric scale.
                    # Convert its translation to the global unit in two steps:
                    # re-inference unit → query submap unit (depth ratio at the shared
                    # query frame), then → global unit (the query submap's
                    # accumulated scale).
                    reinference_to_query_scale = _estimate_depth_scale(
                        frame_b.depth,
                        closure.reinference_submap
                        .frames[closure.query_frame_pos].depth,
                    )
                    global_scale = (
                        reinference_to_query_scale * submap_scales[candidate.submap_idx_b]
                    )
                    relative_b_to_a = _scaled_translation(
                        closure.relative_b_to_a, global_scale
                    )

                    ctx.loop_closure_result_queue.put(
                        (frame_b.seq_idx, frame_a.seq_idx, relative_b_to_a,
                         closure, trusted)
                    )

        except Exception as exc:
            ctx.backend_error = exc
        finally:
            ctx.loop_closure_done.set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_keyframe_poses(
    submaps: list[Submap],
    pose_fn: Callable[[int], np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Return the per-frame cam-to-world poses read from `pose_fn` (seq_idx → 4x4,
    e.g. OptimizationResult.pose or PoseGraph.get_pose).

    The anchor frame (shared between consecutive submaps) is included once —
    the first submap that contributed it wins.  Loop-closure submaps are skipped.
    """
    poses: dict[int, np.ndarray] = {}
    for submap in submaps:
        if submap.is_loop_closure_submap:
            continue
        for frame in submap.frames:
            if frame.seq_idx not in poses:
                poses[frame.seq_idx] = pose_fn(frame.seq_idx).astype(np.float32)
    return poses


def _estimate_boundary_scale(
    prev_submap: Submap,
    curr_submap: Submap,
    overlap: int = 1,
) -> float:
    """
    Estimate the metric scale of curr_submap relative to prev_submap.

    The first shared anchor frame (prev.frames[-overlap] == curr.frames[0])
    observed the same scene in both DA3 batches.  The median depth ratio gives
    the scale factor needed to bring curr_submap's translations into the same
    metric unit as prev_submap.
    """
    return _estimate_depth_scale(
        prev_submap.frames[-overlap].depth,
        curr_submap.frames[0].depth,
    )


def _check_boundary_consistency(
    prev_submap: Submap,
    curr_submap: Submap,
    overlap: int,
    tag: str,
) -> bool:
    """Flag a broken submap boundary (needs submap_overlap >= 2).

    The first shared frame pair is measured by *both* DA3 batches: prev's
    frames[-overlap:-overlap+2] and curr's frames[0:2] are the same physical
    frames.  A large disagreement between the two relative-pose measurements
    means one batch's odometry is wrong at the boundary — the classic cause
    of the trajectory suddenly jumping and the map rebuilding elsewhere.
    The duplicate between-factor added by _add_consecutive_frame_factors
    makes the graph split the difference instead of silently trusting the
    broken measurement; returning True additionally triggers a boundary
    repair re-inference (see _loop_closure_worker), which arbitrates the
    disagreement with a third independent measurement.
    """
    frame_pa = prev_submap.frames[-overlap]
    frame_pb = prev_submap.frames[-overlap + 1]
    frame_ca, frame_cb = curr_submap.frames[0], curr_submap.frames[1]
    if (frame_pa.seq_idx, frame_pb.seq_idx) != (frame_ca.seq_idx, frame_cb.seq_idx):
        return False  # short tail batch — pairing assumption does not hold
    relative_prev = (frame_pa.extrinsic.astype(np.float64)
                     @ frame_pb.cam_to_world.astype(np.float64))
    relative_curr = (frame_ca.extrinsic.astype(np.float64)
                     @ frame_cb.cam_to_world.astype(np.float64))
    rotation_diff = _rotation_angle_deg(np.linalg.inv(relative_prev) @ relative_curr)
    norm_prev = float(np.linalg.norm(relative_prev[:3, 3]))
    norm_curr = float(np.linalg.norm(relative_curr[:3, 3]))
    if norm_prev > 1e-9:
        norm_ratio = norm_curr / norm_prev
    else:
        norm_ratio = float("inf") if norm_curr > 1e-9 else 1.0
    broken = ((np.isfinite(rotation_diff) and rotation_diff > 15.0)
              or norm_ratio < 0.5 or norm_ratio > 2.0)
    # Always emit the measurement, not only the failures.  "How often is a
    # boundary broken" and "how large is the typical disagreement" are
    # different questions, and only the second one distinguishes a genuinely
    # mis-aligned boundary from an intact one that simply costs a little.
    # Only reachable with submap_overlap >= 2, so normal runs print nothing.
    print(f"{tag} [boundary] {prev_submap.idx}->{curr_submap.idx} "
          f"rot_deg={rotation_diff:.4f} norm_ratio={norm_ratio:.4f} "
          f"broken={int(broken)}")
    if broken:
        print(f"{tag} WARNING: boundary {prev_submap.idx}→{curr_submap.idx} "
              f"inconsistent — the two batches disagree on the shared frame "
              f"pair (rotation {rotation_diff:.1f}°, translation-norm ratio "
              f"{norm_ratio:.2f}); possible odometry break here")
    return broken


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
