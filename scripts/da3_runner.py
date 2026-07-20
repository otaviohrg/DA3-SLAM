"""
DA3-SLAM run adapter for the shared benchmark standard.

DA3-SLAM is an in-process Python system: a single Depth-Anything-3 model is
loaded once and reused across sequences; each sequence is run with optical-flow
keyframe selection → per-submap DA3 prediction → SL(4) factor-graph
optimisation, and the optimised camera-to-world poses are read back out.

This adapter exposes:
  * add_da3_cli(parser) — add DA3-SLAM knobs to a benchmark parser.
  * load_da3_model(args) — load the heavy model once (call in main()); returns a
                           SharedSLAM that reuses the model across sequences.
  * run_da3(...)        — run one sequence and return the standardized
                          (est_ts_to_pose, timings, counts) triple.
  * run_benchmark(...)  — the shared per-sequence main() loop used by every
                          benchmark_*.py driver.

DA3-SLAM keys optimised poses by the input frame index (seq_idx).  Each dataset
loader already returns the matching `timestamps` list, so the frame index is
mapped straight to its real/synthetic timestamp in seconds — uniform across TUM,
EuRoC and Replica, with no per-dataset key handling needed.

DA3-SLAM is monocular → the headline ATE is the Sim3-aligned one.

The heavy GPU stack (torch, gtsam, DA3) is only imported when a model is
actually loaded — module import and --help stay GPU-free.
"""

from __future__ import annotations

import time
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np

import benchmark_common as bc
from da3_slam.config import DEFAULT_YAML, load_slam_config
from trajectory_snapshots import TrajectorySnapshotter
from tum_eval_common import SharedSLAM


def add_da3_cli(parser) -> None:
    """Add DA3-SLAM knobs (mirrors run_slam.py) to a benchmark parser."""
    parser.add_argument("--config", default=str(DEFAULT_YAML),
                        help="YAML config file")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap frames per sequence (for quick tests)")

    # SLAM overrides (None = use the YAML value)
    parser.add_argument("--submap_size", type=int, default=None,
                        help="Max keyframes per submap (including anchor overlap)")
    parser.add_argument("--submap_overlap", type=int, default=None,
                        help="Anchor keyframes shared between consecutive "
                             "submaps (>= 2 adds boundary redundancy + the "
                             "boundary-consistency check)")
    parser.add_argument("--confidence_percentile", type=float, default=None,
                        help="Global confidence percentile threshold (0-100)")

    # keyframe selection
    parser.add_argument("--min_disparity_fraction", type=float, default=None,
                        help="Disparity mode: min optical flow as fraction of width")
    parser.add_argument("--selection_mode", choices=["disparity", "segment"],
                        default=None,
                        help="Keyframe policy: 'disparity' (frame-level) or "
                             "'segment' (segment-level density control)")
    parser.add_argument("--segment_length", type=int, default=None,
                        help="Segment mode: frames per segment (N_S)")
    parser.add_argument("--segment_threshold", type=float, default=None,
                        help="Segment mode: accumulated flow (px) → dense stride")
    parser.add_argument("--sharpness_window", type=int, default=None,
                        help="Segment mode blur gate: swap each stride-picked "
                             "keyframe for the sharpest frame within ±window "
                             "(0 = off)")
    parser.add_argument("--min_sharpness_ratio", type=float, default=None,
                        help="Disparity mode blur gate: defer keyframes whose "
                             "sharpness is below this fraction of the recent "
                             "median (0 = off)")

    # loop closure
    parser.add_argument("--no_loop_closure", action="store_true",
                        help="Disable loop closure detection")
    parser.add_argument("--loop_distance_threshold", "--loop_threshold",
                        type=float, default=None,
                        help="DINO-SALAD descriptor L2 distance threshold "
                             "(lower = stricter)")
    parser.add_argument("--min_submaps_apart", type=int, default=None,
                        help="Min submap index gap for loop-closure candidates")
    parser.add_argument("--max_loop_closures", type=int, default=None,
                        help="Max verified loop closures kept per submap")
    parser.add_argument("--min_confidence_ratio", type=float, default=None,
                        help="Min mean DA3 confidence over the matched frames "
                             "to accept a loop closure")
    parser.add_argument("--loop_context_frames", type=int, default=None,
                        help="Temporal neighbours per matched frame in the "
                             "loop-closure re-inference (0 = pair only)")
    parser.add_argument("--loop_max_rotation_error", type=float, default=None,
                        help="Geometric gate: max rotation disagreement (deg) "
                             "between a loop measurement and the graph "
                             "prediction (-1 = disable)")
    parser.add_argument("--loop_max_translation_error", type=float, default=None,
                        help="Geometric gate: max translation disagreement "
                             "(global metric units) between a loop measurement "
                             "and the graph prediction (-1 = disable)")

    # pose graph noise
    parser.add_argument("--between_huber_k", type=float, default=None,
                        help="Huber kernel k on odometry between factors "
                             "(concentrates a broken boundary's error at that "
                             "factor instead of deforming the cycle; "
                             "-1 = plain Gaussian)")

    # inter-submap scale chaining
    parser.add_argument("--boundary_scale_damping", type=float, default=None,
                        help="Damping g for inter-submap scale chaining: each "
                             "boundary depth-ratio is raised to (1-g). "
                             "0 = full chaining, 1 = trust DA3 metric depth")
    parser.add_argument("--boundary_scale_clamp", type=float, default=None,
                        help="Clamp each boundary scale ratio to [1/c, c] "
                             "(unset = no clamping)")
    parser.add_argument("--boundary_scale_deadband", type=float, default=None,
                        help="Scale-break dead-band d: ratios within "
                             "max(r,1/r)-1 <= d are forced to 1.0, ratios "
                             "outside are applied in full (0 = off)")

    # model
    parser.add_argument("--depth_model", default=None)
    parser.add_argument("--depth_model_resolution", type=int, default=None)

    # diagnostics
    parser.add_argument("--snapshot_interval", type=float, default=0.0,
                        help="Save a top-down trajectory snapshot to "
                             "<out_dir>/snapshots every N wall-clock seconds, "
                             "plus a pre-optimisation snapshot (red chord = "
                             "detected drift) at every loop closure (0 = off)")


def build_config(args: Namespace):
    """Build the SLAMConfig from YAML, applying CLI overrides on top."""
    config = load_slam_config(
        args.config,
        submap_size=args.submap_size,
        submap_overlap=args.submap_overlap,
        confidence_percentile=args.confidence_percentile,
        depth_model=args.depth_model,
        depth_model_resolution=args.depth_model_resolution,
        boundary_scale_damping=args.boundary_scale_damping,
        boundary_scale_clamp=args.boundary_scale_clamp,
        boundary_scale_deadband=args.boundary_scale_deadband,
    )
    if args.no_loop_closure:
        config.enable_loop_closure = False
    if args.loop_distance_threshold is not None:
        config.loop_closure.distance_threshold = args.loop_distance_threshold
    if args.min_submaps_apart is not None:
        config.loop_closure.min_submaps_apart = args.min_submaps_apart
    if args.max_loop_closures is not None:
        config.loop_closure.max_loop_closures = args.max_loop_closures
    if args.min_confidence_ratio is not None:
        config.loop_closure.min_confidence_ratio = args.min_confidence_ratio
    if args.loop_context_frames is not None:
        config.loop_closure.context_frames = args.loop_context_frames
    if args.loop_max_rotation_error is not None:
        # -1 disables the gate (argparse cannot pass None explicitly)
        config.loop_closure.max_rotation_error_deg = (
            None if args.loop_max_rotation_error < 0
            else args.loop_max_rotation_error)
    if args.loop_max_translation_error is not None:
        config.loop_closure.max_translation_error = (
            None if args.loop_max_translation_error < 0
            else args.loop_max_translation_error)
    if args.between_huber_k is not None:
        config.noise.between_huber_k = (
            None if args.between_huber_k < 0 else args.between_huber_k)
    if args.min_disparity_fraction is not None:
        config.keyframe.min_disparity_fraction = args.min_disparity_fraction
    if args.selection_mode is not None:
        config.keyframe.selection_mode = args.selection_mode
    if args.segment_length is not None:
        config.keyframe.segment_length = args.segment_length
    if args.segment_threshold is not None:
        config.keyframe.segment_disparity_threshold = args.segment_threshold
    if args.sharpness_window is not None:
        config.keyframe.sharpness_window = args.sharpness_window
    if args.min_sharpness_ratio is not None:
        config.keyframe.min_sharpness_ratio = args.min_sharpness_ratio
    return config


def load_da3_model(args: Namespace):
    """Load the DA3 model once, wrapped in a SharedSLAM (share across sequences).

    SharedSLAM.run() resets the loop-closure detector before each sequence so
    descriptors never leak across sequences.
    """
    print("Loading Depth-Anything-3 model...")
    slam = SharedSLAM(build_config(args))
    print("DA3 model loaded.")
    return slam


class _FanoutCallback:
    """Callable that forwards one callback argument to several consumers."""

    def __init__(self, *callbacks):
        self._callbacks = callbacks

    def __call__(self, update) -> None:
        for callback in self._callbacks:
            callback(update)


def run_da3(
    image_paths: list[str],
    timestamps: list[float],
    model,
    args: Namespace,
    on_update=None,
    out_dir=None,
    gt=None,
) -> tuple[dict[float, np.ndarray], dict, dict]:
    """Run one sequence and return (est_ts_to_pose, timings, counts).

    `timestamps[i]` is the real/synthetic timestamp (seconds) of input frame i;
    DA3-SLAM keys optimised poses by frame index, so the estimate is keyed back
    to those timestamps directly.  `on_update` is forwarded to the pipeline
    (per-submap live-viewer hook; see da3_slam.slam.SLAMUpdate).  With
    --snapshot_interval > 0 (and an `out_dir` to write into), a
    TrajectorySnapshotter also saves periodic trajectory PNGs plus a
    pre-optimisation PNG at every loop closure to <out_dir>/snapshots;
    passing `gt` (benchmark_common.load_groundtruth output) overlays the
    Sim3-aligned ground truth on every snapshot.
    """
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
        timestamps = timestamps[: args.max_frames]

    on_loop_closure = None
    # getattr: kf_submap_grid.py's parser defines neither snapshot flag.
    interval = getattr(args, "snapshot_interval", 0.0) or 0.0
    if interval > 0 and out_dir is not None:
        snapshotter = TrajectorySnapshotter(
            Path(out_dir) / "snapshots", interval_s=interval,
            gt=gt, timestamps=timestamps,
            max_diff=getattr(args, "max_diff", 0.02))
        on_loop_closure = snapshotter.on_loop_closure
        if on_update is None:
            on_update = snapshotter.on_update
        else:
            on_update = _FanoutCallback(on_update, snapshotter.on_update)

    t0 = time.time()
    result = model.run(image_paths, on_update=on_update,
                       on_loop_closure=on_loop_closure)
    total_s = time.time() - t0

    ts_map = {i: ts for i, ts in enumerate(timestamps)}
    est_ts_to_pose = {
        ts_map[seq_idx]: pose
        for seq_idx, pose in result.keyframe_poses.items()
        if seq_idx in ts_map
    }

    # Loop-closure re-inference adds helper submaps; exclude them from the count.
    n_submaps_real = len([s for s in result.submaps if not s.is_loop_closure_submap])

    # Loop-closure endpoints as (query_ts, detected_ts) pairs for the
    # trajectory plot: each candidate's (submap_idx, frame_idx) indexes the
    # original submaps' frames → global seq_idx → timestamp (the same mapping
    # the pipeline uses to key the closure's between-factor).
    submaps_by_idx = {s.idx: s for s in result.submaps
                      if not s.is_loop_closure_submap}
    loop_pairs = []
    for lc in result.loop_closures:
        c = lc.candidate
        try:
            seq_b = submaps_by_idx[c.submap_idx_b].frames[c.frame_idx_b].seq_idx
            seq_a = submaps_by_idx[c.submap_idx_a].frames[c.frame_idx_a].seq_idx
        except (KeyError, IndexError):
            continue
        if seq_b in ts_map and seq_a in ts_map:
            loop_pairs.append((ts_map[seq_b], ts_map[seq_a]))
    if len(loop_pairs) < len(result.loop_closures):
        print(f"  [warn] {len(result.loop_closures) - len(loop_pairs)} loop "
              f"closure(s) could not be mapped to trajectory timestamps")

    timings = dict(result.timings)  # per-module breakdown from the pipeline
    timings["n_frames"] = len(image_paths)
    timings["total_s"] = round(total_s, 3)
    timings["fps"] = round(len(image_paths) / total_s, 3) if total_s > 0 else 0.0

    counts = {
        "n_keyframes": result.n_keyframes,
        "n_submaps": n_submaps_real,
        "n_loop_closures": len(result.loop_closures),
        "loop_pairs": loop_pairs,
    }
    return est_ts_to_pose, timings, counts


def run_benchmark(
    args: Namespace,
    seq_paths: list[str],
    benchmark_sequence,
    dataset: str,
    title: str,
    headline: str = "sim3",
    item_label: str = "Sequence",
) -> None:
    """Shared main() loop for the benchmark_*.py drivers.

    Loads the model once, calls `benchmark_sequence(seq_dir, out_dir, args,
    model)` for each entry of `seq_paths` (a failing sequence is reported and
    skipped, never fatal), then prints the cross-sequence summary and writes
    <out_dir>/<dataset>_summary.json.
    """
    model = load_da3_model(args)
    all_metrics = []
    for seq_path in seq_paths:
        seq_dir = Path(seq_path)
        out_dir = Path(args.out_dir) / seq_dir.name
        print(f"\n{'═'*60}\n  {item_label + ':':<9} {seq_dir.name}\n"
              f"  Input:    {seq_dir}\n  Output:   {out_dir}\n{'═'*60}")
        try:
            metrics = benchmark_sequence(seq_dir, out_dir, args, model)
            if metrics is not None:
                all_metrics.append(metrics)
        except Exception as exc:
            print(f"\n  [ERROR] {seq_dir.name}: {exc}")
            traceback.print_exc()

    if len(all_metrics) > 1:
        bc.print_summary(all_metrics, headline, title)
    bc.save_summary(all_metrics, Path(args.out_dir), dataset)
