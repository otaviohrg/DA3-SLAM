"""
Keyframe-density × submap-size grid — the iso-span experiment.

THE HYPOTHESIS
--------------
The UAS merging sweep (results/merging_uas/) found ATE more than doubling as
submap size grew 8 -> 48.  That looked like "big submaps are bad", but submap
size and *batch span* were locked together: at a fixed keyframe density, a
submap of S keyframes covers S x (frames per keyframe) of the sequence, so
growing S grows how much ground one DA3 batch has to co-register.

Two independent datasets say SPAN is the controlling variable, not S:

  * campus_fog's keyframes are 6-8x denser than the other UAS sequences, so its
    submap-128 batches still span only 33 m — and it is the one sequence whose
    ATE stays flat with submap size.  Grouping the UAS cells by matched submap
    size gives an 8.7 pp ATE spread; grouping by matched span (~22-33 m/batch,
    mixing submap 16 and submap 128) gives 3.0 pp.
  * In the older stride x submap study (outputs/kf_submap_grid/grid.csv),
    corr(log frames_per_submap, ATE) = +0.49 vs corr(log submap_size, ATE) =
    +0.21.

This script unlocks the two axes: it sweeps keyframe DENSITY and submap SIZE
independently, so ATE can be regressed on each.  If span is what matters, cells
along an iso-span diagonal (density x S = const) should agree.

THE CONTROL (different from sweep_merging.py — read this before changing it)
---------------------------------------------------------------------------
`sweep_merging.py` freezes ONE keyframe list per sequence and replays it
everywhere.  That is exactly wrong here: varying density is the point.

But submap size must still not leak into keyframe *selection* — the selector
force-emits a keyframe after `keyframe.max_submap_size` frames, and that is
kept in sync with `submap_size`.  So the list is frozen per
**(sequence, mode, density)** and generated once at a fixed reference submap
size (`--freeze_at`), then replayed across every submap size.  Result:

    keyframe list depends on (mode, density)  — never on submap size
    submap size is purely a batching parameter

DENSITY KNOB — one per selection strategy
-----------------------------------------
`density` is a multiplier relative to the shipped default (1.0 = as configured):

  segment mode    `segment_strides` (a, b), default (8, 16) — frames between
                  keyframes in the dense/sparse regime.  Scaled as
                  round(a / density), so density 2 -> (4, 8), density 4 ->
                  (2, 4).  Strides floor at 1.
  disparity mode  `min_disparity_fraction`, default 0.10 — a keyframe is
                  emitted once mean optical flow exceeds this fraction of the
                  image width.  Scaled as 0.10 / density, so density 2 -> 0.05.

Both go the same direction: higher density = more keyframes = shorter span per
submap at a given submap size.

Usage (one invocation per selection strategy — they are separate experiments):
    python scripts/sweep_keyframe_grid.py --selection_mode segment \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_{desk,xyz,room} \\
        --submap_sizes 8 16 32 64 --densities 0.5 1 2 4 --repeats 3 \\
        --backbone_dtype bf16 --out_dir outputs/kfgrid_segment

    python scripts/report_keyframe_grid.py --rows outputs/kfgrid_segment/kfgrid_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from argparse import Namespace
from pathlib import Path

import results_log
from da3_runner import model_short_name, resolve_model_alias
from da3_slam.config import DEFAULT_YAML, load_slam_config
from sweep_compute import _score, prepare_sequence
from tum_eval_common import SharedSLAM

# Shipped defaults the density multiplier is relative to (config/default.yaml).
BASE_SEGMENT_STRIDES = (8, 16)
BASE_MIN_DISPARITY = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keyframe-density × submap-size grid (iso-span experiment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seq_dir", nargs="+", required=True)
    p.add_argument("--dataset", choices=["tum", "uas", "replica"], default="tum")
    p.add_argument("--selection_mode", choices=["segment", "disparity"],
                   required=True,
                   help="Which keyframe strategy this experiment tests.  Run "
                        "one invocation per strategy — they select differently "
                        "enough that pooling them would be meaningless")
    p.add_argument("--out_dir", default=None,
                   help="Default: outputs/kfgrid_<selection_mode>")
    p.add_argument("--results_row", default=None)

    # ── the grid ──────────────────────────────────────────────────────────────
    p.add_argument("--submap_sizes", nargs="+", type=int,
                   default=[8, 16, 32, 64])
    p.add_argument("--densities", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0, 4.0],
                   help="Keyframe density multiplier vs the shipped default "
                        "(2.0 = twice as many keyframes)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--repeat_offset", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Skip cells already present in --results_row")

    # ── model / config ────────────────────────────────────────────────────────
    p.add_argument("--depth_model", default="nested-giant")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--backbone_dtype", choices=["fp32", "bf16"], default="bf16",
                   help="bf16 keeps the larger submaps within GPU memory")
    p.add_argument("--config", default=str(DEFAULT_YAML))
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--no_loop_closure", action="store_true")
    p.add_argument("--submap_overlap", type=int, default=None,
                   help="Anchor keyframes shared between consecutive submaps.  "
                        ">=2 gives the graph two INDEPENDENT DA3 measurements of "
                        "the shared block — genuinely new information at the "
                        "boundary, unlike within-submap skip factors")
    p.add_argument("--boundary_scale_damping", type=float, default=None,
                   help="g in [0,1]: each boundary depth-ratio is raised to "
                        "(1-g).  1.0 = ignore ratios and trust DA3's metric "
                        "consistency (the shipped default); 0.0 = full "
                        "chaining.  With overlap>=2 the ratio is estimated from "
                        "a block of shared frames rather than one frame")
    p.add_argument("--roma_gate", action="store_true",
                   help="Verify loop candidates with RoMa v2 dense matching "
                        "before accepting them.  Pair with a LOOSE "
                        "--loop_distance_threshold: retrieval becomes recall, "
                        "the gate provides precision")
    p.add_argument("--roma_min_overlap", type=float, default=None,
                   help="Reject a candidate below this predicted overlap")
    p.add_argument("--max_loop_closures", type=int, default=None,
                   help="Candidates kept per submap (raise it when retrieval "
                        "is loose, or the gate has nothing to choose from)")
    p.add_argument("--pose_parameterisation", choices=["sl4", "sim3"],
                   default=None, help="Pose-graph parameterisation")
    p.add_argument("--loop_distance_threshold", type=float, default=None,
                   help="DINO-SALAD L2 distance for a loop candidate (lower = "
                        "stricter).  The shipped 0.45 sits at the 6th percentile "
                        "of observed distances and finds almost nothing indoors; "
                        "0.80 tripled closures and cut TUM ATE 30%%")
    p.add_argument("--submap_skip_strides", type=int, nargs="*", default=None,
                   help="Extra within-submap between-factors linking frames k "
                        "apart (e.g. 2 4 8).  Fixed for the whole sweep — run "
                        "twice into different --out_dir to A/B it")

    # ── controls ──────────────────────────────────────────────────────────────
    p.add_argument("--freeze_at", type=int, default=None,
                   help="Submap size used to GENERATE each (mode, density) "
                        "keyframe list (default: max of --submap_sizes).  Every "
                        "submap size then replays it, so selection never "
                        "depends on batching")
    p.add_argument("--keyframes_dir", default=None)
    p.add_argument("--max_diff", type=float, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seqcache", default=None)
    p.add_argument("--topic", default=None)
    p.add_argument("--calib", default=None)
    p.add_argument("--calib_dir", type=Path, default=None)
    p.add_argument("--no_undistort", dest="undistort", action="store_false")
    return p.parse_args()


def apply_density(config, mode: str, density: float) -> str:
    """Set the mode's density knob and return a short label for logs/paths."""
    config.keyframe.selection_mode = mode
    if mode == "segment":
        strides = tuple(max(1, round(base / density))
                        for base in BASE_SEGMENT_STRIDES)
        config.keyframe.segment_strides = strides
        return f"strides{strides[0]}-{strides[1]}"
    fraction = BASE_MIN_DISPARITY / density
    config.keyframe.min_disparity_fraction = fraction
    return f"disp{fraction:.4g}"


def density_tag(density: float) -> str:
    """Filesystem-safe density label (0.5 -> d0p5)."""
    return "d" + f"{density:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.max_diff is None:
        args.max_diff = {"uas": 0.05, "replica": 0.5 / args.fps}.get(
            args.dataset, 0.02)

    out_dir = Path(args.out_dir or f"outputs/kfgrid_{args.selection_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_row = Path(args.results_row or out_dir / "kfgrid_rows.jsonl")
    keyframes_dir = Path(args.keyframes_dir or out_dir / "keyframes")
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    seqcache = Path(args.seqcache) if args.seqcache else out_dir / "_seqcache"

    seq_dirs = [Path(s) for s in args.seq_dir]
    submap_sizes = sorted(set(args.submap_sizes))
    densities = sorted(set(args.densities))
    freeze_at = args.freeze_at or max(submap_sizes)

    model_id = resolve_model_alias(args.depth_model)
    cfg = load_slam_config(args.config, depth_model=model_id,
                           depth_model_resolution=args.resolution,
                           backbone_dtype=args.backbone_dtype,
                           submap_size=submap_sizes[0])
    if args.no_loop_closure:
        cfg.enable_loop_closure = False
    if args.submap_skip_strides is not None:
        cfg.submap_skip_strides = tuple(args.submap_skip_strides)
    if args.submap_overlap is not None:
        cfg.submap_overlap = args.submap_overlap
    if args.boundary_scale_damping is not None:
        cfg.boundary_scale_damping = args.boundary_scale_damping
    if args.loop_distance_threshold is not None:
        cfg.loop_closure.distance_threshold = args.loop_distance_threshold
    if args.pose_parameterisation is not None:
        cfg.pose_parameterisation = args.pose_parameterisation
    if args.roma_gate:
        cfg.loop_closure.roma_gate = True
    if args.roma_min_overlap is not None:
        cfg.loop_closure.roma_min_overlap = args.roma_min_overlap
    if args.max_loop_closures is not None:
        cfg.loop_closure.max_loop_closures = args.max_loop_closures

    done = _completed_cells(results_row) if args.resume else set()
    n_runs = len(densities) * len(submap_sizes) * args.repeats * len(seq_dirs)
    print(f"\n{'━' * 70}")
    print(f"  Keyframe-density × submap-size grid — mode: {args.selection_mode}")
    print(f"  {len(densities)} densities × {len(submap_sizes)} submap sizes × "
          f"{args.repeats} repeats × {len(seq_dirs)} sequences = {n_runs} runs")
    print(f"  model      {model_short_name(model_id)} @ "
          f"{cfg.depth_model_resolution}  backbone {cfg.backbone_dtype}")
    print(f"  densities  {densities}   submaps {submap_sizes}")
    print(f"  keyframes  frozen per (sequence, density) at submap {freeze_at}")
    if done:
        print(f"  resume     {len(done)} cell(s) already logged will be skipped")
    print(f"  log     → {results_row}\n{'━' * 70}\n")

    model = SharedSLAM(cfg)
    model.set_build_pointclouds(False)
    prepared: dict[str, tuple | None] = {}

    # One frozen keyframe list per (sequence, density) — generated up front so
    # it never depends on the order the grid is walked in.
    for density in densities:
        _freeze_for_density(model, seq_dirs, args, density, freeze_at,
                            keyframes_dir, prepared, seqcache)

    run_idx = 0
    t0 = time.time()
    for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
        for density in densities:
            label = apply_density(model.config, args.selection_mode, density)
            for submap_size in submap_sizes:
                model.config.submap_size = submap_size
                for seq_dir in seq_dirs:
                    run_idx += 1
                    tag = (f"[{run_idx}/{n_runs}] d={density:g}({label}) "
                           f"submap={submap_size} rep={repeat} {seq_dir.name}")
                    key = (seq_dir.name, submap_size, density, repeat,
                           model.config.submap_overlap,
                           model.config.boundary_scale_damping,
                           model.config.loop_closure.distance_threshold,
                           model.config.pose_parameterisation,
                           model.config.loop_closure.roma_gate,
                           model.config.loop_closure.roma_min_overlap)
                    if key in done:
                        print(f"  {tag}  [skip: already logged]")
                        continue
                    _run_cell(model, seq_dir, args, density, label, submap_size,
                              repeat, results_row, keyframes_dir, tag,
                              prepared, seqcache)

    print(f"\n{'═' * 70}\n  Grid done: {run_idx} runs in {time.time() - t0:.0f}s"
          f"\n  rows → {results_row}")
    print(f"  next: python scripts/report_keyframe_grid.py --rows {results_row}")


def _kf_path(keyframes_dir: Path, seq_name: str, density: float) -> Path:
    return keyframes_dir / f"{seq_name}__{density_tag(density)}.txt"


def _completed_cells(results_row: Path) -> set[tuple]:
    if not results_row.exists():
        return set()
    done = set()
    with open(results_row) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done.add((row.get("sequence"), row.get("submap_size"),
                          row.get("kf_density"), row.get("repeat")))
    return done


def _freeze_for_density(model, seq_dirs, args, density: float, freeze_at: int,
                        keyframes_dir: Path, prepared: dict,
                        seqcache: Path) -> None:
    """Generate the keyframe list for one density, at the reference submap size."""
    from da3_runner import run_da3

    todo = [s for s in seq_dirs
            if not _kf_path(keyframes_dir, s.name, density).exists()]
    if not todo:
        return

    label = apply_density(model.config, args.selection_mode, density)
    saved_submap = model.config.submap_size
    saved_max = model.config.keyframe.max_submap_size
    model.config.submap_size = freeze_at
    model.config.keyframe.max_submap_size = freeze_at
    print(f"  frozen keyframes: density {density:g} ({label}) — "
          f"{len(todo)} sequence(s)")
    try:
        for seq_dir in todo:
            seq = prepare_sequence(seq_dir, args, seqcache, prepared)
            if seq is None:
                continue
            image_paths, timestamps, _ = seq
            kf_path = _kf_path(keyframes_dir, seq_dir.name, density)
            model.set_keyframe_io(dump_keyframes=str(kf_path))
            try:
                run_da3(image_paths, timestamps, model,
                        Namespace(max_frames=args.max_frames, repeat=0,
                                  snapshot_interval=0.0))
                n_kf = sum(1 for line in open(kf_path)
                           if line.strip() and not line.startswith("#"))
                print(f"    {seq_dir.name}: {n_kf} keyframes "
                      f"({len(image_paths) / max(n_kf, 1):.1f} frames/keyframe)")
            except Exception as exc:
                print(f"    [ERROR] {seq_dir.name}: {exc}")
                traceback.print_exc()
    finally:
        model.config.submap_size = saved_submap
        model.config.keyframe.max_submap_size = saved_max
        model.set_keyframe_io()


def _run_cell(model, seq_dir: Path, args, density: float, label: str,
              submap_size: int, repeat: int, results_row: Path,
              keyframes_dir: Path, tag: str, prepared: dict,
              seqcache: Path) -> None:
    from da3_runner import run_da3

    seq = prepare_sequence(seq_dir, args, seqcache, prepared)
    if seq is None:
        print(f"  [SKIP] {tag}: sequence could not be prepared")
        return
    image_paths, timestamps, gt_all = seq
    kf_path = _kf_path(keyframes_dir, seq_dir.name, density)
    if not kf_path.exists():
        print(f"  [SKIP] {tag}: no keyframe list at {kf_path}")
        return
    print(f"\n  {tag}")
    model.set_keyframe_io(keyframes_from=str(kf_path))

    run_args = Namespace(max_frames=args.max_frames, repeat=repeat,
                         snapshot_interval=0.0)
    try:
        est_ts_to_pose, timings, counts = run_da3(
            image_paths, timestamps, model, run_args)
    except Exception as exc:
        status = _classify(exc)
        print(f"    [{status.upper()}] {tag}: {exc}")
        if status != "oom":
            traceback.print_exc()
        _append(results_row, model, run_args, seq_dir, args, density, label,
                {"status": status}, len(image_paths))
        return

    metrics = _score(est_ts_to_pose, gt_all, seq_dir.name, timings, counts,
                     args.dataset, args.max_diff)
    if metrics is None:
        print(f"    [SKIP] {seq_dir.name}: too few matched poses")
        _append(results_row, model, run_args, seq_dir, args, density, label,
                {"status": "unmatched"}, len(image_paths))
        return

    _append(results_row, model, run_args, seq_dir, args, density, label,
            metrics, len(image_paths))
    span = len(image_paths) / max(metrics.get("n_submaps") or 1, 1)
    print(f"    ATE sim3={metrics['ate_sim3']['rmse']:.4f}m  "
          f"kf={metrics.get('n_keyframes')}  submaps={metrics.get('n_submaps')}  "
          f"span={span:.0f} frames/batch  "
          f"backbone={timings['backbone_forward_s']:.1f}s")


def _classify(exc: Exception) -> str:
    import torch
    if "out of memory" in str(exc).lower():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return "oom"
    return "error"


def _append(results_row: Path, model, run_args, seq_dir: Path, args,
            density: float, label: str, metrics: dict, n_input_frames: int) -> None:
    """Log one cell, carrying the grid's two axes and the derived span."""
    metrics.setdefault("system", "DA3-SLAM")
    metrics.setdefault("dataset", args.dataset)
    metrics.setdefault("sequence", seq_dir.name)
    metrics.setdefault("timings", {})
    row = results_log.build_row(model.config, run_args, metrics)
    row["selection_mode"] = args.selection_mode
    row["skip_strides"] = list(model.config.submap_skip_strides)
    row["submap_overlap"] = model.config.submap_overlap
    row["boundary_scale_damping"] = model.config.boundary_scale_damping
    row["loop_distance_threshold"] = model.config.loop_closure.distance_threshold
    row["pose_parameterisation"] = model.config.pose_parameterisation
    row["roma_gate"] = model.config.loop_closure.roma_gate
    row["roma_min_overlap"] = model.config.loop_closure.roma_min_overlap
    row["max_loop_closures"] = model.config.loop_closure.max_loop_closures
    row["kf_density"] = density
    row["kf_setting"] = label
    row["n_input_frames"] = n_input_frames
    # The hypothesis variable: input frames covered by ONE DA3 batch.  Derived
    # here rather than in the report so every row is self-contained.
    n_submaps = row.get("n_submaps")
    row["frames_per_submap"] = (n_input_frames / n_submaps) if n_submaps else None
    n_kf = row.get("n_keyframes")
    row["frames_per_keyframe"] = (n_input_frames / n_kf) if n_kf else None
    results_log.append_row(results_row, row)


if __name__ == "__main__":
    main()
