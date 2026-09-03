"""
Static-compute sweep driver (plan Step 0f): the accuracy-vs-speed-vs-memory
grid over DA3 model size and input resolution.

For each (size, resolution, repeat, sequence) it runs DA3-SLAM on FROZEN
keyframes and appends one machine-readable row (config + Sim3/SE3 ATE + per-
stage latency + backbone-only latency + peak GPU memory + token proxy) to a
JSONL log.  plot_sweep.py turns that log into the tables/figures.

Design choices that make the grid cheap and honest:
  * One model load per SIZE; resolution is swapped in place (set_resolution) and
    repeats/sequences reuse it.  Size is a different checkpoint → fresh load.
  * FROZEN KEYFRAMES: the first run to touch a sequence dumps its keyframe list
    (optical-flow selection is model/resolution-independent); every later run
    replays it, so ATE differences come from the network, not selection drift.
  * N repeats per config capture the bf16 non-determinism band — a config only
    "wins" if it beats another beyond that band (see plot_sweep aggregation).
  * Lean by default (no point clouds); --map_detail turns them on and saves a
    per-config cloud for the map-quality proxy (Chamfer vs a reference cloud).

Example (the plan's dev subset):
    python scripts/sweep_compute.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_desk \\
                  data/tum/rgbd_dataset_freiburg1_xyz \\
                  data/tum/rgbd_dataset_freiburg1_room \\
        --sizes nested-giant --resolutions 336 392 448 504 --repeats 3 \\
        --out_dir outputs/sweep/resolution
"""

from __future__ import annotations

import argparse
import time
import traceback
from argparse import Namespace
from pathlib import Path

import benchmark_common as bc
import results_log
from da3_runner import model_short_name, resolve_model_alias
from da3_slam.config import DEFAULT_YAML, load_slam_config
from tum_eval_common import SharedSLAM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DA3-SLAM size × resolution compute sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seq_dir", nargs="+", required=True,
                   help="TUM sequence directories (rgb.txt + groundtruth.txt)")
    p.add_argument("--sizes", nargs="+", default=["nested-giant"],
                   help="Model sizes: aliases (small/base/large/giant/"
                        "nested-giant) or full HF IDs")
    p.add_argument("--resolutions", nargs="+", type=int,
                   default=[336, 392, 448, 504],
                   help="DA3 processing resolutions (multiples of patch 14)")
    p.add_argument("--repeats", type=int, default=3,
                   help="Runs per config (>=3 to capture the bf16 noise band)")
    p.add_argument("--out_dir", default="outputs/sweep",
                   help="Root output directory")
    p.add_argument("--results_row", default=None,
                   help="JSONL sweep log (default: <out_dir>/sweep_rows.jsonl)")
    p.add_argument("--keyframes_dir", default=None,
                   help="Where frozen keyframe lists live/are generated "
                        "(default: <out_dir>/keyframes)")
    p.add_argument("--config", default=str(DEFAULT_YAML), help="Base YAML config")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Cap frames per sequence (quick tests)")
    p.add_argument("--no_loop_closure", action="store_true",
                   help="Disable loop closure for the whole sweep")

    # dataset selection + per-dataset I/O
    p.add_argument("--dataset", choices=["tum", "uas", "replica"], default="tum",
                   help="Sequence format: 'tum' (rgb.txt + groundtruth.txt), "
                        "'uas' (ROS bag + fisheye undistort + .tum GT), or "
                        "'replica' (results/frame*.jpg + gt_tum.txt, synthetic ts)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Replica frame rate (synthetic timestamps = idx/fps)")
    p.add_argument("--max_diff", type=float, default=None,
                   help="Estimate↔GT association tolerance (s).  Default is "
                        "dataset-aware: 0.02 for tum (30 Hz GT), 0.05 for uas "
                        "(10 Hz GT + a camera/odometry clock offset up to ~50 ms "
                        "— 0.02 matches zero keyframes and the sequence is "
                        "silently dropped), 0.5/fps for replica (half-frame).")
    # UAS-only (ignored for tum); mirror benchmark_uas.py
    p.add_argument("--topic", default=None, help="UAS camera topic override")
    p.add_argument("--calib", default=None, help="UAS calibration YAML override")
    p.add_argument("--calib_dir", type=Path, default=None,
                   help="UAS calibration/ folder override")
    p.add_argument("--no_undistort", dest="undistort", action="store_false",
                   help="UAS: skip fisheye undistortion")
    p.add_argument("--map_detail", action="store_true",
                   help="Enable point clouds, save a per-config cloud and log "
                        "point count + Chamfer vs a reference cloud")
    p.add_argument("--ref_size", default=None,
                   help="Map-detail reference size (default: first --sizes)")
    p.add_argument("--ref_resolution", type=int, default=None,
                   help="Map-detail reference resolution (default: max --resolutions)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Dataset-aware GT association tolerance (see --max_diff help): UAS GT is
    # 10 Hz with a camera/odometry clock offset, so 0.02 s drops whole sequences;
    # Replica synthesises timestamps at idx/fps, matched to a half-frame.
    if args.max_diff is None:
        args.max_diff = {"uas": 0.05, "replica": 0.5 / args.fps}.get(
            args.dataset, 0.02)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_row = Path(args.results_row) if args.results_row \
        else out_dir / "sweep_rows.jsonl"
    if results_row.exists():
        print(f"  [warn] {results_row} already exists — rows are APPENDED. "
              f"plot_sweep dedupes by (size,res,repeat,seq) keeping the latest; "
              f"delete the file for a clean run.")
    keyframes_dir = Path(args.keyframes_dir) if args.keyframes_dir \
        else out_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = [Path(s) for s in args.seq_dir]
    ref_short = model_short_name(resolve_model_alias(args.ref_size or args.sizes[0]))
    ref_res = args.ref_resolution or max(args.resolutions)
    map_detail_rows: list[dict] = []
    # Prepared (image_paths, timestamps, gt) per sequence — memoised so a UAS
    # bag is extracted/undistorted once and reused across all runs; also under
    # a stable cache dir so it survives between sweeps.
    prepared: dict[str, tuple | None] = {}
    seqcache = out_dir / "_seqcache"

    n_runs = len(args.sizes) * len(args.resolutions) * args.repeats * len(seq_dirs)
    print(f"Sweep: {len(args.sizes)} sizes × {len(args.resolutions)} resolutions "
          f"× {args.repeats} repeats × {len(seq_dirs)} sequences = {n_runs} runs")
    print(f"  log → {results_row}")
    run_idx = 0
    sweep_t0 = time.time()

    # ── one model load per size; resolution swapped in place ──────────────────
    for size in args.sizes:
        model_id = resolve_model_alias(size)
        short = model_short_name(model_id)
        # A checkpoint that fails to download (HF outage) or OOMs must skip its
        # size, not kill the whole ladder.
        try:
            cfg = load_slam_config(args.config, depth_model=model_id,
                                   depth_model_resolution=args.resolutions[0])
            if args.no_loop_closure:
                cfg.enable_loop_closure = False
            print(f"\n{'━'*60}\n  Loading model: {short}  ({model_id})\n{'━'*60}")
            model = SharedSLAM(cfg)
            model.set_build_pointclouds(args.map_detail)  # lean unless map-detail
        except Exception as exc:
            print(f"\n  [ERROR] size {short} failed to load ({exc}) — skipping")
            traceback.print_exc()
            continue

        for resolution in args.resolutions:
            model.set_resolution(resolution)
            for repeat in range(args.repeats):
                for seq_dir in seq_dirs:
                    run_idx += 1
                    tag = f"[{run_idx}/{n_runs}] {short} res={resolution} " \
                          f"rep={repeat} {seq_dir.name}"
                    ply_path = (out_dir / "clouds"
                                / f"{short}_{resolution}_r{repeat}"
                                / f"{seq_dir.name}.ply") if args.map_detail else None
                    try:
                        row = _run_one(model, seq_dir, args, repeat, results_row,
                                       keyframes_dir, ply_path, tag,
                                       prepared, seqcache)
                    except Exception as exc:
                        print(f"    [ERROR] {tag}: {exc} — skipping run")
                        traceback.print_exc()
                        row = None
                    if row is not None and args.map_detail:
                        map_detail_rows.append(
                            {"backbone_size": short, "resolution": resolution,
                             "repeat": repeat, "sequence": seq_dir.name,
                             "n_points": row.get("_n_points"),
                             "ply_path": row.get("_ply_path")})
        # Free the model before loading the next size.
        del model

    if args.map_detail:
        _finish_map_detail(map_detail_rows, ref_short, ref_res,
                           out_dir / "map_detail.jsonl")

    print(f"\n{'═'*60}\n  Sweep done: {run_idx} runs in "
          f"{time.time() - sweep_t0:.0f}s\n  rows → {results_row}")
    print(f"  next: python scripts/plot_sweep.py --rows {results_row} "
          f"--out_dir {out_dir}")


def prepare_sequence(seq_dir: Path, args: Namespace, seqcache: Path,
                     prepared: dict) -> tuple | None:
    """Resolve one sequence to (image_paths, timestamps, gt_all), memoised.

    TUM reads rgb.txt + groundtruth.txt directly; UAS extracts frames from the
    ROS bag (cached under seqcache/<seq>) and fisheye-undistorts them, then
    loads its .tum GT — done once per sequence and reused across every run.
    """
    key = str(seq_dir)
    if key in prepared:
        return prepared[key]

    if args.dataset == "uas":
        import uas_common as uas
        bag, gt_txt = uas.find_bag(seq_dir), uas.find_gt(seq_dir)
        if bag is None or gt_txt is None:
            print(f"  [SKIP] {seq_dir.name}: missing UAS bag or .tum GT")
            prepared[key] = None
            return None
        topic, calib_path = uas.resolve_camera(
            seq_dir, args.topic, args.calib, args.calib_dir)
        calibration = uas.load_calibration(calib_path)
        cache = seqcache / seq_dir.name
        image_paths, timestamps = uas.extract_bag_frames(
            bag, topic, cache / "frames", max_frames=args.max_frames)
        if image_paths and args.undistort:
            image_paths = uas.undistort_frames(
                image_paths, calibration, cache / "undistorted")
    elif args.dataset == "replica":
        gt_txt = seq_dir / "gt_tum.txt"
        if not gt_txt.exists():
            print(f"  [SKIP] {seq_dir.name}: missing gt_tum.txt")
            prepared[key] = None
            return None
        image_paths, timestamps = bc.load_replica_images(
            seq_dir, args.max_frames, args.fps)
    else:
        rgb_txt, gt_txt = seq_dir / "rgb.txt", seq_dir / "groundtruth.txt"
        if not rgb_txt.exists() or not gt_txt.exists():
            print(f"  [SKIP] {seq_dir.name}: missing rgb.txt / groundtruth.txt")
            prepared[key] = None
            return None
        image_paths, timestamps = bc.load_rgb_list(seq_dir, args.max_frames)

    if not image_paths:
        prepared[key] = None
        return None
    prepared[key] = (image_paths, timestamps, bc.load_groundtruth(gt_txt))
    return prepared[key]


def _run_one(model, seq_dir: Path, args: Namespace, repeat: int,
             results_row: Path, keyframes_dir: Path,
             ply_path: Path | None, tag: str,
             prepared: dict, seqcache: Path) -> dict | None:
    """Run one (config, sequence), append its row, and return it (or None)."""
    from da3_runner import run_da3  # local: pulls the heavy stack only when run

    seq = prepare_sequence(seq_dir, args, seqcache, prepared)
    if seq is None:
        print(f"  [SKIP] {tag}: sequence could not be prepared")
        return None
    image_paths, timestamps, gt_all = seq
    print(f"\n  {tag}")

    # Frozen keyframes: first run to touch a sequence generates the list
    # (selection is model/resolution-independent); the rest replay it.
    kf_path = keyframes_dir / f"{seq_dir.name}.txt"
    if kf_path.exists():
        model.set_keyframe_io(keyframes_from=str(kf_path))
    else:
        print(f"    generating frozen keyframe list → {kf_path}")
        model.set_keyframe_io(dump_keyframes=str(kf_path))

    run_args = Namespace(max_frames=args.max_frames, repeat=repeat,
                         snapshot_interval=0.0)
    est_ts_to_pose, timings, counts = run_da3(
        image_paths, timestamps, model, run_args,
        save_ply=str(ply_path) if ply_path is not None else None)

    metrics = _score(est_ts_to_pose, gt_all, seq_dir.name, timings, counts,
                     args.dataset, args.max_diff)
    if metrics is None:
        print(f"    [SKIP] {seq_dir.name}: too few matched poses")
        return None

    row = results_log.build_row(model.config, run_args, metrics)
    # Stash map-detail fields for the caller without polluting the sweep row.
    row["_n_points"] = counts.get("n_points")
    row["_ply_path"] = counts.get("ply_path")
    results_log.append_row(results_row,
                           {k: v for k, v in row.items() if not k.startswith("_")})
    print(f"    ATE sim3={metrics['ate_sim3']['rmse']:.4f}m  "
          f"backbone={timings['backbone_forward_s']:.1f}s  "
          f"peak={timings['peak_gpu_mem_mb']:.0f}MB")
    return row


def _score(est_ts_to_pose: dict, gt_all: list, seq_name: str,
           timings: dict, counts: dict, dataset: str = "tum",
           max_diff: float = 0.02) -> dict | None:
    """Association + ATE/RPE → the metrics dict shape results_log.build_row
    expects (matches benchmark_common.evaluate_trajectory, minus the per-run
    plot/JSON files a 36-run sweep does not want)."""
    gt_stamps = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose)
    pairs = bc.associate(est_stamps, gt_stamps, max_diff=max_diff)
    if len(pairs) < 3:
        return None
    gt_m = [gt_all[ib][1] for _, ib in pairs]
    est_m = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]
    ate_se3 = bc.compute_ate(gt_m, est_m, align="se3")
    ate_sim3 = bc.compute_ate(gt_m, est_m, align="sim3")
    rpe1 = bc.compute_rpe(gt_m, est_m, delta=1)
    drop = ("per_frame_errors", "align_T", "per_frame_trans", "per_frame_rot")
    return {
        "system": "DA3-SLAM", "dataset": dataset, "sequence": seq_name,
        "n_frames": timings.get("n_frames"),
        "n_keyframes": counts.get("n_keyframes"),
        "n_submaps": counts.get("n_submaps"),
        "n_loop_closures": counts.get("n_loop_closures"),
        "ate_se3": {k: v for k, v in ate_se3.items() if k not in drop},
        "ate_sim3": {k: v for k, v in ate_sim3.items() if k not in drop},
        "rpe_delta1": {k: v for k, v in rpe1.items() if k not in drop},
        "timings": timings,
    }


def _finish_map_detail(rows: list[dict], ref_short: str, ref_res: int,
                       out_path: Path) -> None:
    """Post-pass: Chamfer each config's cloud against the per-sequence
    reference cloud (ref_short / ref_res, repeat 0) and write map_detail.jsonl."""
    import map_detail
    ref_by_seq = {
        r["sequence"]: r["ply_path"]
        for r in rows
        if r["backbone_size"] == ref_short and r["resolution"] == ref_res
        and r["repeat"] == 0 and r["ply_path"]
    }
    for r in rows:
        ref = ref_by_seq.get(r["sequence"])
        chamfer = None
        if ref and r["ply_path"] and r["ply_path"] != ref:
            try:
                chamfer = map_detail.chamfer_vs_reference(r["ply_path"], ref)
            except Exception as exc:
                print(f"  [warn] chamfer {r['ply_path']}: {exc}")
        results_log.append_row(out_path, {**r, "chamfer_vs_ref": chamfer,
                                          "ref": ref})
    print(f"  map-detail → {out_path}  (reference: {ref_short}/{ref_res})")


if __name__ == "__main__":
    main()
