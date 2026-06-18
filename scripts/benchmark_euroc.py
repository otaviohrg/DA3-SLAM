"""
Benchmark DA3-SLAM on EuRoC MAV sequences.

Evaluates Absolute Trajectory Error (ATE) and Relative Pose Error (RPE)
against the Vicon/Leica ground truth shipped with each EuRoC sequence.
Uses the left camera (cam0) as the monocular input.

Dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

EuRoC sequences are distributed in the ASL format:

    <seq>/mav0/cam0/data/<timestamp_ns>.png      left grayscale frames
    <seq>/mav0/cam0/data.csv                      timestamp,filename index
    <seq>/mav0/cam0/sensor.yaml                   intrinsics, distortion, T_BS
    <seq>/mav0/state_groundtruth_estimate0/data.csv   body pose @ ~200 Hz

Three EuRoC-specific details handled here (vs. the TUM benchmark):

  1. Ground truth is the **IMU/body** pose in the world frame, with a
     (w, x, y, z) quaternion.  DA3-SLAM estimates the **camera** trajectory,
     so GT is converted to the camera frame with the body→camera extrinsic
     T_BS from cam0/sensor.yaml:  T_WC = T_WB @ T_BS.  A constant body↔camera
     offset does not commute with the single global ATE alignment, so this
     conversion is required for a fair comparison.

  2. Timestamps are integer nanoseconds; they are converted to seconds so the
     shared association / metric code (scripts/tum_eval_common.py) applies
     unchanged.

  3. EuRoC frames carry significant radial-tangential distortion, but DA3
     assumes a pinhole model.  Frames are undistorted by default (cv2.undistort
     with the sensor.yaml intrinsics); pass --no_undistort to skip.

DA3-SLAM is monocular, so its trajectory has an arbitrary global scale.
The meaningful accuracy number is therefore the **Sim3 ATE**.

Usage — single sequence (point --seq_dir at the directory containing mav0,
or any ancestor of it):
    python scripts/benchmark_euroc.py \\
        --seq_dir data/EuRoC/vicon_room2/V2_01_easy \\
        --out_dir outputs/euroc/V2_01_easy

Usage — multiple sequences (summary table at the end):
    python scripts/benchmark_euroc.py \\
        --seq_dir data/EuRoC/vicon_room1/V1_01_easy \\
                  data/EuRoC/vicon_room2/V2_01_easy \\
        --out_dir outputs/euroc

SLAM config knobs (same as run_slam.py):
    --config, --submap_size, --confidence_percentile,
    --no_loop_closure, --loop_distance_threshold, --max_frames
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from scipy.spatial.transform import Rotation

from euroc_common import (
    find_mav0,
    load_camera_calibration,
    load_euroc_images,
    load_euroc_groundtruth,
    undistort_images,
)
from tum_eval_common import (
    SharedSLAM,
    associate,
    compute_ate,
    compute_rpe,
)
# plot_results and the metric-formatting helpers are dataset-agnostic.
from benchmark_tum import plot_results, _m_str, _cm_str, _deg_str


# ── single-sequence benchmark ─────────────────────────────────────────────────

def benchmark_sequence(slam, seq_dir: Path, out_dir: Path, args) -> dict | None:
    """Run SLAM on one EuRoC sequence and score it against ground truth.

    Returns the metrics dict, or None if the sequence could not be evaluated.
    """
    seq_name = seq_dir.name

    mav0 = find_mav0(seq_dir)
    if mav0 is None:
        print(f"  [SKIP] {seq_name}: no extracted mav0/ found under {seq_dir} "
              f"(is the sequence still a .zip?)")
        return None

    calibration = load_camera_calibration(mav0, cam=args.cam)
    image_paths, timestamps = load_euroc_images(mav0, cam=args.cam,
                                                max_frames=args.max_frames)
    if not image_paths:
        print(f"  [SKIP] {seq_name}: no images listed in {args.cam}/data.csv")
        return None
    print(f"\n  {len(image_paths)} {args.cam} frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.undistort:
        image_paths = undistort_images(
            image_paths, calibration, out_dir / f"undistorted_{args.cam}"
        )

    result = slam.run(image_paths)

    ts_map = {i: ts for i, ts in enumerate(timestamps)}

    # ── save estimated trajectory (real timestamps, seconds) ──────────────────
    est_tum = str(out_dir / "trajectory_est.txt")
    result.save_tum(est_tum, timestamps=ts_map)
    print(f"  Saved estimated trajectory → {est_tum}")

    # ── associate estimated keyframes with ground truth ───────────────────────
    est_ts_to_pose = {
        ts_map[seq_idx]: pose
        for seq_idx, pose in result.keyframe_poses.items()
        if seq_idx in ts_map
    }
    gt_all = load_euroc_groundtruth(mav0, calibration["T_BS"])
    gt_stamps = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())

    pairs = associate(est_stamps, gt_stamps, max_diff=0.02)
    n_matched = len(pairs)
    print(f"  Matched {n_matched}/{len(est_stamps)} keyframes to GT (max Δt=20ms)")

    if n_matched < 3:
        print("  [SKIP] too few matched poses for evaluation")
        return None

    gt_poses_matched = [gt_all[ib][1] for _, ib in pairs]
    est_poses_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

    # ── save matched GT subset (camera frame, TUM format) ─────────────────────
    gt_tum = str(out_dir / "trajectory_gt.txt")
    with open(gt_tum, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ia, ib in pairs:
            ts = est_stamps[ia]
            T = gt_all[ib][1]
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.6f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")

    # ── metrics ───────────────────────────────────────────────────────────────
    ate_se3 = compute_ate(gt_poses_matched, est_poses_matched, align="se3")
    ate_sim3 = compute_ate(gt_poses_matched, est_poses_matched, align="sim3")
    rpe_1 = compute_rpe(gt_poses_matched, est_poses_matched, delta=1)
    rpe_n = compute_rpe(gt_poses_matched, est_poses_matched,
                        delta=max(1, n_matched // 8))

    print("\n  ┌─ ATE (Sim3 alignment — the monocular metric) ──────────┐")
    print(f"  │  RMSE   {_m_str(ate_sim3['rmse']):<14}  Mean   {_m_str(ate_sim3['mean']):<14}  │")
    print(f"  │  Median {_m_str(ate_sim3['median']):<14}  scale  {ate_sim3['scale']:<14.4f}  │")
    print("  ├─ ATE (SE3 alignment, no scale) ──────────────────────────┤")
    print(f"  │  RMSE   {_m_str(ate_se3['rmse']):<14}  Mean   {_m_str(ate_se3['mean']):<14}  │")
    print("  ├─ RPE  δ=1 frame ─────────────────────────────────────────┤")
    print(f"  │  Trans  {_cm_str(rpe_1['trans_rmse']):<14}  Rot    {_deg_str(rpe_1['rot_rmse_deg']):<14}  │")
    print(f"  ├─ RPE  δ={rpe_n['delta']} frames ────────────────────────────────────┤")
    print(f"  │  Trans  {_cm_str(rpe_n['trans_rmse']):<14}  Rot    {_deg_str(rpe_n['rot_rmse_deg']):<14}  │")
    print("  └───────────────────────────────────────────────────────────┘")
    print(f"  Keyframes: {result.n_keyframes}  Submaps: {len(result.submaps)}  "
          f"Loop closures: {len(result.loop_closures)}")

    metrics = {
        "sequence": seq_name,
        "n_frames": len(image_paths),
        "n_keyframes": result.n_keyframes,
        "n_submaps": len(result.submaps),
        "n_loop_closures": len(result.loop_closures),
        "n_matched_gt": n_matched,
        "ate_se3": {k: v for k, v in ate_se3.items()
                    if k not in ("per_frame_errors", "align_T")},
        "ate_sim3": {k: v for k, v in ate_sim3.items()
                     if k not in ("per_frame_errors", "align_T")},
        "rpe_delta1": {k: v for k, v in rpe_1.items()
                       if k not in ("per_frame_trans", "per_frame_rot")},
        "rpe_delta_n": {k: v for k, v in rpe_n.items()
                        if k not in ("per_frame_trans", "per_frame_rot")},
        "timings": result.timings,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved → {out_dir}/results.json")

    # Plot against the Sim3-aligned trajectory (the meaningful one for monocular).
    try:
        plot_results(gt_poses_matched, est_poses_matched, ate_sim3, out_dir, seq_name)
    except Exception as e:
        print(f"  [warn] Plotting failed: {e}")

    return metrics


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(all_metrics: list[dict]) -> None:
    if not all_metrics:
        return
    names = [m["sequence"] for m in all_metrics]
    col = max(len(n) for n in names) + 2

    header = (f"{'Sequence':<{col}}  "
              f"{'ATE Sim3 (m)':>13}  {'ATE SE3 (m)':>13}  "
              f"{'RPE-t (m)':>10}  {'RPE-r (°)':>10}  {'KFs':>5}  {'LCs':>4}")
    sep = "═" * (len(header) + 2)
    print(f"\n{sep}")
    print("  EuRoC BENCHMARK SUMMARY  —  ATE RMSE (Sim3 = monocular metric)")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))
    sim3_values = []
    for m, name in zip(all_metrics, names):
        sim3 = m["ate_sim3"]["rmse"]
        sim3_values.append(sim3)
        print(f"  {name:<{col}}  "
              f"{sim3:>13.4f}  {m['ate_se3']['rmse']:>13.4f}  "
              f"{m['rpe_delta1']['trans_rmse']:>10.4f}  "
              f"{m['rpe_delta1']['rot_rmse_deg']:>10.3f}  "
              f"{m['n_keyframes']:>5}  {m['n_loop_closures']:>4}")

    print("  " + "─" * len(header))
    print(f"  {'avg':<{col}}  {sum(sim3_values) / len(sim3_values):>13.4f}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace):
    """Build the SLAMConfig from YAML + CLI overrides (mirrors benchmark_tum)."""
    from da3_slam.config import load_slam_config

    config = load_slam_config(
        args.config,
        submap_size=args.submap_size,
        confidence_percentile=args.confidence_percentile,
        depth_model=args.depth_model,
        depth_model_resolution=args.depth_model_resolution,
        boundary_scale_damping=args.boundary_scale_damping,
        boundary_scale_clamp=args.boundary_scale_clamp,
    )
    if args.no_loop_closure:
        config.enable_loop_closure = False
    if args.loop_distance_threshold is not None:
        config.loop_closure.distance_threshold = args.loop_distance_threshold
    if args.min_submaps_apart is not None:
        config.loop_closure.min_submaps_apart = args.min_submaps_apart
    if args.max_loop_closures is not None:
        config.loop_closure.max_loop_closures = args.max_loop_closures
    if args.min_disparity_fraction is not None:
        config.keyframe.min_disparity_fraction = args.min_disparity_fraction
    return config


def parse_args() -> argparse.Namespace:
    from da3_slam.config import DEFAULT_YAML

    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on EuRoC MAV sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="EuRoC sequence directory/directories "
                             "(the mav0 parent, or any ancestor of it)")
    parser.add_argument("--out_dir", default="outputs/euroc",
                        help="Root output directory")
    parser.add_argument("--config", default=str(DEFAULT_YAML),
                        help="YAML config file")
    parser.add_argument("--cam", default="cam0",
                        help="Which camera to use (cam0 = left)")
    parser.add_argument("--no_undistort", dest="undistort", action="store_false",
                        help="Skip radial-tangential undistortion of the frames")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap frames per sequence (for quick tests)")

    # SLAM overrides (None = use the YAML value)
    parser.add_argument("--submap_size", type=int, default=None)
    parser.add_argument("--confidence_percentile", type=float, default=None)
    parser.add_argument("--min_disparity_fraction", type=float, default=None)
    parser.add_argument("--no_loop_closure", action="store_true")
    parser.add_argument("--loop_distance_threshold", "--loop_threshold",
                        type=float, default=None,
                        help="DINO-SALAD descriptor L2 distance threshold "
                             "(lower = stricter)")
    parser.add_argument("--min_submaps_apart", type=int, default=None,
                        help="Min submap index gap for loop-closure candidates")
    parser.add_argument("--max_loop_closures", type=int, default=None,
                        help="Max verified loop closures kept per submap")
    parser.add_argument("--boundary_scale_damping", type=float, default=None,
                        help="Damping g for inter-submap scale chaining: each "
                             "boundary depth-ratio is raised to (1-g). "
                             "0 = full chaining, 1 = trust DA3 metric depth")
    parser.add_argument("--boundary_scale_clamp", type=float, default=None,
                        help="Clamp each boundary scale ratio to [1/c, c] "
                             "(unset = no clamping)")
    parser.add_argument("--depth_model", default=None)
    parser.add_argument("--depth_model_resolution", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The DA3 model is loaded once; SharedSLAM.run() resets the loop-closure
    # detector before each sequence so descriptors never leak across sequences.
    slam = SharedSLAM(build_config(args))

    all_metrics = []
    for seq_path in args.seq_dir:
        seq_dir = Path(seq_path)
        out_dir = Path(args.out_dir) / seq_dir.name

        print(f"\n{'═'*60}")
        print(f"  Sequence: {seq_dir.name}")
        print(f"  Input:    {seq_dir}")
        print(f"  Output:   {out_dir}")
        print(f"{'═'*60}")

        try:
            m = benchmark_sequence(slam, seq_dir, out_dir, args)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            print(f"\n  [ERROR] {seq_dir.name}: {e}")
            traceback.print_exc()

    if len(all_metrics) > 1:
        print_summary(all_metrics)

    if all_metrics:
        agg_path = Path(args.out_dir) / "euroc_summary.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAggregate results saved → {agg_path}")


if __name__ == "__main__":
    main()
