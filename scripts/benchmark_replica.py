"""
Benchmark DA3-SLAM on Replica indoor scenes.

Evaluates Absolute Trajectory Error (ATE) and Relative Pose Error (RPE)
against the camera ground truth provided by each Replica scene.

Dataset: https://github.com/cvg/nice-slam (Replica dataset)

Usage — single scene:
    python scripts/benchmark_replica.py \\
        --scene_dir data/Replica/office0 \\
        --out_dir outputs/benchmark/replica/office0

Usage — all scenes (summary table printed at the end):
    python scripts/benchmark_replica.py \\
        --scene_dir data/Replica/office0 data/Replica/office1 \\
                    data/Replica/room0 \\
        --out_dir outputs/benchmark/replica

Outputs per scene (inside <out_dir>/<scene_name>/):
    trajectory_est.txt      TUM-format estimated trajectory
    trajectory_gt.txt       GT subset matched to estimated keyframes
    results.json            all metrics
    trajectory_xy.png       top-down estimated vs GT comparison
    ate_errors.png          per-frame ATE over time

SLAM config knobs (same as run_slam.py):
    --config, --submap_size, --confidence_percentile,
    --no_loop_closure, --loop_distance_threshold, --max_frames
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from tum_eval_common import (
    SharedSLAM,
    load_groundtruth,
    associate,
    compute_ate,
    compute_rpe,
)


# ── formatting helpers ────────────────────────────────────────────────────────

def _m_str(v: float) -> str:
    return f"{v:.4f} m"


def _cm_str(v: float) -> str:
    return f"{v * 100:.2f} cm"


def _deg_str(d: float) -> str:
    return f"{d:.3f}°"


# ── Replica dataset helpers ──────────────────────────────────────────────────

def load_replica_images(
    scene_dir: Path, max_frames: int | None = None, fps: float = 30.0
) -> tuple[list[str], list[float]]:
    """
    Discover RGB frames in <scene_dir>/results/frame*.jpg (sorted by frame number).
    Returns (absolute_image_paths, synthetic_timestamps).

    Replica filenames are non-numeric (frame000000.jpg), so timestamps are
    synthesized as frame_idx / fps.
    """
    results_dir = scene_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"results/ directory not found in {scene_dir}")

    rgb_files = sorted(results_dir.glob("frame*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"No frame*.jpg found in {results_dir}")

    if max_frames:
        rgb_files = rgb_files[:max_frames]

    image_paths = [str(f) for f in rgb_files]
    timestamps = [i / fps for i in range(len(rgb_files))]
    return image_paths, timestamps


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    ate_result: dict,
    out_dir: Path,
    scene_name: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_pos = np.array([p[:3, 3] for p in gt_poses])
    est_pos = np.array([p[:3, 3] for p in est_poses])

    # Align estimated for visualisation
    T_align = np.array(ate_result["align_T"])
    est_h = np.hstack([est_pos, np.ones((len(est_pos), 1))])
    est_aligned = (T_align @ est_h.T).T[:, :3]

    # ── top-down XZ comparison ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], "g-", linewidth=1.5, label="Ground truth")
    ax.plot(est_aligned[:, 0], est_aligned[:, 2], "b--", linewidth=1.5,
            label="DA3-SLAM (aligned)")
    ax.scatter(gt_pos[0, 0], gt_pos[0, 2], color="green", s=80, zorder=5)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 2], color="darkgreen", s=80,
               marker="*", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{scene_name} — top-down view  ATE RMSE "
                 f"{ate_result['rmse']*100:.1f} cm")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "trajectory_xy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ATE per frame ─────────────────────────────────────────────────────────
    errors = np.array(ate_result["per_frame_errors"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors * 100, linewidth=1.2, color="royalblue")
    ax.axhline(ate_result["rmse"] * 100, color="red", linestyle="--",
               linewidth=1, label=f"RMSE = {ate_result['rmse']*100:.1f} cm")
    ax.axhline(ate_result["mean"] * 100, color="orange", linestyle=":",
               linewidth=1, label=f"Mean = {ate_result['mean']*100:.1f} cm")
    ax.set_xlabel("Keyframe index")
    ax.set_ylabel("ATE (cm)")
    ax.set_title(f"{scene_name} — ATE over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "ate_errors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ── single-scene benchmark ────────────────────────────────────────────────────

def build_config(args: argparse.Namespace):
    """Build the SLAMConfig from YAML + CLI overrides."""
    from da3_slam.config import load_slam_config

    config = load_slam_config(
        args.config,
        submap_size=args.submap_size,
        confidence_percentile=args.confidence_percentile,
        depth_model=args.depth_model,
        depth_model_resolution=args.depth_model_resolution,
    )
    if args.no_loop_closure:
        config.enable_loop_closure = False
    if args.loop_distance_threshold is not None:
        config.loop_closure.distance_threshold = args.loop_distance_threshold
    if args.min_disparity_fraction is not None:
        config.keyframe.min_disparity_fraction = args.min_disparity_fraction
    return config


def benchmark_scene(slam, scene_dir: Path, out_dir: Path, args) -> dict | None:
    """
    Run the full benchmark for one Replica scene.
    Returns the metrics dict, or None if the scene could not be evaluated.
    """
    scene_name = scene_dir.name
    fps = args.fps

    gt_txt = scene_dir / "gt_tum.txt"
    if not gt_txt.exists():
        print(f"  [SKIP] {scene_name}: gt_tum.txt not found in {scene_dir}")
        return None

    image_paths, timestamps = load_replica_images(scene_dir, args.max_frames, fps)
    print(f"\n  {len(image_paths)} RGB frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s  @ {fps} fps)")

    result = slam.run(image_paths)

    # Map seq_idx → synthetic timestamp for saving and association
    ts_map = {i: ts for i, ts in enumerate(timestamps)}

    # ── save estimated trajectory with synthetic timestamps ───────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    est_tum = str(out_dir / "trajectory_tum.txt")
    result.save_tum(est_tum, timestamps=ts_map)
    print(f"  Saved estimated trajectory → {est_tum}")

    # ── associate estimated keyframes with ground truth ───────────────────────
    est_ts_to_pose = {
        ts_map[seq_idx]: pose
        for seq_idx, pose in result.keyframe_poses.items()
        if seq_idx in ts_map
    }
    gt_all = load_groundtruth(gt_txt)
    gt_stamps = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())

    # Replica GT timestamps are at multiples of 1/fps (0, 1/30, 2/30, ...)
    # Use max_diff = 0.5 * (1/fps) to allow half-frame tolerance
    max_diff = 0.5 / fps
    pairs = associate(est_stamps, gt_stamps, max_diff=max_diff)
    n_matched = len(pairs)
    print(f"  Matched {n_matched}/{len(est_stamps)} keyframes to GT "
          f"(max Δt={max_diff*1000:.1f}ms)")

    if n_matched < 3:
        print("  [SKIP] too few matched poses for evaluation")
        return None

    gt_poses_matched = [gt_all[ib][1] for _, ib in pairs]
    est_poses_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

    # ── save matched GT subset ────────────────────────────────────────────────
    from scipy.spatial.transform import Rotation
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

    # ── compute metrics ───────────────────────────────────────────────────────
    ate_se3 = compute_ate(gt_poses_matched, est_poses_matched, align="se3")
    ate_sim3 = compute_ate(gt_poses_matched, est_poses_matched, align="sim3")
    rpe_1 = compute_rpe(gt_poses_matched, est_poses_matched, delta=1)
    rpe_n = compute_rpe(gt_poses_matched, est_poses_matched,
                        delta=max(1, n_matched // 8))

    # ── print results ─────────────────────────────────────────────────────────
    print("\n  ┌─ ATE (SE3 alignment) ──────────────────────────────────┐")
    print(f"  │  RMSE   {_m_str(ate_se3['rmse']):<14}  Mean   {_m_str(ate_se3['mean']):<14}  │")
    print(f"  │  Median {_m_str(ate_se3['median']):<14}  Max    {_m_str(ate_se3['max']):<14}  │")
    print(f"  ├─ ATE (Sim3 alignment, scale={ate_sim3['scale']:.4f}) ───────────────┤")
    print(f"  │  RMSE   {_m_str(ate_sim3['rmse']):<14}  Mean   {_m_str(ate_sim3['mean']):<14}  │")
    print("  ├─ RPE  δ=1 frame ─────────────────────────────────────────┤")
    print(f"  │  Trans  {_cm_str(rpe_1['trans_rmse']):<14}  Rot    {_deg_str(rpe_1['rot_rmse_deg']):<14}  │")
    print(f"  ├─ RPE  δ={rpe_n['delta']} frames ────────────────────────────────────┤")
    print(f"  │  Trans  {_cm_str(rpe_n['trans_rmse']):<14}  Rot    {_deg_str(rpe_n['rot_rmse_deg']):<14}  │")
    print("  └───────────────────────────────────────────────────────────┘")
    print(f"  Keyframes: {result.n_keyframes}  Submaps: {len(result.submaps)}  "
          f"Loop closures: {len(result.loop_closures)}")

    # ── save full results ─────────────────────────────────────────────────────
    n_submaps_real = len([s for s in result.submaps if not s.is_lc_submap])
    metrics = {
        "sequence": scene_name,
        "fps": fps,
        "n_frames": len(image_paths),
        "n_keyframes": result.n_keyframes,
        "n_submaps": n_submaps_real,
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

    try:
        plot_results(gt_poses_matched, est_poses_matched, ate_se3, out_dir,
                     scene_name)
    except Exception as e:
        print(f"  [warn] Plotting failed: {e}")

    return metrics


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(all_metrics: list[dict]) -> None:
    if not all_metrics:
        return
    names = [m["sequence"] for m in all_metrics]
    col = max(len(n) for n in names) + 2

    header = (f"{'Scene':<{col}}  "
              f"{'ATE RMSE (m)':>13}  {'ATE Sim3 (m)':>13}  "
              f"{'RPE-t (m)':>10}  {'RPE-r (°)':>10}  {'KFs':>5}  {'LCs':>4}")
    sep = "═" * (len(header) + 2)
    print(f"\n{sep}")
    print("  BENCHMARK SUMMARY  —  Replica indoor scenes")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))
    ate_values = []
    for m, name in zip(all_metrics, names):
        ate = m["ate_se3"]["rmse"]
        ate_values.append(ate)
        print(f"  {name:<{col}}  "
              f"{ate:>13.4f}  {m['ate_sim3']['rmse']:>13.4f}  "
              f"{m['rpe_delta1']['trans_rmse']:>10.4f}  "
              f"{m['rpe_delta1']['rot_rmse_deg']:>10.3f}  "
              f"{m['n_keyframes']:>5}  {m['n_loop_closures']:>4}")

    avg = sum(ate_values) / len(ate_values)
    print("  " + "─" * len(header))
    print(f"  {'avg':<{col}}  {avg:>13.4f}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    from da3_slam.config import DEFAULT_YAML

    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on Replica indoor scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene_dir", nargs="+", required=True,
                        help="Path(s) to Replica scene directory/directories "
                             "(e.g. data/Replica/office0)")
    parser.add_argument("--out_dir", default="outputs/benchmark_replica",
                        help="Root output directory")
    parser.add_argument("--config", default=str(DEFAULT_YAML),
                        help="YAML config file")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap frames per scene (for quick tests)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Replica frame rate (default: 30.0)")

    # SLAM overrides (None = use the YAML value)
    parser.add_argument("--submap_size", type=int, default=None)
    parser.add_argument("--confidence_percentile", type=float, default=None)
    parser.add_argument("--min_disparity_fraction", type=float, default=None)
    parser.add_argument("--no_loop_closure", action="store_true")
    parser.add_argument("--loop_distance_threshold", "--loop_threshold",
                        type=float, default=None,
                        help="DINO-SALAD descriptor L2 distance threshold "
                             "(lower = stricter)")
    parser.add_argument("--depth_model", default=None)
    parser.add_argument("--depth_model_resolution", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The DA3 model is loaded once; SharedSLAM.run() resets the loop-closure
    # detector before each scene so descriptors never leak across scenes.
    slam = SharedSLAM(build_config(args))

    all_metrics = []
    for scene_path in args.scene_dir:
        scene_dir = Path(scene_path)
        scene_name = scene_dir.name
        out_dir = Path(args.out_dir) / scene_name

        print(f"\n{'═'*60}")
        print(f"  Scene:    {scene_name}")
        print(f"  Input:    {scene_dir}")
        print(f"  Output:   {out_dir}")
        print(f"{'═'*60}")

        try:
            m = benchmark_scene(slam, scene_dir, out_dir, args)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            print(f"\n  [ERROR] {scene_name}: {e}")
            traceback.print_exc()

    if len(all_metrics) > 1:
        print_summary(all_metrics)

    if all_metrics:
        agg_path = Path(args.out_dir) / "benchmark_summary.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAggregate results saved → {agg_path}")


if __name__ == "__main__":
    main()
