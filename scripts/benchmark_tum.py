"""
Benchmark DA3-SLAM on TUM RGB-D sequences.

Evaluates Absolute Trajectory Error (ATE) and Relative Pose Error (RPE)
against the motion-capture ground truth provided by each TUM sequence.

Dataset: https://cvg.cit.tum.de/data/datasets/rgbd-dataset

Usage — single sequence:
    python scripts/benchmark_tum.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_xyz \\
        --out_dir outputs/benchmark/fr1_xyz

Usage — multiple sequences (summary table printed at the end):
    python scripts/benchmark_tum.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_xyz \\
                  data/tum/rgbd_dataset_freiburg1_desk \\
                  data/tum/rgbd_dataset_freiburg2_xyz \\
        --out_dir outputs/benchmark

Outputs per sequence (inside <out_dir>/<seq_name>/):
    trajectory_est.txt      TUM-format estimated trajectory (real timestamps)
    trajectory_gt.txt       GT subset matched to estimated keyframes
    results.json            all metrics
    trajectory_xy.png       top-down estimated vs GT comparison
    ate_errors.png          per-frame ATE over time

SLAM config knobs (same as run_slam.py):
    --config, --submap_size, --confidence_percentile,
    --no_loop_closure, --loop_threshold, --max_frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── formatting helpers ────────────────────────────────────────────────────────

def _m_str(v: float) -> str:
    return f"{v:.4f} m"

def _cm_str(v: float) -> str:
    return f"{v * 100:.2f} cm"

def _deg_str(d: float) -> str:
    return f"{d:.3f}°"


# ── TUM dataset helpers ───────────────────────────────────────────────────────

def parse_tum_file(path: str | Path) -> list[tuple[float, str]]:
    """
    Parse an association file (rgb.txt, depth.txt, groundtruth.txt).
    Returns a list of (timestamp, value) sorted by timestamp.
    Lines starting with '#' are skipped.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append((float(parts[0]), " ".join(parts[1:])))
    return sorted(entries, key=lambda x: x[0])


def load_groundtruth(gt_path: str | Path) -> list[tuple[float, np.ndarray]]:
    """
    Load groundtruth.txt.
    Returns list of (timestamp, T_4x4) sorted by timestamp,
    where T is the cam-to-world pose.
    """
    from scipy.spatial.transform import Rotation
    poses = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses.append((ts, T))
    return sorted(poses, key=lambda x: x[0])


def associate(
    stamps_a: list[float],
    stamps_b: list[float],
    max_diff: float = 0.02,
) -> list[tuple[int, int]]:
    """
    Associate two timestamp lists by nearest match.
    Returns list of (idx_a, idx_b) pairs where |t_a - t_b| <= max_diff.
    Each index from list_a is matched to at most one index in list_b.
    """
    pairs = []
    b_arr = np.array(stamps_b)
    used_b: set[int] = set()
    for ia, ta in enumerate(stamps_a):
        diffs = np.abs(b_arr - ta)
        ib = int(diffs.argmin())
        if diffs[ib] <= max_diff and ib not in used_b:
            pairs.append((ia, ib))
            used_b.add(ib)
    return pairs


# ── alignment & metrics ───────────────────────────────────────────────────────

def se3_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Rigid SE(3) alignment of two (N, 3) position sets.
    Minimises sum ||dst_i - (R @ src_i + t)||^2.
    Returns (4, 4) transform T such that dst ≈ T @ src_homogeneous.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    A = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(A)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def sim3_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Sim(3) alignment of two (N, 3) position sets (SE(3) + scale).
    Returns (4, 4) transform (scale is folded into the rotation block).
    Useful for evaluating monocular / scale-ambiguous systems.
    """
    n = len(src)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    var_s = np.mean(np.sum((src - mu_s) ** 2, axis=1))
    H = (dst - mu_d).T @ (src - mu_s) / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = float(np.sum(D * S.diagonal()) / var_s)
    t = mu_d - scale * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def compute_ate(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    align: str = "se3",
) -> dict:
    """
    Compute Absolute Trajectory Error (ATE) after alignment.

    Args:
        gt_poses:  list of (4,4) GT cam-to-world poses
        est_poses: list of (4,4) estimated cam-to-world poses (same length)
        align:     "se3" | "sim3" | "none"

    Returns:
        dict with keys: rmse, mean, median, std, max, n_pairs, scale (sim3 only)
    """
    gt_pos  = np.array([p[:3, 3] for p in gt_poses])
    est_pos = np.array([p[:3, 3] for p in est_poses])

    scale = 1.0
    if align == "se3":
        T_align = se3_align(est_pos, gt_pos)
    elif align == "sim3":
        T_align = sim3_align(est_pos, gt_pos)
        scale = float(np.linalg.det(T_align[:3, :3]) ** (1 / 3))
    else:
        T_align = np.eye(4)

    # Apply alignment to estimated positions
    est_h = np.hstack([est_pos, np.ones((len(est_pos), 1))])
    est_aligned = (T_align @ est_h.T).T[:, :3]

    errors = np.linalg.norm(gt_pos - est_aligned, axis=1)
    return {
        "rmse":   float(np.sqrt((errors ** 2).mean())),
        "mean":   float(errors.mean()),
        "median": float(np.median(errors)),
        "std":    float(errors.std()),
        "max":    float(errors.max()),
        "n_pairs": len(errors),
        "scale":  scale,
        "per_frame_errors": errors.tolist(),
        "align_T": T_align.tolist(),
    }


def compute_rpe(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    delta: int = 1,
) -> dict:
    """
    Compute Relative Pose Error (RPE) with a fixed frame delta.

    For each pair (i, i+delta):
        Q = inv(gt_i)  @ gt_{i+delta}     (relative GT)
        P = inv(est_i) @ est_{i+delta}    (relative estimated)
        E = inv(Q) @ P                    (error transform)
        trans_err = ||E[:3, 3]||
        rot_err   = arccos((trace(E[:3,:3]) - 1) / 2)  in degrees

    Returns:
        dict with trans_rmse, trans_mean, rot_rmse_deg, rot_mean_deg, n_pairs
    """
    trans_errors = []
    rot_errors_deg = []

    for i in range(len(gt_poses) - delta):
        Q = np.linalg.inv(gt_poses[i])  @ gt_poses[i + delta]
        P = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        E = np.linalg.inv(Q) @ P

        trans_errors.append(float(np.linalg.norm(E[:3, 3])))
        cos_angle = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors_deg.append(float(np.degrees(np.arccos(cos_angle))))

    if not trans_errors:
        return {"n_pairs": 0}

    t = np.array(trans_errors)
    r = np.array(rot_errors_deg)
    return {
        "delta":         delta,
        "trans_rmse":    float(np.sqrt((t ** 2).mean())),
        "trans_mean":    float(t.mean()),
        "trans_median":  float(np.median(t)),
        "trans_max":     float(t.max()),
        "rot_rmse_deg":  float(np.sqrt((r ** 2).mean())),
        "rot_mean_deg":  float(r.mean()),
        "rot_median_deg":float(np.median(r)),
        "n_pairs":       len(t),
        "per_frame_trans": t.tolist(),
        "per_frame_rot":   r.tolist(),
    }


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    gt_poses: list[np.ndarray],
    est_poses: list[np.ndarray],
    ate_result: dict,
    out_dir: Path,
    seq_name: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_pos  = np.array([p[:3, 3] for p in gt_poses])
    est_pos = np.array([p[:3, 3] for p in est_poses])

    # Align estimated for visualisation
    T_align = np.array(ate_result["align_T"])
    est_h = np.hstack([est_pos, np.ones((len(est_pos), 1))])
    est_aligned = (T_align @ est_h.T).T[:, :3]

    # ── top-down XZ comparison ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(gt_pos[:, 0],       gt_pos[:, 2],       "g-",  linewidth=1.5, label="Ground truth")
    ax.plot(est_aligned[:, 0],  est_aligned[:, 2],  "b--", linewidth=1.5, label="DA3-SLAM (aligned)")
    ax.scatter(gt_pos[0, 0],  gt_pos[0, 2],  color="green", s=80, zorder=5)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 2], color="darkgreen", s=80, marker="*", zorder=5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title(f"{seq_name} — top-down view  ATE RMSE {ate_result['rmse']*100:.1f} cm")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "trajectory_xy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ATE per frame ─────────────────────────────────────────────────────────
    errors = np.array(ate_result["per_frame_errors"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors * 100, linewidth=1.2, color="royalblue")
    ax.axhline(ate_result["rmse"] * 100, color="red",    linestyle="--",
               linewidth=1, label=f"RMSE = {ate_result['rmse']*100:.1f} cm")
    ax.axhline(ate_result["mean"] * 100, color="orange", linestyle=":",
               linewidth=1, label=f"Mean = {ate_result['mean']*100:.1f} cm")
    ax.set_xlabel("Keyframe index"); ax.set_ylabel("ATE (cm)")
    ax.set_title(f"{seq_name} — ATE over time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "ate_errors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ── single-sequence benchmark ─────────────────────────────────────────────────

def benchmark_sequence(seq_dir: Path, out_dir: Path, args) -> dict | None:
    """
    Run the full benchmark for one TUM sequence.
    Returns the metrics dict, or None if the sequence could not be evaluated.
    """
    seq_name = seq_dir.name

    # ── validate sequence structure ───────────────────────────────────────────
    rgb_txt      = seq_dir / "rgb.txt"
    gt_txt       = seq_dir / "groundtruth.txt"
    if not rgb_txt.exists():
        print(f"  [SKIP] {seq_name}: rgb.txt not found in {seq_dir}")
        return None
    if not gt_txt.exists():
        print(f"  [SKIP] {seq_name}: groundtruth.txt not found in {seq_dir}")
        return None

    # ── load RGB frame list ───────────────────────────────────────────────────
    rgb_entries = parse_tum_file(rgb_txt)   # [(timestamp, rel_path), ...]
    image_paths = [str(seq_dir / entry[1]) for entry in rgb_entries]
    timestamps  = [entry[0] for entry in rgb_entries]

    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
        timestamps  = timestamps[: args.max_frames]

    print(f"\n  {len(image_paths)} RGB frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s)")

    # ── run SLAM ──────────────────────────────────────────────────────────────
    from da3_slam.slam import DA3SLAM
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
    if args.loop_threshold is not None:
        config.loop_closure.similarity_threshold = args.loop_threshold
    if args.min_disparity_fraction is not None:
        config.keyframe.min_disparity_fraction = args.min_disparity_fraction

    slam = DA3SLAM(config)
    result = slam.run(image_paths)

    # seq_idx → real timestamp mapping
    ts_map = {i: timestamps[i] for i in range(len(timestamps))}

    # ── save estimated trajectory with real timestamps ─────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    est_tum = str(out_dir / "trajectory_est.txt")
    result.save_tum(est_tum, timestamps=ts_map)
    print(f"  Saved estimated trajectory → {est_tum}")

    # ── build estimated pose list (keyed by real timestamp) ───────────────────
    est_ts_to_pose: dict[float, np.ndarray] = {}
    for seq_idx, pose in result.keyframe_poses.items():
        if seq_idx in ts_map:
            est_ts_to_pose[ts_map[seq_idx]] = pose

    # ── load and associate ground truth ───────────────────────────────────────
    gt_all = load_groundtruth(gt_txt)           # [(timestamp, T_4x4)]
    gt_stamps  = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())

    pairs = associate(est_stamps, gt_stamps, max_diff=0.02)
    n_matched = len(pairs)
    print(f"  Matched {n_matched}/{len(est_stamps)} keyframes to GT "
          f"(max Δt=20ms)")

    if n_matched < 3:
        print("  [SKIP] too few matched poses for evaluation")
        return None

    gt_poses_matched  = [gt_all[ib][1]                    for _, ib in pairs]
    est_poses_matched = [est_ts_to_pose[est_stamps[ia]]   for ia, _ in pairs]

    # ── save matched GT subset ────────────────────────────────────────────────
    from scipy.spatial.transform import Rotation
    gt_tum = str(out_dir / "trajectory_gt.txt")
    with open(gt_tum, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ia, ib in pairs:
            ts = est_stamps[ia]
            T  = gt_all[ib][1]
            t  = T[:3, 3]
            q  = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{ts:.6f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")

    # ── compute metrics ───────────────────────────────────────────────────────
    ate_se3  = compute_ate(gt_poses_matched, est_poses_matched, align="se3")
    ate_sim3 = compute_ate(gt_poses_matched, est_poses_matched, align="sim3")
    rpe_1    = compute_rpe(gt_poses_matched, est_poses_matched, delta=1)
    rpe_n    = compute_rpe(gt_poses_matched, est_poses_matched,
                           delta=max(1, n_matched // 8))

    # ── print results ─────────────────────────────────────────────────────────
    print(f"\n  ┌─ ATE (SE3 alignment) ──────────────────────────────────┐")
    print(f"  │  RMSE   {_m_str(ate_se3['rmse']):<14}  Mean   {_m_str(ate_se3['mean']):<14}  │")
    print(f"  │  Median {_m_str(ate_se3['median']):<14}  Max    {_m_str(ate_se3['max']):<14}  │")
    print(f"  ├─ ATE (Sim3 alignment, scale={ate_sim3['scale']:.4f}) ───────────────┤")
    print(f"  │  RMSE   {_m_str(ate_sim3['rmse']):<14}  Mean   {_m_str(ate_sim3['mean']):<14}  │")
    print(f"  ├─ RPE  δ=1 frame ─────────────────────────────────────────┤")
    print(f"  │  Trans  {_cm_str(rpe_1['trans_rmse']):<14}  Rot    {_deg_str(rpe_1['rot_rmse_deg']):<14}  │")
    print(f"  ├─ RPE  δ={rpe_n['delta']} frames ────────────────────────────────────┤")
    print(f"  │  Trans  {_cm_str(rpe_n['trans_rmse']):<14}  Rot    {_deg_str(rpe_n['rot_rmse_deg']):<14}  │")
    print(f"  └───────────────────────────────────────────────────────────┘")
    print(f"  Keyframes: {result.n_keyframes}  Submaps: {len(result.submaps)}  "
          f"Loop closures: {len(result.loop_closures)}")

    # ── save full results ─────────────────────────────────────────────────────
    metrics = {
        "sequence": seq_name,
        "n_frames": len(image_paths),
        "n_keyframes": result.n_keyframes,
        "n_submaps": len(result.submaps),
        "n_loop_closures": len(result.loop_closures),
        "n_matched_gt": n_matched,
        "ate_se3":  {k: v for k, v in ate_se3.items()
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

    # ── plots ─────────────────────────────────────────────────────────────────
    try:
        plot_results(gt_poses_matched, est_poses_matched, ate_se3, out_dir, seq_name)
    except Exception as e:
        print(f"  [warn] Plotting failed: {e}")

    return metrics


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(all_metrics: list[dict]) -> None:
    if not all_metrics:
        return
    # Strip common prefix for cleaner sequence names
    seqs = [m["sequence"] for m in all_metrics]
    prefix = "rgbd_dataset_freiburg1_"
    names  = [s[len(prefix):] if s.startswith(prefix) else s for s in seqs]
    col = max(len(n) for n in names) + 2

    header = (f"{'Sequence':<{col}}  "
              f"{'ATE RMSE (m)':>13}  {'ATE Sim3 (m)':>13}  "
              f"{'RPE-t (m)':>10}  {'RPE-r (°)':>10}  {'KFs':>5}  {'LCs':>4}")
    sep = "═" * (len(header) + 2)
    print(f"\n{sep}")
    print("  BENCHMARK SUMMARY  —  ATE RMSE of the Absolute Trajectory Error")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))
    ate_values = []
    for m, name in zip(all_metrics, names):
        ate  = m["ate_se3"]["rmse"]
        ate3 = m["ate_sim3"]["rmse"]
        rpt  = m["rpe_delta1"]["trans_rmse"]
        rpr  = m["rpe_delta1"]["rot_rmse_deg"]
        kfs  = m["n_keyframes"]
        lcs  = m["n_loop_closures"]
        ate_values.append(ate)
        print(f"  {name:<{col}}  "
              f"{ate:>13.4f}  {ate3:>13.4f}  "
              f"{rpt:>10.4f}  {rpr:>10.3f}  {kfs:>5}  {lcs:>4}")

    # Average row
    avg = sum(ate_values) / len(ate_values)
    print("  " + "─" * len(header))
    print(f"  {'avg':<{col}}  {avg:>13.4f}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    from da3_slam.config import load_slam_config, DEFAULT_YAML
    cfg = load_slam_config()
    lc  = cfg.loop_closure

    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on TUM RGB-D sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="Path(s) to TUM sequence directory/directories")
    parser.add_argument("--out_dir", default="/app/outputs/benchmark",
                        help="Root output directory")
    parser.add_argument("--config", default=str(DEFAULT_YAML),
                        help="YAML config file")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap frames per sequence (for quick tests)")

    # SLAM overrides
    parser.add_argument("--submap_size",       type=int,   default=None)
    parser.add_argument("--confidence_percentile",   type=float, default=None)
    parser.add_argument("--min_disparity_fraction",  type=float, default=None)
    parser.add_argument("--no_loop_closure",         action="store_true")
    parser.add_argument("--loop_threshold",          type=float, default=None)
    parser.add_argument("--depth_model",             default=None)
    parser.add_argument("--depth_model_resolution",  type=int,   default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_metrics = []
    for seq_path in args.seq_dir:
        seq_dir = Path(seq_path)
        seq_name = seq_dir.name
        out_dir  = Path(args.out_dir) / seq_name

        print(f"\n{'═'*60}")
        print(f"  Sequence: {seq_name}")
        print(f"  Input:    {seq_dir}")
        print(f"  Output:   {out_dir}")
        print(f"{'═'*60}")

        try:
            m = benchmark_sequence(seq_dir, out_dir, args)
            if m is not None:
                all_metrics.append(m)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {seq_name}: {e}")
            traceback.print_exc()

    if len(all_metrics) > 1:
        print_summary(all_metrics)

    # Save aggregate results
    if all_metrics:
        agg_path = Path(args.out_dir) / "benchmark_summary.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAggregate results saved → {agg_path}")


if __name__ == "__main__":
    main()
