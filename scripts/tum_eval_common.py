"""
Shared TUM RGB-D evaluation utilities.

Used by benchmark_tum.py, ablation_tum.py, and grid_search_tum.py:
  - TUM dataset parsing (rgb.txt, groundtruth.txt) and timestamp association
  - SE(3) / Sim(3) trajectory alignment and ATE / RPE metrics
  - SharedSLAM: reuses one loaded DA3 model across many configurations
  - evaluate_sequence(): run SLAM on one sequence and score it against GT

These scripts are run from the repo root as `python scripts/<name>.py`, so
Python puts the scripts/ directory on sys.path and `import tum_eval_common`
resolves without any path manipulation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


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


def load_rgb_list(
    seq_dir: Path, max_frames: int | None = None
) -> tuple[list[str], list[float]]:
    """Read rgb.txt: returns (absolute image paths, timestamps), optionally capped."""
    rgb_entries = parse_tum_file(seq_dir / "rgb.txt")
    image_paths = [str(seq_dir / entry[1]) for entry in rgb_entries]
    timestamps = [entry[0] for entry in rgb_entries]
    if max_frames:
        image_paths = image_paths[:max_frames]
        timestamps = timestamps[:max_frames]
    return image_paths, timestamps


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
            qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                              float(parts[6]), float(parts[7]))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
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
        dict with keys: rmse, mean, median, std, max, n_pairs, scale (sim3),
        per_frame_errors, align_T
    """
    gt_pos = np.array([p[:3, 3] for p in gt_poses])
    est_pos = np.array([p[:3, 3] for p in est_poses])

    scale = 1.0
    if align == "se3":
        T_align = se3_align(est_pos, gt_pos)
    elif align == "sim3":
        T_align = sim3_align(est_pos, gt_pos)
        scale = float(np.linalg.det(T_align[:3, :3]) ** (1 / 3))
    else:
        T_align = np.eye(4)

    est_h = np.hstack([est_pos, np.ones((len(est_pos), 1))])
    est_aligned = (T_align @ est_h.T).T[:, :3]

    errors = np.linalg.norm(gt_pos - est_aligned, axis=1)
    return {
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "std": float(errors.std()),
        "max": float(errors.max()),
        "n_pairs": len(errors),
        "scale": scale,
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
        Q = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
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
        "delta": delta,
        "trans_rmse": float(np.sqrt((t ** 2).mean())),
        "trans_mean": float(t.mean()),
        "trans_median": float(np.median(t)),
        "trans_max": float(t.max()),
        "rot_rmse_deg": float(np.sqrt((r ** 2).mean())),
        "rot_mean_deg": float(r.mean()),
        "rot_median_deg": float(np.median(r)),
        "n_pairs": len(t),
        "per_frame_trans": t.tolist(),
        "per_frame_rot": r.tolist(),
    }


# ── shared SLAM wrapper ───────────────────────────────────────────────────────

class SharedSLAM:
    """
    Loads the DA3 depth model once and reuses it across many configurations.
    Only the cheap components (SubmapBuilder, LoopClosureDetector) are rebuilt
    when the configuration changes.
    """

    def __init__(self, base_config) -> None:
        from da3_slam.slam import DA3SLAM
        self._slam = DA3SLAM(base_config)

    def reconfigure(self, config) -> None:
        """Swap in a new config without reloading the depth model."""
        from da3_slam.backend.inference.submap import SubmapBuilder

        self._slam.config = config
        self._slam.builder = SubmapBuilder(
            self._slam.estimator,
            confidence_percentile=config.confidence_percentile,
        )
        self._slam.detector = self._make_detector()

    def run(self, image_paths: list[str]):
        """Run SLAM, resetting the loop-closure detector between sequences
        so descriptors from a previous sequence don't produce cross-sequence
        loop closures with stale indices."""
        self._slam.detector = self._make_detector()
        return self._slam.run(image_paths)

    def _make_detector(self):
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector
        cfg = self._slam.config
        if not cfg.enable_loop_closure:
            return None
        return LoopClosureDetector(cfg.loop_closure, builder=self._slam.builder)


# ── per-sequence evaluation ───────────────────────────────────────────────────

def evaluate_sequence(
    run_slam: Callable[[list[str]], Any],
    seq_dir: Path,
    out_dir: Path,
    max_frames: int | None,
) -> dict | None:
    """
    Run SLAM on one TUM sequence and score it against ground truth.

    Args:
        run_slam:   callable mapping image paths → SLAMResult
                    (e.g. SharedSLAM.run or DA3SLAM.run)
        seq_dir:    TUM sequence directory (must contain rgb.txt + groundtruth.txt)
        out_dir:    where to write trajectory_est.txt
        max_frames: optional frame cap

    Returns:
        compact metrics dict, or None if the sequence could not be evaluated.
    """
    seq_name = seq_dir.name
    rgb_txt = seq_dir / "rgb.txt"
    gt_txt = seq_dir / "groundtruth.txt"
    if not rgb_txt.exists() or not gt_txt.exists():
        print(f"  [SKIP] {seq_name}: missing rgb.txt or groundtruth.txt")
        return None

    image_paths, timestamps = load_rgb_list(seq_dir, max_frames)

    t0 = time.time()
    result = run_slam(image_paths)
    wall = time.time() - t0

    ts_map = {i: ts for i, ts in enumerate(timestamps)}
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_tum(str(out_dir / "trajectory_est.txt"), timestamps=ts_map)

    est_ts_to_pose = {
        ts_map[seq_idx]: pose
        for seq_idx, pose in result.keyframe_poses.items()
        if seq_idx in ts_map
    }

    gt_all = load_groundtruth(gt_txt)
    gt_stamps = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())
    pairs = associate(est_stamps, gt_stamps, max_diff=0.02)

    if len(pairs) < 3:
        print(f"  [SKIP] {seq_name}: too few matched poses ({len(pairs)})")
        return None

    gt_matched = [gt_all[ib][1] for _, ib in pairs]
    est_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

    ate_se3 = compute_ate(gt_matched, est_matched, align="se3")
    ate_sim3 = compute_ate(gt_matched, est_matched, align="sim3")
    rpe_1 = compute_rpe(gt_matched, est_matched, delta=1)

    n_submaps_real = len([s for s in result.submaps if not s.is_lc_submap])

    return {
        "sequence": seq_name,
        "ate_se3_rmse": ate_se3["rmse"],
        "ate_sim3_rmse": ate_sim3["rmse"],
        "rpe_trans_rmse": rpe_1["trans_rmse"],
        "rpe_rot_rmse_deg": rpe_1["rot_rmse_deg"],
        "n_frames": len(image_paths),
        "n_keyframes": result.n_keyframes,
        "n_submaps": n_submaps_real,
        "n_loop_closures": len(result.loop_closures),
        "wall_seconds": round(wall, 1),
        "timings": result.timings,
    }


# ── cross-sequence aggregation ────────────────────────────────────────────────

TIMING_MODULES = ("keyframe_selection", "submap_building",
                  "graph_building", "loop_closure", "optimization")


def average_metrics(seq_metrics: list[dict]) -> dict:
    """Cross-sequence averages of the dicts produced by evaluate_sequence().

    Timing ratios (s/frame, s/keyframe, s/submap) are computed per sequence
    and then averaged, so longer sequences don't dominate the mean.
    """
    if not seq_metrics:
        return {}

    def mean(key: str) -> float:
        return float(np.mean([m[key] for m in seq_metrics]))

    module_ms_per_frame = {
        module: float(np.mean([
            m["timings"].get(module, 0.0) / max(m["n_frames"], 1) * 1000
            for m in seq_metrics
        ]))
        for module in TIMING_MODULES
    }

    return {
        "ate_se3_rmse": mean("ate_se3_rmse"),
        "ate_sim3_rmse": mean("ate_sim3_rmse"),
        "rpe_trans_rmse": mean("rpe_trans_rmse"),
        "rpe_rot_rmse_deg": mean("rpe_rot_rmse_deg"),
        "n_keyframes": mean("n_keyframes"),
        "n_submaps": mean("n_submaps"),
        "n_loop_closures": mean("n_loop_closures"),
        "wall_seconds": mean("wall_seconds"),
        "s_per_frame": float(np.mean(
            [m["wall_seconds"] / m["n_frames"] for m in seq_metrics]
        )),
        "s_per_kf": float(np.mean(
            [m["wall_seconds"] / m["n_keyframes"]
             for m in seq_metrics if m["n_keyframes"] > 0]
        )),
        "s_per_submap": float(np.mean(
            [m["wall_seconds"] / max(m["n_submaps"], 1) for m in seq_metrics]
        )),
        "module_ms_per_frame": module_ms_per_frame,
    }
