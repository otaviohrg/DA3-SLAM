"""
Shared SLAM benchmark standard (dataset-agnostic).

This module is the single source of truth for *how a SLAM trajectory is scored
and saved*, so that every system in the SLAM/ workspace (DA3-SLAM, VGGT-SLAM,
DROID-SLAM, LeanGate, DA3-Streaming, ScaRF-SLAM, ORB-SLAM3) produces directly
comparable numbers and identical output files.  It is copied verbatim into each
repo's `scripts/` directory; only the thin per-dataset benchmark scripts
(`benchmark_tum.py`, `benchmark_euroc.py`, `benchmark_replica.py`) differ, and
they differ *only* in the small adapter that runs that particular SLAM system.

It provides:

  * dataset I/O      TUM RGB-D, EuRoC MAV (ASL), Replica  — images + ground truth
  * camera calib     per-dataset intrinsics/distortion (for systems that need them)
  * association      nearest-timestamp matching of estimate ↔ ground truth
  * metrics          SE(3) / Sim(3) ATE, RPE (translation + rotation)
  * evaluation       evaluate_trajectory(): one call scores + saves everything
  * output           trajectory_est.txt, trajectory_gt.txt, results.json,
                     trajectory_xy.png, ate_errors.png  (+ aggregate summary json)

Dependencies: numpy, scipy, opencv-python, pyyaml, matplotlib (matplotlib only
imported lazily inside the plotting function so headless/CI use still works).

The metric definitions mirror DA3-SLAM's scripts/tum_eval_common.py exactly
(SE3/Sim3 Umeyama alignment, RMSE/mean/median/max ATE, fixed-delta RPE) so the
two are numerically interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Output-file schema
# ══════════════════════════════════════════════════════════════════════════════
#
# Per sequence, inside <out_dir>/<seq_name>/:
#     trajectory_est.txt   TUM-format estimated trajectory (real timestamps)
#     trajectory_gt.txt    GT subset matched to the estimated poses
#     results.json         all metrics + timings + counts
#     trajectory_xy.png    top-down estimated-vs-GT comparison
#     ate_errors.png       per-frame ATE over time
# Per run, inside <out_dir>/:
#     <dataset>_summary.json   list of every sequence's metrics dict
#
# Every results.json has the same top-level keys regardless of system/dataset:
#     system, dataset, sequence, n_frames, n_keyframes, n_submaps,
#     n_loop_closures, n_matched_gt, ate_se3, ate_sim3, rpe_delta1,
#     rpe_delta_n, timings
# `timings` always carries at least {"total_s", "fps"}; systems may add their own
# module breakdown keys on top.
# ══════════════════════════════════════════════════════════════════════════════


# ── formatting helpers ────────────────────────────────────────────────────────

def m_str(v: float) -> str:
    return f"{v:.4f} m"


def cm_str(v: float) -> str:
    return f"{v * 100:.2f} cm"


def deg_str(d: float) -> str:
    return f"{d:.3f}°"


# ══════════════════════════════════════════════════════════════════════════════
#  TUM RGB-D dataset I/O
# ══════════════════════════════════════════════════════════════════════════════

def parse_tum_file(path: str | Path) -> list[tuple[float, str]]:
    """Parse a TUM association file (rgb.txt, depth.txt, groundtruth.txt).

    Returns a list of (timestamp, rest-of-line) sorted by timestamp; comment
    lines starting with '#' are skipped.
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
    """Read rgb.txt → (absolute image paths, timestamps), optionally capped."""
    rgb_entries = parse_tum_file(seq_dir / "rgb.txt")
    image_paths = [str(seq_dir / entry[1]) for entry in rgb_entries]
    timestamps = [entry[0] for entry in rgb_entries]
    if max_frames:
        image_paths = image_paths[:max_frames]
        timestamps = timestamps[:max_frames]
    return image_paths, timestamps


def load_groundtruth(gt_path: str | Path) -> list[tuple[float, np.ndarray]]:
    """Load a TUM-format groundtruth.txt (timestamp tx ty tz qx qy qz qw).

    Returns list of (timestamp, 4x4 cam-to-world pose) sorted by timestamp.
    Used for TUM and Replica (whose gt_tum.txt uses the same layout).
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


# TUM camera intrinsics by Freiburg group (fx, fy, cx, cy, k1, k2, p1, p2, k3).
# Systems that consume calibration (DROID, ORB-SLAM3) look these up by sequence
# name; calibration-free systems (DA3-SLAM, VGGT-SLAM) ignore them.
TUM_INTRINSICS = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989,
                  0.262383, -0.953104, -0.005358, 0.002628, 1.163314),
    "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764,
                  0.231222, -0.784899, -0.003257, -0.000105, 0.917205),
    "freiburg3": (535.4, 539.2, 320.1, 247.6, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def tum_intrinsics(seq_name: str) -> tuple[float, ...]:
    """Return (fx, fy, cx, cy, k1, k2, p1, p2, k3) for a TUM sequence name."""
    for group, calib in TUM_INTRINSICS.items():
        if group in seq_name:
            return calib
    # Default to the freiburg1 camera if the group cannot be inferred.
    return TUM_INTRINSICS["freiburg1"]


# ══════════════════════════════════════════════════════════════════════════════
#  EuRoC MAV dataset I/O (ASL format)
# ══════════════════════════════════════════════════════════════════════════════
#
#   <seq>/mav0/cam0/data/<timestamp_ns>.png      left grayscale frames
#   <seq>/mav0/cam0/data.csv                      timestamp,filename index
#   <seq>/mav0/cam0/sensor.yaml                   intrinsics, distortion, T_BS
#   <seq>/mav0/state_groundtruth_estimate0/data.csv   body pose @ ~200 Hz
#
# Ground truth is the IMU/body pose in the world frame with a (w,x,y,z)
# quaternion; SLAM systems estimate the *camera* trajectory, so GT is converted
# to camera-to-world via T_WC = T_WB @ T_BS (T_BS from cam0/sensor.yaml).

def find_mav0(seq_dir: Path) -> Path | None:
    """Locate the `mav0` directory for a EuRoC sequence.

    Accepts a path that *is* the mav0 parent or any ancestor of it (EuRoC
    archives often unpack to a doubly-nested <seq>/<seq>/mav0 layout); macOS
    __MACOSX resource-fork copies are ignored.
    """
    direct = seq_dir / "mav0"
    if direct.is_dir():
        return direct
    candidates = [
        p for p in seq_dir.rglob("mav0")
        if p.is_dir() and "__MACOSX" not in p.parts
    ]
    return candidates[0] if candidates else None


def load_camera_calibration(mav0: Path, cam: str = "cam0") -> dict:
    """Read intrinsics, distortion coefficients and T_BS from cam sensor.yaml.

    Returns {"K": 3x3, "distortion": (4,) radtan [k1,k2,p1,p2], "T_BS": 4x4
    sensor(camera)→body}.
    """
    import yaml
    with open(mav0 / cam / "sensor.yaml") as f:
        sensor = yaml.safe_load(f)

    fu, fv, cu, cv = sensor["intrinsics"]
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.array(sensor["distortion_coefficients"], dtype=np.float64)
    T_BS = np.array(sensor["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return {"K": K, "distortion": distortion, "T_BS": T_BS}


def load_euroc_images(
    mav0: Path, cam: str = "cam0", max_frames: int | None = None,
) -> tuple[list[str], list[float]]:
    """Read cam/data.csv → (absolute image paths, timestamps in seconds)."""
    data_dir = mav0 / cam / "data"
    image_paths: list[str] = []
    timestamps: list[float] = []
    with open(mav0 / cam / "data.csv") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts_ns, filename = line.split(",")[:2]
            image_paths.append(str(data_dir / filename.strip()))
            timestamps.append(int(ts_ns) * 1e-9)

    order = np.argsort(timestamps)
    image_paths = [image_paths[i] for i in order]
    timestamps = [timestamps[i] for i in order]
    if max_frames:
        image_paths = image_paths[:max_frames]
        timestamps = timestamps[:max_frames]
    return image_paths, timestamps


def load_euroc_groundtruth(
    mav0: Path, T_BS: np.ndarray, seconds: bool = True,
) -> list[tuple[float, np.ndarray]]:
    """Read state_groundtruth_estimate0/data.csv as camera-to-world poses.

    The CSV stores the body pose T_WB (position + a w,x,y,z quaternion); the
    returned poses are camera-to-world T_WC = T_WB @ T_BS, sorted by timestamp.
    `seconds=True` returns timestamps in seconds (ns × 1e-9).
    """
    from scipy.spatial.transform import Rotation
    gt_csv = mav0 / "state_groundtruth_estimate0" / "data.csv"
    scale = 1e-9 if seconds else 1.0
    poses: list[tuple[float, np.ndarray]] = []
    with open(gt_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            ts = int(parts[0]) * scale
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = (float(parts[4]), float(parts[5]),
                              float(parts[6]), float(parts[7]))
            T_WB = np.eye(4, dtype=np.float64)
            T_WB[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T_WB[:3, 3] = [px, py, pz]
            poses.append((ts, T_WB @ T_BS))
    return sorted(poses, key=lambda x: x[0])


def undistort_images(
    image_paths: list[str], calibration: dict, cache_dir: Path,
) -> list[str]:
    """Undistort EuRoC frames with the pinhole+radtan model into a cache dir.

    Filenames are preserved (downstream timestamp parsing still works) and
    writes are idempotent (existing frames are reused).  Returns the new paths.
    """
    import cv2
    cache_dir.mkdir(parents=True, exist_ok=True)
    K, distortion = calibration["K"], calibration["distortion"]
    out_paths: list[str] = []
    written = 0
    for src in image_paths:
        dst = cache_dir / Path(src).name
        if not dst.exists():
            image = cv2.imread(src)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {src}")
            cv2.imwrite(str(dst), cv2.undistort(image, K, distortion))
            written += 1
        out_paths.append(str(dst))
    print(f"  {'Undistorted ' + str(written) if written else 'Using cached'} "
          f"frame(s) in {cache_dir}")
    return out_paths


# ══════════════════════════════════════════════════════════════════════════════
#  Replica dataset I/O
# ══════════════════════════════════════════════════════════════════════════════
#
#   <scene>/results/frame*.jpg        RGB frames (non-numeric names)
#   <scene>/gt_tum.txt                TUM-format camera ground truth
#
# Replica is rendered at a fixed FPS with no real timestamps, so timestamps are
# synthesised as frame_idx / fps.

# Standard Replica pinhole intrinsics (NICE-SLAM convention, 1200×680 frames).
REPLICA_INTRINSICS = (600.0, 600.0, 599.5, 339.5, 0.0, 0.0, 0.0, 0.0, 0.0)


def replica_intrinsics() -> tuple[float, ...]:
    """Return (fx, fy, cx, cy, k1, k2, p1, p2, k3) for Replica scenes."""
    return REPLICA_INTRINSICS


def load_replica_images(
    scene_dir: Path, max_frames: int | None = None, fps: float = 30.0,
) -> tuple[list[str], list[float]]:
    """Discover <scene>/results/frame*.jpg, synthesise timestamps = idx / fps."""
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


def load_7scenes_sequence(
    seq_dir: Path, max_frames: int | None = None, fps: float = 30.0,
) -> tuple[list[str], list[float], list[tuple[float, np.ndarray]]]:
    """Load one Microsoft 7-Scenes sequence: RGB frames + camera ground truth.

    Layout is <scene>/seq-XX/frame-NNNNNN.{color.png,depth.png,pose.txt}, where
    each pose.txt is a 4x4 CAMERA-TO-WORLD matrix in plain text — the same
    convention TUM ground truth uses, so no frame conversion is needed.

    7-Scenes ships no timestamps (Kinect at 30 Hz), so they are synthesised as
    frame_idx / fps exactly as for Replica; pair this with a half-frame
    association tolerance.

    Some frames carry a non-finite pose (the tracker lost the frame during
    capture).  Those are dropped from the GROUND TRUTH but the image is still
    returned: the SLAM system should process the full sequence, it simply
    cannot be scored at those frames.

    Returns (image_paths, timestamps, gt) with gt as [(timestamp, 4x4)].
    """
    rgb_files = sorted(seq_dir.glob("frame-*.color.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No frame-*.color.png found in {seq_dir}")
    if max_frames:
        rgb_files = rgb_files[:max_frames]

    image_paths, timestamps, gt = [], [], []
    n_bad = 0
    for i, rgb in enumerate(rgb_files):
        ts = i / fps
        image_paths.append(str(rgb))
        timestamps.append(ts)
        pose_file = rgb.with_name(rgb.name.replace(".color.png", ".pose.txt"))
        if not pose_file.exists():
            n_bad += 1
            continue
        T = np.loadtxt(pose_file)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            n_bad += 1
            continue
        gt.append((ts, T))
    if n_bad:
        print(f"  [7scenes] {n_bad} frame(s) without a usable pose — excluded "
              f"from GT, still fed to SLAM")
    return image_paths, timestamps, gt


# ══════════════════════════════════════════════════════════════════════════════
#  Association
# ══════════════════════════════════════════════════════════════════════════════

def associate(
    stamps_a: list[float], stamps_b: list[float], max_diff: float = 0.02,
) -> list[tuple[int, int]]:
    """Nearest-timestamp association.

    Returns (idx_a, idx_b) pairs where |t_a − t_b| ≤ max_diff; each b index is
    used at most once.
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


# ══════════════════════════════════════════════════════════════════════════════
#  Alignment & metrics
# ══════════════════════════════════════════════════════════════════════════════

def se3_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid SE(3) Umeyama alignment of two (N,3) point sets.

    Minimises Σ‖dst_i − (R·src_i + t)‖²; returns 4x4 T with dst ≈ T·src.
    """
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    A = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(A)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def sim3_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Sim(3) alignment (SE(3)+scale); scale is folded into the rotation block.

    Required for monocular / scale-ambiguous systems.
    """
    n = len(src)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    var_s = np.mean(np.sum((src - mu_s) ** 2, axis=1))
    H = (dst - mu_d).T @ (src - mu_s) / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = float(np.sum(D * S.diagonal()) / var_s)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = mu_d - scale * R @ mu_s
    return T


def _compute_ate_evo(gt_poses, est_poses, align: str) -> dict:
    """ATE via evo's own APE pipeline — the reference implementation.

    Mirrors `evo_ape tum <gt> <est>` (SE3, `-a`) and `-as` (Sim3).  Trajectories
    are already associated by the caller, so index timestamps are used and evo's
    own association is a no-op.

    Verified against the in-house path on all 46 7-Scenes sequences: agreement
    <= 5e-7 m on both Sim3 and SE3, which is evo's print precision.
    """
    from evo.core import metrics
    from evo.core.trajectory import PosePath3D

    gt = PosePath3D(poses_se3=[np.asarray(p, dtype=np.float64) for p in gt_poses])
    est = PosePath3D(poses_se3=[np.asarray(p, dtype=np.float64) for p in est_poses])

    scale = 1.0
    if align in ("se3", "sim3"):
        # evo returns the similarity scale it applied; for SE3 it is fixed at 1.
        r, t, s = est.align(gt, correct_scale=(align == "sim3"))
        scale = float(s)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est))
    err = np.asarray(ape.error, dtype=np.float64)
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "std": float(err.std()),
        "max": float(err.max()),
        "n_pairs": int(len(err)),
        "scale": scale,
        "engine": "evo",
    }


def compute_ate(
    gt_poses: list[np.ndarray], est_poses: list[np.ndarray], align: str = "se3",
    engine: str = "auto",
) -> dict:
    """Absolute Trajectory Error after alignment ("se3" | "sim3" | "none").

    `engine`: "evo" forces evo and raises if it is unavailable; "numpy" forces
    the in-house Umeyama path; "auto" (default) uses evo when importable and
    falls back otherwise, recording which ran in the returned "engine" key so a
    results.json always says how its numbers were produced.

    The two agree to <= 5e-7 m (measured over 46 sequences), so the fallback is
    a convenience for environments without evo, not a different metric.
    """
    if engine in ("evo", "auto") and align in ("se3", "sim3", "none"):
        try:
            return _compute_ate_evo(gt_poses, est_poses, align)
        except Exception as exc:
            if engine == "evo":
                raise RuntimeError(f"evo scoring requested but failed: {exc}") from exc
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
        "engine": "numpy",
        "scale": scale,
        "per_frame_errors": errors.tolist(),
        "align_T": T_align.tolist(),
    }


def compute_rpe(
    gt_poses: list[np.ndarray], est_poses: list[np.ndarray], delta: int = 1,
) -> dict:
    """Relative Pose Error with a fixed frame delta.

    For each (i, i+delta): E = inv(inv(gt_i)·gt_{i+δ}) · (inv(est_i)·est_{i+δ}),
    trans_err = ‖E[:3,3]‖, rot_err = arccos((tr(E[:3,:3])−1)/2) in degrees.
    """
    trans_errors, rot_errors_deg = [], []
    for i in range(len(gt_poses) - delta):
        Q = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        P = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        E = np.linalg.inv(Q) @ P
        trans_errors.append(float(np.linalg.norm(E[:3, 3])))
        cos_angle = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors_deg.append(float(np.degrees(np.arccos(cos_angle))))

    if not trans_errors:
        return {"delta": delta, "n_pairs": 0}
    t, r = np.array(trans_errors), np.array(rot_errors_deg)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Trajectory writers
# ══════════════════════════════════════════════════════════════════════════════

def save_tum_trajectory(
    ts_to_pose: dict[float, np.ndarray], path: str | Path,
) -> None:
    """Write {timestamp: 4x4 cam-to-world} to a TUM trajectory file."""
    from scipy.spatial.transform import Rotation
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts in sorted(ts_to_pose.keys()):
            T = ts_to_pose[ts]
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (qx, qy, qz, qw)
            f.write(f"{ts:.6f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def load_tum_trajectory(
    path: str | Path, normalize_ns: bool = True,
) -> dict[float, np.ndarray]:
    """Parse a TUM-format trajectory file → {timestamp: 4x4 cam-to-world}.

    Used by systems that write their estimate to disk (LeanGate's MASt3R-SLAM,
    ORB-SLAM3's KeyFrameTrajectory.txt).  When `normalize_ns` and the timestamps
    look like raw integer nanoseconds (median > 1e14 — real epoch nanoseconds are
    ~1e18, whereas epoch *seconds* are ~1e9), they are scaled to seconds so
    association with seconds-based ground truth works unchanged.  The 1e14 cutoff
    deliberately leaves already-in-seconds EuRoC stamps (~1.4e9) untouched.
    """
    from scipy.spatial.transform import Rotation
    stamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            ts = float(p[0])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat(
                [float(p[4]), float(p[5]), float(p[6]), float(p[7])]).as_matrix()
            T[:3, 3] = [float(p[1]), float(p[2]), float(p[3])]
            stamps.append(ts)
            poses.append(T)
    if normalize_ns and stamps and float(np.median(stamps)) > 1e14:
        stamps = [s * 1e-9 for s in stamps]
    return {ts: T for ts, T in zip(stamps, poses)}


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_results(
    gt_poses: list[np.ndarray], est_poses: list[np.ndarray], ate_result: dict,
    out_dir: Path, seq_name: str, system: str = "estimate",
    loop_segments: Optional[list[np.ndarray]] = None,
) -> None:
    """Save trajectory_xy.png (top-down) and ate_errors.png (per-frame ATE).

    Each `loop_segments` entry is a (2, 3) array of loop-closure endpoint
    positions in the (unaligned) estimate frame; they are mapped through the
    same alignment transform as the trajectory and drawn as red chords on the
    top-down plot.  Positions come from the full estimated trajectory, so a
    chord may reach outside the plotted GT-matched section.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_pos = np.array([p[:3, 3] for p in gt_poses])
    est_pos = np.array([p[:3, 3] for p in est_poses])
    T_align = np.array(ate_result["align_T"])
    est_h = np.hstack([est_pos, np.ones((len(est_pos), 1))])
    est_aligned = (T_align @ est_h.T).T[:, :3]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], "g-", linewidth=1.5, label="Ground truth")
    ax.plot(est_aligned[:, 0], est_aligned[:, 2], "b--", linewidth=1.5,
            label=f"{system} (aligned)")
    # Endpoint markers keep converged closures visible: a successful closure
    # pulls both frames onto the same spot, collapsing the chord to a point.
    for k, seg in enumerate(loop_segments or []):
        seg_h = np.hstack([seg, np.ones((2, 1))])
        seg_a = (T_align @ seg_h.T).T
        ax.plot(seg_a[:, 0], seg_a[:, 2],
                "r-o", linewidth=1.2, markersize=5, markerfacecolor="none",
                alpha=0.9, zorder=4,
                label="Loop closure" if k == 0 else None)
    ax.scatter(gt_pos[0, 0], gt_pos[0, 2], color="green", s=80, zorder=5)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 2], color="darkgreen", s=80,
               marker="*", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"{seq_name} — top-down view  "
                 f"ATE RMSE {ate_result['rmse']*100:.1f} cm")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "trajectory_xy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    errors = np.array(ate_result["per_frame_errors"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors * 100, linewidth=1.2, color="royalblue")
    ax.axhline(ate_result["rmse"] * 100, color="red", linestyle="--",
               linewidth=1, label=f"RMSE = {ate_result['rmse']*100:.1f} cm")
    ax.axhline(ate_result["mean"] * 100, color="orange", linestyle=":",
               linewidth=1, label=f"Mean = {ate_result['mean']*100:.1f} cm")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("ATE (cm)")
    ax.set_title(f"{seq_name} — ATE over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "ate_errors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  Unified evaluation entry point
# ══════════════════════════════════════════════════════════════════════════════

def _provenance(config: Optional[dict]) -> dict:
    """Resolved run configuration, stamped into every results.json.

    `config` is whatever the caller passes (DA3-SLAM passes its SLAMConfig as a
    dict; external baselines pass their own argument namespace or None).  Values
    are coerced to JSON-safe primitives, and anything unserialisable is stored
    as its repr rather than dropped — an approximate record beats none.
    Git state is included so a number can be traced to a commit.
    """
    import subprocess
    out: dict = {}
    if config:
        def safe(v):
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, (list, tuple)):
                return [safe(x) for x in v]
            if isinstance(v, dict):
                return {str(k): safe(x) for k, x in v.items()}
            return repr(v)
        out["settings"] = {str(k): safe(v) for k, v in dict(config).items()}
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, timeout=5, cwd=str(Path(__file__).resolve().parent),
        ).stdout.strip() or None
        out["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=5, cwd=str(Path(__file__).resolve().parent)).stdout.strip())
    except Exception:
        pass
    return out


def evaluate_trajectory(
    est_ts_to_pose: dict[float, np.ndarray],
    gt_all: list[tuple[float, np.ndarray]],
    out_dir: Path,
    seq_name: str,
    *,
    system: str,
    dataset: str,
    n_frames: int,
    timings: dict,
    n_keyframes: Optional[int] = None,
    n_submaps: int = 0,
    n_loop_closures: int = 0,
    loop_pairs: Optional[list[tuple[float, float]]] = None,
    max_diff: float = 0.02,
    headline: str = "se3",
    config: Optional[dict] = None,
) -> dict | None:
    """Score one estimated trajectory against ground truth and save all outputs.

    This is the single shared scoring path used by every benchmark_*.py and
    every SLAM system: the only thing that varies upstream is how
    `est_ts_to_pose` (timestamp → 4x4 camera-to-world) was produced.

    Args:
        est_ts_to_pose:  estimated poses keyed by real/synthetic timestamp
        gt_all:          ground truth as (timestamp, 4x4 cam-to-world), sorted
        out_dir:         <root>/<seq_name>; created if missing
        system:          SLAM system name (stamped into results.json + plots)
        dataset:         "tum" | "euroc" | "replica"
        n_frames:        number of input frames fed to the system
        timings:         dict; must contain at least total_s and fps
        n_keyframes:     keyframe count (defaults to number of estimated poses)
        n_submaps:       submap count (0 for systems without submaps)
        n_loop_closures: detected loop closures (0 if N/A)
        loop_pairs:      loop-closure endpoint timestamps as (ts_a, ts_b)
                         pairs; drawn as red chords on the trajectory plot
        max_diff:        association tolerance in seconds
        headline:        which ATE drives the printed/plotted headline number
                         ("se3" for RGB-D scale-correct, "sim3" for monocular)

    Returns the metrics dict, or None if the sequence could not be evaluated.
    """
    from scipy.spatial.transform import Rotation

    out_dir.mkdir(parents=True, exist_ok=True)

    if not est_ts_to_pose:
        print(f"  [SKIP] {seq_name}: system produced no poses")
        return None

    # ── save full estimated trajectory ────────────────────────────────────────
    save_tum_trajectory(est_ts_to_pose, out_dir / "trajectory_est.txt")
    print(f"  Saved estimated trajectory → {out_dir / 'trajectory_est.txt'}")

    # ── associate estimate ↔ ground truth ─────────────────────────────────────
    gt_stamps = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())
    pairs = associate(est_stamps, gt_stamps, max_diff=max_diff)
    n_matched = len(pairs)
    print(f"  Matched {n_matched}/{len(est_stamps)} poses to GT "
          f"(max Δt={max_diff*1000:.1f}ms)")
    if n_matched < 3:
        print("  [SKIP] too few matched poses for evaluation")
        return None

    gt_matched = [gt_all[ib][1] for _, ib in pairs]
    est_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

    # ── save matched GT subset ────────────────────────────────────────────────
    with open(out_dir / "trajectory_gt.txt", "w") as f:
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
    ate_se3 = compute_ate(gt_matched, est_matched, align="se3")
    ate_sim3 = compute_ate(gt_matched, est_matched, align="sim3")
    rpe_1 = compute_rpe(gt_matched, est_matched, delta=1)
    rpe_n = compute_rpe(gt_matched, est_matched, delta=max(1, n_matched // 8))

    head = ate_sim3 if headline == "sim3" else ate_se3
    head_name = "Sim3" if headline == "sim3" else "SE3"
    print(f"\n  ┌─ ATE ({head_name} alignment) — headline ───────────────┐")
    print(f"  │  RMSE   {m_str(head['rmse']):<14}  Mean   {m_str(head['mean']):<14}  │")
    print(f"  │  Median {m_str(head['median']):<14}  Max    {m_str(head['max']):<14}  │")
    print(f"  ├─ ATE (SE3 {ate_se3['rmse']:.4f} m / Sim3 {ate_sim3['rmse']:.4f} m, "
          f"scale {ate_sim3['scale']:.4f}) ┤")
    print(f"  │  RPE δ=1   trans {cm_str(rpe_1['trans_rmse']):<12}  "
          f"rot {deg_str(rpe_1['rot_rmse_deg']):<10}  │")
    print(f"  │  RPE δ={rpe_n['delta']:<3}  trans {cm_str(rpe_n['trans_rmse']):<12}  "
          f"rot {deg_str(rpe_n['rot_rmse_deg']):<10}  │")
    print("  └─────────────────────────────────────────────────────────┘")
    kf = n_keyframes if n_keyframes is not None else len(est_ts_to_pose)
    print(f"  Frames: {n_frames}  Poses: {kf}  Submaps: {n_submaps}  "
          f"Loop closures: {n_loop_closures}  "
          f"Wall: {timings.get('total_s', float('nan')):.1f}s  "
          f"FPS: {timings.get('fps', float('nan')):.2f}")

    # ── assemble + save results.json ──────────────────────────────────────────
    metrics = {
        "system": system,
        "dataset": dataset,
        "sequence": seq_name,
        "n_frames": n_frames,
        "n_keyframes": kf,
        "n_submaps": n_submaps,
        "n_loop_closures": n_loop_closures,
        "n_matched_gt": n_matched,
        "ate_se3": {k: v for k, v in ate_se3.items()
                    if k not in ("per_frame_errors", "align_T")},
        "ate_sim3": {k: v for k, v in ate_sim3.items()
                     if k not in ("per_frame_errors", "align_T")},
        "rpe_delta1": {k: v for k, v in rpe_1.items()
                       if k not in ("per_frame_trans", "per_frame_rot")},
        "rpe_delta_n": {k: v for k, v in rpe_n.items()
                        if k not in ("per_frame_trans", "per_frame_rot")},
        "timings": timings,
        # PROVENANCE.  results.json used to record no configuration at all, so
        # "which settings produced this number" depended on the config file and
        # the sweep script still being around and unchanged.  That is exactly
        # how a table becomes unreproducible months later, so the resolved
        # config is stamped in alongside the metrics.
        "config": _provenance(config),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved → {out_dir / 'results.json'}")

    try:
        # Loop-closure chord endpoints come from the full estimated trajectory
        # (their stamps are exact est_ts_to_pose keys), not the GT-matched
        # subset — closures often anchor in stretches without GT coverage.
        loop_segments = [
            np.stack([est_ts_to_pose[ta][:3, 3], est_ts_to_pose[tb][:3, 3]])
            for ta, tb in (loop_pairs or [])
            if ta in est_ts_to_pose and tb in est_ts_to_pose
        ]
        if loop_pairs:
            print(f"  Loop-closure chords plotted: "
                  f"{len(loop_segments)}/{len(loop_pairs)}")
        plot_results(gt_matched, est_matched, head, out_dir, seq_name, system,
                     loop_segments=loop_segments)
    except Exception as e:  # plotting is best-effort
        print(f"  [warn] Plotting failed: {e}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-sequence summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_metrics: list[dict], headline: str, title: str) -> None:
    """Print a cross-sequence summary table; headline is "se3" or "sim3"."""
    if not all_metrics:
        return
    names = [m["sequence"] for m in all_metrics]
    col = max(len(n) for n in names) + 2
    head_key = "ate_sim3" if headline == "sim3" else "ate_se3"
    other_key = "ate_se3" if headline == "sim3" else "ate_sim3"
    head_lbl = "ATE Sim3 (m)" if headline == "sim3" else "ATE SE3 (m)"
    other_lbl = "ATE SE3 (m)" if headline == "sim3" else "ATE Sim3 (m)"

    header = (f"{'Sequence':<{col}}  {head_lbl:>13}  {other_lbl:>13}  "
              f"{'RPE-t (m)':>10}  {'RPE-r (°)':>10}  "
              f"{'FPS':>6}  {'KFs':>5}  {'LCs':>4}")
    sep = "═" * (len(header) + 2)
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))
    head_vals = []
    for m, name in zip(all_metrics, names):
        hv = m[head_key]["rmse"]
        head_vals.append(hv)
        print(f"  {name:<{col}}  {hv:>13.4f}  {m[other_key]['rmse']:>13.4f}  "
              f"{m['rpe_delta1'].get('trans_rmse', float('nan')):>10.4f}  "
              f"{m['rpe_delta1'].get('rot_rmse_deg', float('nan')):>10.3f}  "
              f"{m['timings'].get('fps', float('nan')):>6.2f}  "
              f"{m['n_keyframes']:>5}  {m['n_loop_closures']:>4}")
    print("  " + "─" * len(header))
    print(f"  {'avg':<{col}}  {sum(head_vals)/len(head_vals):>13.4f}")
    print(sep)


def save_summary(all_metrics: list[dict], out_dir: Path, dataset: str) -> None:
    """Write the aggregate <dataset>_summary.json next to the per-seq folders."""
    if not all_metrics:
        return
    agg_path = out_dir / f"{dataset}_summary.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAggregate results saved → {agg_path}")
