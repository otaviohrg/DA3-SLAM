"""
Shared EuRoC MAV (ASL format) dataset I/O.

Used by scripts/benchmark_euroc.py (in-house ATE/RPE metrics) and the
evals/ evo-based harness (evals/convert_euroc_gt.py, evals/eval_euroc.sh).

EuRoC sequences are distributed in the ASL format:

    <seq>/mav0/cam0/data/<timestamp_ns>.png      left grayscale frames
    <seq>/mav0/cam0/data.csv                      timestamp,filename index
    <seq>/mav0/cam0/sensor.yaml                   intrinsics, distortion, T_BS
    <seq>/mav0/state_groundtruth_estimate0/data.csv   body pose @ ~200 Hz

Two conversions live here because they are easy to get wrong:

  - Ground truth is the **IMU/body** pose in the world frame with a
    (w, x, y, z) quaternion; DA3-SLAM estimates the **camera** trajectory.
    load_euroc_groundtruth() returns camera-to-world poses via T_WC = T_WB @ T_BS
    (T_BS = sensor→body, from cam0/sensor.yaml).

  - Frames carry radial-tangential distortion but DA3 assumes a pinhole
    model, so undistort_images() optionally rectifies them.

This module imports without the GPU stack (numpy/cv2/yaml/scipy only).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def find_mav0(seq_dir: Path) -> Path | None:
    """Locate the `mav0` directory for a sequence.

    Accepts a path that *is* the mav0 parent, or any ancestor of it (EuRoC
    archives often unpack to a doubly-nested <seq>/<seq>/mav0 layout).  The
    macOS resource-fork copies under __MACOSX are ignored.
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
    """Read intrinsics, distortion coefficients, and T_BS from cam sensor.yaml.

    Returns a dict with:
        K          (3, 3) pinhole intrinsic matrix
        distortion (4,)   radial-tangential coefficients [k1, k2, p1, p2]
        T_BS       (4, 4) sensor(camera)→body transform
    """
    with open(mav0 / cam / "sensor.yaml") as f:
        sensor = yaml.safe_load(f)

    fu, fv, cu, cv = sensor["intrinsics"]
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.array(sensor["distortion_coefficients"], dtype=np.float64)
    T_BS = np.array(sensor["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return {"K": K, "distortion": distortion, "T_BS": T_BS}


def load_euroc_images(
    mav0: Path,
    cam: str = "cam0",
    max_frames: int | None = None,
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
    mav0: Path,
    T_BS: np.ndarray,
    seconds: bool = True,
) -> list[tuple[float, np.ndarray]]:
    """Read state_groundtruth_estimate0/data.csv as camera-to-world poses.

    The CSV stores the body pose T_WB (position + a w,x,y,z quaternion).  The
    returned poses are camera-to-world: T_WC = T_WB @ T_BS, sorted by
    timestamp.

    Args:
        seconds: when True, timestamps are returned in seconds (ns × 1e-9);
                 when False, the raw nanosecond integers (as floats) are kept
                 so they match run_slam.py's filename-derived timestamps.
    """
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
            # scipy expects (x, y, z, w); EuRoC stores (w, x, y, z)
            T_WB[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T_WB[:3, 3] = [px, py, pz]
            poses.append((ts, T_WB @ T_BS))
    return sorted(poses, key=lambda x: x[0])


def write_tum_trajectory(poses: list[tuple[float, np.ndarray]], path: str | Path) -> None:
    """Write (timestamp, 4x4 cam-to-world) poses to a TUM trajectory file."""
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in poses:
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # (qx, qy, qz, qw)
            f.write(f"{ts:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def ensure_euroc_gt_tum(
    seq_dir: str | Path,
    cam: str = "cam0",
    seconds: bool = True,
    out_path: str | Path | None = None,
) -> Path:
    """Convert a EuRoC sequence's ground truth to a camera-frame TUM file.

    Returns the path to the written file.  Used by the evo wrappers
    (scripts/evo_batch.py, scripts/evo_compare.py) so EuRoC sequences can be
    scored like any other TUM-format dataset.  Conversion is cheap (~0.5 s),
    so the file is regenerated each call rather than cached on mtime.

    Args:
        seq_dir:  EuRoC sequence directory (the mav0 parent, or an ancestor)
        cam:      camera whose frame the GT is expressed in
        seconds:  timestamp units (see load_euroc_groundtruth)
        out_path: output path; defaults to
                  <mav0-parent>/groundtruth_<cam>_tum[_ns].txt
    """
    mav0 = find_mav0(Path(seq_dir))
    if mav0 is None:
        raise FileNotFoundError(f"No extracted mav0/ found under {seq_dir}")
    if out_path is None:
        suffix = "" if seconds else "_ns"
        out_path = mav0.parent / f"groundtruth_{cam}_tum{suffix}.txt"

    calibration = load_camera_calibration(mav0, cam=cam)
    gt_poses = load_euroc_groundtruth(mav0, calibration["T_BS"], seconds=seconds)
    write_tum_trajectory(gt_poses, out_path)
    return Path(out_path)


def undistort_images(
    image_paths: list[str],
    calibration: dict,
    cache_dir: Path,
) -> list[str]:
    """Undistort frames with the EuRoC pinhole+radtan model into a cache dir.

    Filenames are preserved (so downstream timestamp parsing still works) and
    writes are idempotent: an existing undistorted frame is reused, so
    re-running does not redo the work.  Returns the undistorted image paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    K = calibration["K"]
    distortion = calibration["distortion"]

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
    if written:
        print(f"  Undistorted {written} frame(s) → {cache_dir}")
    else:
        print(f"  Using cached undistorted frames in {cache_dir}")
    return out_paths
