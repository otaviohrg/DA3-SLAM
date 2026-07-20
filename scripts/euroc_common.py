"""
Shared EuRoC MAV (ASL format) dataset helpers.

The dataset I/O (find_mav0, calibration, image list, ground truth,
undistortion) lives in benchmark_common.py — the dataset-agnostic benchmark
standard shared across the SLAM/ workspace — and is re-exported here.  This
module adds the DA3-SLAM-specific glue used by the evals/ harness
(evals/convert_euroc_gt.py, evals/eval_euroc.sh):

  - write_tum_trajectory():  save (timestamp, pose) pairs in TUM format
  - ensure_euroc_gt_tum():   convert a sequence's ground truth to a
                             camera-frame TUM file on disk

Two conversions handled by the shared loaders are easy to get wrong:

  - Ground truth is the **IMU/body** pose in the world frame with a
    (w, x, y, z) quaternion; DA3-SLAM estimates the **camera** trajectory.
    load_euroc_groundtruth() returns camera-to-world poses via
    T_WC = T_WB @ T_BS (T_BS = sensor→body, from cam0/sensor.yaml).

  - Frames carry radial-tangential distortion but DA3 assumes a pinhole
    model, so undistort_images() optionally rectifies them.

This module imports without the GPU stack (numpy/cv2/yaml/scipy only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_common import (
    find_mav0,
    load_camera_calibration,
    load_euroc_groundtruth,
    load_euroc_images,
    undistort_images,
)

__all__ = [
    "find_mav0", "load_camera_calibration", "load_euroc_images",
    "load_euroc_groundtruth", "undistort_images",
    "write_tum_trajectory", "ensure_euroc_gt_tum",
]


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

    Returns the path to the written file.  Used by evals/convert_euroc_gt.py
    so EuRoC sequences can be
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
