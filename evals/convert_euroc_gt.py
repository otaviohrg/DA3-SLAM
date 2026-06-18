#!/usr/bin/env python3
"""Convert a EuRoC sequence's ground truth to TUM format for evo_ape.

EuRoC ships ground truth as the **IMU/body** pose in the world frame, sampled
at ~200 Hz, in mav0/state_groundtruth_estimate0/data.csv with a (w, x, y, z)
quaternion and nanosecond timestamps.  DA3-SLAM estimates the **camera**
trajectory, so this script converts each pose to the camera frame
(T_WC = T_WB @ T_BS, with T_BS read from cam0/sensor.yaml) and writes the
standard TUM format evo_ape expects:

    timestamp tx ty tz qx qy qz qw

Timestamps default to nanoseconds so they line up with run_slam.py's
estimated trajectory, whose timestamps come from the EuRoC image filenames
(which are nanosecond integers).  evo_ape associates the two by timestamp, so
pass a matching --t_max_diff (e.g. 20000000 ns = 20 ms); see evals/eval_euroc.sh.

Usage:
    python evals/convert_euroc_gt.py \\
        --seq_dir data/EuRoC/vicon_room2/V2_01_easy \\
        --output  data/EuRoC/vicon_room2/V2_01_easy/groundtruth_cam0_tum.txt
"""

import argparse
import sys
from pathlib import Path

# EuRoC dataset I/O lives in scripts/euroc_common.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from euroc_common import (  # noqa: E402
    find_mav0,
    load_camera_calibration,
    load_euroc_groundtruth,
    write_tum_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EuRoC ground truth to camera-frame TUM format",
    )
    parser.add_argument("--seq_dir", required=True,
                        help="EuRoC sequence directory (the mav0 parent, or an ancestor)")
    parser.add_argument("--output", required=True,
                        help="Output TUM trajectory path")
    parser.add_argument("--cam", default="cam0",
                        help="Camera whose frame the GT is expressed in (default: cam0)")
    parser.add_argument("--timestamp_units", choices=["ns", "seconds"], default="ns",
                        help="ns matches run_slam.py on raw EuRoC frames (default); "
                             "use seconds only if the estimated trajectory is also in seconds")
    args = parser.parse_args()

    mav0 = find_mav0(Path(args.seq_dir))
    if mav0 is None:
        raise SystemExit(f"No extracted mav0/ found under {args.seq_dir}")

    calibration = load_camera_calibration(mav0, cam=args.cam)
    gt_poses = load_euroc_groundtruth(
        mav0, calibration["T_BS"], seconds=(args.timestamp_units == "seconds")
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tum_trajectory(gt_poses, out_path)
    print(f"Saved {len(gt_poses)} {args.cam}-frame GT poses "
          f"({args.timestamp_units} timestamps) → {out_path}")


if __name__ == "__main__":
    main()
