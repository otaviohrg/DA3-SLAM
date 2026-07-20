"""
Benchmark DA3-SLAM on EuRoC MAV sequences (monocular, cam0).

Uses the shared SLAM benchmark standard (scripts/benchmark_common.py): ATE
(SE3 + Sim3) and RPE against the Vicon/Leica ground truth, with identical output
files to the other systems in this workspace.

EuRoC frames carry radial-tangential distortion but DA3 assumes a pinhole model,
so frames are undistorted by default (--no_undistort to skip).  Ground truth is
converted to the camera frame (T_WC = T_WB @ T_BS).  DA3-SLAM is monocular →
headline is the Sim3 ATE.

Usage:
    python scripts/benchmark_euroc.py \\
        --seq_dir data/EuRoC/vicon_room1/V1_01_easy \\
                  data/EuRoC/vicon_room2/V2_01_easy \\
        --out_dir outputs/euroc
Point --seq_dir at the directory containing mav0 (or any ancestor).
Aggregate: <out_dir>/euroc_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_common as bc
from da3_runner import add_da3_cli, run_benchmark, run_da3

SYSTEM = "DA3-SLAM"
DATASET = "euroc"
HEADLINE = "sim3"


def benchmark_sequence(seq_dir: Path, out_dir: Path, args, model) -> dict | None:
    """Run DA3-SLAM on one EuRoC sequence and score it via the shared standard.

    Undistorts cam0 frames first (unless --no_undistort) and converts GT to
    the camera frame.  Returns the metrics dict, or None when no mav0/ or no
    frames are found.
    """
    seq_name = seq_dir.name
    mav0 = bc.find_mav0(seq_dir)
    if mav0 is None:
        print(f"  [SKIP] {seq_name}: no extracted mav0/ under {seq_dir}")
        return None

    calibration = bc.load_camera_calibration(mav0, cam=args.cam)
    image_paths, timestamps = bc.load_euroc_images(
        mav0, cam=args.cam, max_frames=args.max_frames)
    if not image_paths:
        print(f"  [SKIP] {seq_name}: no images in {args.cam}/data.csv")
        return None
    print(f"\n  {len(image_paths)} {args.cam} frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.undistort:
        # undistort_images preserves filenames so frame-id parsing stays valid.
        image_paths = bc.undistort_images(
            image_paths, calibration, out_dir / f"undistorted_{args.cam}")

    est_ts_to_pose, timings, counts = run_da3(image_paths, timestamps, model, args)

    gt_all = bc.load_euroc_groundtruth(mav0, calibration["T_BS"])
    return bc.evaluate_trajectory(
        est_ts_to_pose, gt_all, out_dir, seq_name,
        system=SYSTEM, dataset=DATASET, n_frames=timings["n_frames"],
        timings=timings, max_diff=0.02, headline=HEADLINE, **counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on EuRoC MAV sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="EuRoC sequence dir(s) (the mav0 parent or ancestor)")
    parser.add_argument("--out_dir", default="outputs/euroc",
                        help="Root output directory")
    parser.add_argument("--cam", default="cam0", help="Camera (cam0 = left)")
    parser.add_argument("--no_undistort", dest="undistort", action="store_false",
                        help="Skip radial-tangential undistortion")
    add_da3_cli(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        args, args.seq_dir, benchmark_sequence, DATASET,
        title=f"{SYSTEM} — EuRoC MAV  (ATE RMSE, Sim3 = monocular metric)",
        headline=HEADLINE)


if __name__ == "__main__":
    main()
