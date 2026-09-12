"""
Benchmark DA3-SLAM on the Unified Autonomy Stack (UAS) dataset (monocular).

NTNU-ARL unified_autonomy_stack_datasets:
https://huggingface.co/datasets/ntnu-arl/unified_autonomy_stack_datasets

Uses the shared SLAM benchmark standard (scripts/benchmark_common.py): ATE
(SE3 + Sim3) and RPE against the dataset's TUM ground truth, with identical
output files to the other systems/datasets in this workspace.  DA3-SLAM is
monocular → headline is the Sim3 ATE.

UAS-specific I/O lives in scripts/uas_common.py (the analog of euroc_common.py):
  * sequences ship as ROS 1 bags (sensors_only.bag) — frames are extracted from
    the platform's camera topic with the pure-Python `rosbags` package;
  * cameras use the equidistant (fisheye) distortion model, so frames are
    undistorted with cv2.fisheye by default (--no_undistort to skip);
  * ground truth is TUM format in seconds (only 4 sequences ship GT:
    fyllingsdalen_tunnel, runehamar_tunnel/hornbill, campus_fog, frozen_lake).

Reading ROS 1 bags requires:  pip install rosbags

Usage:
    python scripts/benchmark_uas.py \\
        --seq_dir data/UAS/fyllingsdalen_tunnel \\
                  data/UAS/runehamar_tunnel/hornbill \\
                  data/UAS/campus_fog \\
                  data/UAS/frozen_lake \\
        --out_dir outputs/uas
Point --seq_dir at the sequence folder containing sensors_only.bag and the .tum.
Aggregate: <out_dir>/uas_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_common as bc
import uas_common as uas
from da3_runner import add_da3_cli, run_benchmark, run_da3

SYSTEM = "DA3-SLAM"
DATASET = "uas"
HEADLINE = "sim3"


def _cache_dir(seq_name: str, out_dir: Path, args, kind: str) -> Path:
    """Cache dir for `kind` (frames|undistorted): shared --frames_dir ROOT/<seq>/<kind>
    when given, else out_dir/<kind>.

    Mirrors VGGT-SLAM's helper of the same name, so a single extracted-frame
    cache serves both systems.  Bag extraction is ~20 GB and idempotent, and a
    cross-system comparison should be reading byte-identical frames anyway.
    """
    if args.frames_dir:
        return Path(args.frames_dir) / seq_name / kind
    return out_dir / kind


def benchmark_sequence(seq_dir: Path, out_dir: Path, args, model) -> dict | None:
    """Run DA3-SLAM on one UAS sequence and score it via the shared standard.

    Extracts (and caches) frames from the ROS 1 bag, fisheye-undistorts them
    by default, and passes GT into run_da3 so trajectory snapshots can
    overlay it.  Returns the metrics dict, or None when the bag or the
    ground-truth .tum file is missing.
    """
    seq_name = seq_dir.name

    bag = uas.find_bag(seq_dir)
    if bag is None:
        print(f"  [SKIP] {seq_name}: no sensors_only.bag under {seq_dir}")
        return None
    gt_txt = uas.find_gt(seq_dir)
    if gt_txt is None:
        print(f"  [SKIP] {seq_name}: no ground-truth .tum file (not all "
              f"sequences ship GT)")
        return None

    topic, calib_path = uas.resolve_camera(
        seq_dir, args.topic, args.calib, args.calib_dir)
    calibration = uas.load_calibration(calib_path)
    print(f"\n  Bag:   {bag.name}   topic: {topic}")
    print(f"  Calib: {calib_path.name}  ({calibration['model']}, "
          f"{calibration['size'][0]}x{calibration['size'][1]})")

    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths, timestamps = uas.extract_bag_frames(
        bag, topic, _cache_dir(seq_name, out_dir, args, "frames"),
        max_frames=args.max_frames)
    if not image_paths:
        print(f"  [SKIP] {seq_name}: no frames extracted")
        return None
    print(f"  {len(image_paths)} frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s)")

    if args.undistort:
        image_paths = uas.undistort_frames(
            image_paths, calibration,
            _cache_dir(seq_name, out_dir, args, "undistorted"))

    gt_all = bc.load_groundtruth(gt_txt)
    est_ts_to_pose, timings, counts = run_da3(
        image_paths, timestamps, model, args, out_dir=out_dir, gt=gt_all)

    return bc.evaluate_trajectory(
        est_ts_to_pose, gt_all, out_dir, seq_name,
        system=SYSTEM, dataset=DATASET, n_frames=timings["n_frames"],
        timings=timings, max_diff=args.max_diff, headline=HEADLINE, **counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on the Unified Autonomy Stack dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="UAS sequence dir(s) containing sensors_only.bag + .tum")
    parser.add_argument("--out_dir", default="outputs/uas",
                        help="Root output directory")
    parser.add_argument("--topic", default=None,
                        help="Camera topic override (else inferred per sequence)")
    parser.add_argument("--calib", default=None,
                        help="Calibration YAML override (else inferred per sequence)")
    parser.add_argument("--calib_dir", type=Path, default=None,
                        help="Dataset calibration/ folder (else auto-discovered)")
    parser.add_argument("--frames_dir", default=None,
                        help="Shared extracted-frame cache ROOT; frames are "
                             "read from ROOT/<seq>/frames and undistorted into "
                             "ROOT/<seq>/undistorted.  Same flag and layout as "
                             "VGGT-SLAM's, so both systems consume identical "
                             "frames instead of re-extracting ~20 GB of bags")
    parser.add_argument("--no_undistort", dest="undistort", action="store_false",
                        help="Skip fisheye undistortion")
    parser.add_argument("--max_diff", type=float, default=0.02,
                        help="Estimate↔GT association tolerance in seconds")
    add_da3_cli(parser)
    # UAS default: trajectory time-lapse + pre-optimisation loop-closure
    # snapshots every 10 s (see trajectory_snapshots.py); 0 disables.
    parser.set_defaults(snapshot_interval=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        args, args.seq_dir, benchmark_sequence, DATASET,
        title=f"{SYSTEM} — UAS  (ATE RMSE, Sim3 = monocular metric)",
        headline=HEADLINE)


if __name__ == "__main__":
    main()
