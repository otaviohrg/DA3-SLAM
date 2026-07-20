"""
Benchmark DA3-SLAM on TUM RGB-D sequences.

Uses the shared SLAM benchmark standard (scripts/benchmark_common.py) so the
numbers and output files line up exactly with VGGT-SLAM / DROID-SLAM /
ORB-SLAM3.  DA3-SLAM is monocular → the headline accuracy number is the
Sim3-aligned ATE (SE3 ATE is also recorded).

The DA3 model is loaded once and reused across all sequences.

Usage:
    python scripts/benchmark_tum.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_xyz \\
                  data/tum/rgbd_dataset_freiburg1_desk \\
        --out_dir outputs/benchmark
Aggregate: <out_dir>/tum_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_common as bc
from da3_runner import add_da3_cli, run_benchmark, run_da3

SYSTEM = "DA3-SLAM"
DATASET = "tum"
HEADLINE = "sim3"


def benchmark_sequence(seq_dir: Path, out_dir: Path, args, model) -> dict | None:
    """Run DA3-SLAM on one TUM sequence and score it via the shared standard.

    Returns the metrics dict from bc.evaluate_trajectory, or None when the
    sequence is missing rgb.txt / groundtruth.txt or cannot be evaluated.
    """
    seq_name = seq_dir.name
    if not (seq_dir / "rgb.txt").exists():
        print(f"  [SKIP] {seq_name}: rgb.txt not found in {seq_dir}")
        return None
    gt_txt = seq_dir / "groundtruth.txt"
    if not gt_txt.exists():
        print(f"  [SKIP] {seq_name}: groundtruth.txt not found in {seq_dir}")
        return None

    image_paths, timestamps = bc.load_rgb_list(seq_dir, args.max_frames)
    print(f"\n  {len(image_paths)} RGB frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s)")

    est_ts_to_pose, timings, counts = run_da3(image_paths, timestamps, model, args)

    gt_all = bc.load_groundtruth(gt_txt)
    return bc.evaluate_trajectory(
        est_ts_to_pose, gt_all, out_dir, seq_name,
        system=SYSTEM, dataset=DATASET, n_frames=timings["n_frames"],
        timings=timings, max_diff=0.02, headline=HEADLINE, **counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on TUM RGB-D sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="Path(s) to TUM sequence directory/directories")
    parser.add_argument("--out_dir", default="outputs/benchmark",
                        help="Root output directory")
    add_da3_cli(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        args, args.seq_dir, benchmark_sequence, DATASET,
        title=f"{SYSTEM} — TUM RGB-D  (ATE RMSE, Sim3 = monocular metric)",
        headline=HEADLINE)


if __name__ == "__main__":
    main()
