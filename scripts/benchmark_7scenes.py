"""
Benchmark DA3-SLAM on the Microsoft 7-Scenes RGB-D dataset.

Uses the shared SLAM benchmark standard (scripts/benchmark_common.py): ATE
(SE3 + Sim3) and RPE against each sequence's camera ground truth, with
identical output files to the other systems in this workspace.

7-Scenes ships a 4x4 CAMERA-TO-WORLD pose per frame (`frame-NNNNNN.pose.txt`),
the same convention as TUM ground truth, so no frame conversion is needed. It
has no timestamps (Kinect, 30 Hz), so they are synthesised as frame_idx / fps
and GT is associated with a half-frame tolerance — exactly as for Replica.
DA3-SLAM is monocular → headline is the Sim3 ATE.

The benchmark unit is a SEQUENCE, not a scene: `chess/seq-01`, `chess/seq-02`,
… Each scene contains several sequences of the same room, so pooling them would
hide per-trajectory behaviour and double-count the scene.

Expected layout:
    data/7scenes/<scene>/seq-XX/frame-NNNNNN.color.png
    data/7scenes/<scene>/seq-XX/frame-NNNNNN.pose.txt

Usage:
    python scripts/benchmark_7scenes.py --seq_dir data/7scenes/chess/seq-01
    python scripts/benchmark_7scenes.py --seq_dir data/7scenes/*/seq-*   # all
Aggregate: <out_dir>/7scenes_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_common as bc
from da3_runner import add_da3_cli, run_benchmark, run_da3

SYSTEM = "DA3-SLAM"
DATASET = "7scenes"
HEADLINE = "sim3"


def benchmark_sequence(seq_dir: Path, out_dir: Path, args, model) -> dict | None:
    """Run DA3-SLAM on one 7-Scenes sequence and score it via the shared standard.

    Returns the metrics dict, or None when the sequence has no usable frames or
    no usable ground truth.
    """
    # Name it "<scene>_<seq>" so results from different scenes never collide —
    # every scene has a seq-01.
    seq_name = f"{seq_dir.parent.name}_{seq_dir.name}"

    try:
        image_paths, timestamps, gt_all = bc.load_7scenes_sequence(
            seq_dir, args.max_frames, args.fps)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {seq_name}: {exc}")
        return None
    if len(gt_all) < 3:
        print(f"  [SKIP] {seq_name}: only {len(gt_all)} usable GT poses")
        return None

    print(f"\n  {len(image_paths)} RGB frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s  @ {args.fps} fps)  "
          f"{len(gt_all)} GT poses")

    est_ts_to_pose, timings, counts = run_da3(image_paths, timestamps, model, args)

    return bc.evaluate_trajectory(
        est_ts_to_pose, gt_all, out_dir, seq_name,
        system=SYSTEM, dataset=DATASET, n_frames=timings["n_frames"],
        timings=timings, max_diff=0.5 / args.fps, headline=HEADLINE, **counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on Microsoft 7-Scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="Path(s) to sequence dir(s), e.g. "
                             "data/7scenes/chess/seq-01")
    parser.add_argument("--out_dir", default="outputs/benchmark_7scenes",
                        help="Root output directory")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="7-Scenes capture rate (Kinect, 30 Hz)")
    add_da3_cli(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        args, args.seq_dir, benchmark_sequence, DATASET,
        title=f"{SYSTEM} — 7-Scenes  (ATE RMSE, Sim3 = monocular metric)",
        headline=HEADLINE, item_label="Sequence",
        # Every scene has a seq-01, so the leaf name alone is not unique and
        # all seven would share one output directory.
        name_fn=lambda p: f"{p.parent.name}_{p.name}")


if __name__ == "__main__":
    main()
