"""
Benchmark DA3-SLAM on Replica indoor scenes.

Uses the shared SLAM benchmark standard (scripts/benchmark_common.py): ATE
(SE3 + Sim3) and RPE against each scene's rendered camera ground truth, with
identical output files to the other systems in this workspace.

Replica has no real timestamps, so they are synthesised as frame_idx / fps and
GT is associated with a half-frame tolerance.  DA3-SLAM is monocular → headline
is the Sim3 ATE.

Expected layout:
    <scene>/results/frame*.jpg     RGB frames
    <scene>/gt_tum.txt             TUM-format camera ground truth

Usage:
    python scripts/benchmark_replica.py \\
        --scene_dir data/Replica/office0 data/Replica/room0 \\
        --out_dir outputs/benchmark_replica
Aggregate: <out_dir>/replica_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_common as bc
from da3_runner import build_config, add_da3_cli, run_benchmark, run_da3

SYSTEM = "DA3-SLAM"
DATASET = "replica"
HEADLINE = "sim3"


def benchmark_scene(scene_dir: Path, out_dir: Path, args, model) -> dict | None:
    """Run DA3-SLAM on one Replica scene and score it via the shared standard.

    Timestamps are synthesised as frame_idx / fps (Replica has none), so GT
    association uses a half-frame tolerance.  Returns the metrics dict, or
    None when gt_tum.txt is missing.
    """
    scene_name = scene_dir.name
    gt_txt = scene_dir / "gt_tum.txt"
    if not gt_txt.exists():
        print(f"  [SKIP] {scene_name}: gt_tum.txt not found in {scene_dir}")
        return None

    image_paths, timestamps = bc.load_replica_images(
        scene_dir, args.max_frames, args.fps)
    print(f"\n  {len(image_paths)} RGB frames  "
          f"({timestamps[0]:.3f}s – {timestamps[-1]:.3f}s  @ {args.fps} fps)")

    est_ts_to_pose, timings, counts = run_da3(image_paths, timestamps, model, args)

    gt_all = bc.load_groundtruth(gt_txt)
    return bc.evaluate_trajectory(
        est_ts_to_pose, gt_all, out_dir, scene_name,
        system=SYSTEM, dataset=DATASET, n_frames=timings["n_frames"],
        timings=timings, max_diff=0.5 / args.fps, headline=HEADLINE, config=_resolved_config(args), **counts)



def _resolved_config(args):
    """The SLAMConfig actually used, as a dict, for results.json provenance.

    Without this, results.json recorded no settings at all, so tracing a number
    back to (say) whether token_merging was enabled relied on config/default.yaml
    and the sweep script being unchanged months later.
    """
    try:
        from dataclasses import asdict, is_dataclass
        cfg = build_config(args)
        return asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    except Exception as exc:            # never fail a run over provenance
        return {"error": f"{type(exc).__name__}: {exc}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on Replica indoor scenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene_dir", nargs="+", required=True,
                        help="Path(s) to Replica scene dir(s), e.g. data/Replica/office0")
    parser.add_argument("--out_dir", default="outputs/benchmark_replica",
                        help="Root output directory")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Replica frame rate (default: 30.0)")
    add_da3_cli(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        args, args.scene_dir, benchmark_scene, DATASET,
        title=f"{SYSTEM} — Replica  (ATE RMSE, Sim3 = monocular metric)",
        headline=HEADLINE, item_label="Scene")


if __name__ == "__main__":
    main()
