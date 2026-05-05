"""
DA3-SLAM entry point.

Usage:
    python scripts/run_slam.py --image_dir data/video1_30fps
    python scripts/run_slam.py --image_dir data/video1_30fps \
        --out_dir outputs/run1 --submap_size 8 --no_loop_closure
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="DA3-SLAM runner")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="/app/outputs/slam",
                        help="Output directory for trajectory and map files")
    parser.add_argument("--submap_size", type=int, default=8)
    parser.add_argument("--min_disparity_frac", type=float, default=0.15)
    parser.add_argument("--conf_percentile", type=float, default=40.0)
    parser.add_argument("--loop_threshold", type=float, default=0.85)
    parser.add_argument("--no_loop_closure", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap the number of input frames (for quick tests)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── collect image paths ────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        sys.exit(1)
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
    print(f"[run_slam] {len(image_paths)} images from {args.image_dir}")

    # ── build config ───────────────────────────────────────────────────────
    from da3_slam.slam import DA3SLAM, SLAMConfig
    from da3_slam.keyframe_selector import KeyframeSelectorConfig
    from da3_slam.loop_closure import LoopClosureConfig

    config = SLAMConfig(
        submap_size=args.submap_size,
        conf_percentile=args.conf_percentile,
        enable_loop_closure=not args.no_loop_closure,
        keyframe=KeyframeSelectorConfig(
            min_disparity_frac=args.min_disparity_frac,
            max_submap_size=args.submap_size,
        ),
        loop_closure=LoopClosureConfig(
            similarity_threshold=args.loop_threshold,
        ),
    )

    # ── run ────────────────────────────────────────────────────────────────
    t_total = time.time()
    slam = DA3SLAM(config)
    result = slam.run(image_paths)
    t_total = time.time() - t_total

    # ── print summary ──────────────────────────────────────────────────────
    traj = result.trajectory
    positions = traj[:, :3, 3]
    dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = dists.sum()

    print("\n" + "─" * 60)
    print("  SLAM Summary")
    print("─" * 60)
    print(f"  Frames processed:    {len(image_paths)}")
    print(f"  Keyframes:           {result.n_keyframes}")
    print(f"  Submaps:             {len(result.submaps)}")
    print(f"  Loop closures:       {len(result.loop_closures)}")
    print(f"  Opt. final error:    {result.optimization.final_error:.6f}")
    print(f"  Trajectory length:   {path_length:.3f} m")
    print(f"  Total wall time:     {t_total:.1f}s")
    print()
    print("  Timing breakdown:")
    for k, v in result.timings.items():
        print(f"    {k:<25} {v:.1f}s")

    # ── save outputs ───────────────────────────────────────────────────────
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kitti_path = str(out / "trajectory_kitti.txt")
    tum_path   = str(out / "trajectory_tum.txt")
    ply_path   = str(out / "map.ply")

    result.save_kitti(kitti_path)
    result.save_tum(tum_path)
    result.save_ply(ply_path)

    print(f"\n  Outputs saved to {out}/")
    print(f"    trajectory_kitti.txt  ({result.n_keyframes} poses)")
    print(f"    trajectory_tum.txt    ({result.n_keyframes} poses)")
    n_pts = sum(len(sm.points_world) for sm in result.submaps)
    print(f"    map.ply               ({n_pts:,} points)")


if __name__ == "__main__":
    main()
