"""
DA3-SLAM entry point.

Defaults are read from config/default.yaml.
Any value can be overridden on the command line.

Usage:
    python scripts/run_slam.py --image_dir data/video1_30fps
    python scripts/run_slam.py --image_dir data/video1_30fps --config config/default.yaml
    python scripts/run_slam.py --image_dir data/video1_30fps \\
        --out_dir outputs/run1 --submap_size 12 --conf_percentile 75 --no_loop_closure
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml


# ── config loading ─────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args(cfg: dict) -> argparse.Namespace:
    """Build argument parser with defaults drawn from the loaded config."""
    lc = cfg.get("loop_closure", {})
    kf = cfg.get("keyframe", {})
    noise = cfg.get("noise", {})

    parser = argparse.ArgumentParser(
        description="DA3-SLAM runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── meta ──────────────────────────────────────────────────────────────────
    parser.add_argument("--image_dir", required=True,
                        help="Directory of input images")
    parser.add_argument("--out_dir", default="/app/outputs/slam",
                        help="Output directory for trajectory and map files")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG),
                        help="Path to YAML config file")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap the number of input frames (for quick tests)")

    # ── DA3 model ─────────────────────────────────────────────────────────────
    parser.add_argument("--da3_model", default=cfg.get("da3_model"),
                        help="DA3 model ID")
    parser.add_argument("--da3_process_res", type=int,
                        default=cfg.get("da3_process_res"),
                        help="DA3 processing resolution")

    # ── submap ────────────────────────────────────────────────────────────────
    parser.add_argument("--submap_size", type=int,
                        default=cfg.get("submap_size"),
                        help="Max keyframes per submap (including anchor overlap)")

    # ── keyframe selection ────────────────────────────────────────────────────
    parser.add_argument("--min_disparity_frac", type=float,
                        default=kf.get("min_disparity_frac"),
                        help="Min optical flow as fraction of image width [0,1]")

    # ── point cloud ───────────────────────────────────────────────────────────
    parser.add_argument("--conf_percentile", type=float,
                        default=cfg.get("conf_percentile"),
                        help="Global confidence percentile threshold (0-100). "
                             "Higher = fewer but cleaner points")

    # ── loop closure ──────────────────────────────────────────────────────────
    parser.add_argument("--no_loop_closure", action="store_true",
                        default=not lc.get("enable", True),
                        help="Disable loop closure detection")
    parser.add_argument("--loop_threshold", type=float,
                        default=lc.get("similarity_threshold"),
                        help="DINOv2 cosine similarity threshold for loop detection")

    return parser.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # Load YAML first (to get defaults), then parse CLI on top
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_DEFAULT_CONFIG))
    known, _ = pre.parse_known_args()

    cfg = load_config(known.config)
    args = parse_args(cfg)

    # ── collect image paths ────────────────────────────────────────────────────
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

    # ── build config (YAML → CLI overrides) ───────────────────────────────────
    from da3_slam.slam import DA3SLAM
    from da3_slam.config import load_slam_config

    # load_slam_config reads the YAML; keyword args override top-level scalars
    config = load_slam_config(
        args.config,
        submap_size=args.submap_size,
        conf_percentile=args.conf_percentile,
        da3_model=args.da3_model,
        da3_process_res=args.da3_process_res,
    )
    # Boolean / nested overrides not covered by load_slam_config scalars
    if args.no_loop_closure:
        config.enable_loop_closure = False
    if args.loop_threshold is not None:
        config.loop_closure.similarity_threshold = args.loop_threshold
    if args.min_disparity_frac is not None:
        config.keyframe.min_disparity_frac = args.min_disparity_frac

    # ── run ────────────────────────────────────────────────────────────────────
    t_total = time.time()
    slam = DA3SLAM(config)
    result = slam.run(image_paths)
    t_total = time.time() - t_total

    # ── print summary ──────────────────────────────────────────────────────────
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

    # ── save outputs ───────────────────────────────────────────────────────────
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
