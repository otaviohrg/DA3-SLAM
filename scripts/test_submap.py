"""
Test da3_slam.submap.

Usage:
    python scripts/test_submap.py --image_dir data/video1_5fps
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--submap_size", type=int, default=8)
    parser.add_argument("--conf_percentile", type=float, default=40.0)
    parser.add_argument("--save_ply", default=None,
                        help="Save merged point cloud to this .ply path")
    return parser.parse_args()


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        sys.exit(1)


def main():
    args = parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )[: args.submap_size]
    check(f"Found {len(paths)} images", len(paths) > 0)

    from da3_slam.depth_estimator import DepthEstimator
    from da3_slam.submap import SubmapBuilder

    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator, conf_percentile=args.conf_percentile)

    # ── build submap ──────────────────────────────────────────────────────────
    header("SubmapBuilder.build()")
    seq_indices = list(range(len(paths)))
    submap = builder.build(paths, seq_indices, submap_idx=0)
    N = len(paths)

    # ── submap-level checks ───────────────────────────────────────────────────
    header("Submap properties")
    check(f"n_frames = {N}", submap.n_frames == N)
    check("points_world shape (N_total, 3)", submap.points_world.ndim == 2
          and submap.points_world.shape[1] == 3)
    check("colors shape (N_total, 3)", submap.colors.shape == submap.points_world.shape)
    check("colors dtype uint8", submap.colors.dtype == np.uint8)
    check(f"extrinsics shape ({N}, 4, 4)", submap.extrinsics.shape == (N, 4, 4))
    check(f"positions_world shape ({N}, 3)", submap.positions_world.shape == (N, 3))

    total_pts = len(submap.points_world)
    print(f"  Total points: {total_pts:,}")

    # ── per-frame checks ──────────────────────────────────────────────────────
    header("Per-frame checks")
    for frame in submap.frames:
        check(
            f"frame {frame.seq_idx:03d}: points_cam z > 0",
            (frame.points_cam[:, 2] > 0).all(),
        )
        check(
            f"frame {frame.seq_idx:03d}: points_cam and points_world same count",
            len(frame.points_cam) == len(frame.points_world),
        )
        check(
            f"frame {frame.seq_idx:03d}: colors match point count",
            len(frame.colors) == len(frame.points_world),
        )
        check(
            f"frame {frame.seq_idx:03d}: det(R) ≈ 1.0",
            abs(np.linalg.det(frame.extrinsic[:3, :3]) - 1.0) < 1e-3,
        )
        print(
            f"  frame {frame.seq_idx:03d}  "
            f"{frame.n_points:,} pts  "
            f"cam_pos=[{frame.position_world[0]:.3f}, "
            f"{frame.position_world[1]:.3f}, "
            f"{frame.position_world[2]:.3f}]"
        )

    # ── cam-to-world roundtrip ────────────────────────────────────────────────
    header("cam-to-world roundtrip")
    frame = submap.frames[0]
    # Transform a cam point to world and back, should recover original
    pt_cam = frame.points_cam[:1]
    E = frame.extrinsic
    c2w = frame.cam_to_world
    pt_world = (pt_cam @ c2w[:3, :3].T) + c2w[:3, 3]
    pt_cam_back = (pt_world @ E[:3, :3].T) + E[:3, 3]
    err = float(np.linalg.norm(pt_cam - pt_cam_back))
    check(f"roundtrip error < 1e-4 m  (got {err:.2e})", err < 1e-4)

    # ── trajectory sanity ─────────────────────────────────────────────────────
    header("Trajectory sanity")
    positions = submap.positions_world
    dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    print(f"  Camera positions (world):")
    for i, pos in enumerate(positions):
        print(f"    frame {i:02d}  [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"  Inter-frame distances: {dists.round(3)}")
    check("camera moves between frames", dists.max() > 0.0)

    # ── optional PLY export ───────────────────────────────────────────────────
    if args.save_ply:
        header(f"Saving PLY → {args.save_ply}")
        _save_ply(submap.points_world, submap.colors, args.save_ply)
        print(f"  Saved {total_pts:,} points.")

    header("All checks passed")


def _save_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    """Write a coloured point cloud to a PLY file."""
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{col[0]} {col[1]} {col[2]}\n")


if __name__ == "__main__":
    main()
