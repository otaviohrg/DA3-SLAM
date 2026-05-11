"""
Diagnostic script for map quality issues.

Runs the pipeline on a small number of frames and saves:
  - Per-submap PLY in local (DA3 world) frame
  - Per-submap PLY in global frame (after T_opt)
  - Combined PLY
  - Text report with T_opt values and bounding boxes

Usage:
    python scripts/debug_map.py --image_dir data/video1_30fps --max_frames 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a coloured point cloud to PLY (ASCII)."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    colors = colors[finite]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])}\n")
    print(f"  Saved {path}  ({len(points)} pts)")


def bbox_str(pts: np.ndarray) -> str:
    if len(pts) == 0:
        return "empty"
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    return (f"X[{mn[0]:.2f},{mx[0]:.2f}]  "
            f"Y[{mn[1]:.2f},{mx[1]:.2f}]  "
            f"Z[{mn[2]:.2f},{mx[2]:.2f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="/app/outputs/debug_map")
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--submap_size", type=int, default=8)
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )[:args.max_frames]
    print(f"Using {len(image_paths)} frames")

    from da3_slam.slam import DA3SLAM
    from da3_slam.config import load_slam_config

    config = load_slam_config(submap_size=args.submap_size)
    config.enable_loop_closure = False

    slam = DA3SLAM(config)
    result = slam.run(image_paths)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 70)
    print("  PER-SUBMAP DIAGNOSTICS")
    print("═" * 70)

    all_global_pts = []
    all_global_col = []

    for sm in result.submaps:
        T_opt = result.optimization.pose(sm.idx)
        local_pts = sm.points_world   # (M, 3) in DA3 local frame
        local_col = sm.colors

        # Apply T_opt to get global frame
        pts_h = np.hstack([local_pts, np.ones((len(local_pts), 1), dtype=np.float32)])
        global_pts = (T_opt.astype(np.float64) @ pts_h.T.astype(np.float64)).T[:, :3].astype(np.float32)
        global_col = local_col

        print(f"\nSubmap {sm.idx}  ({sm.n_frames} frames, {len(local_pts):,} pts)")
        print(f"  T_opt translation : {T_opt[:3, 3]}")
        print(f"  T_opt det(R)      : {np.linalg.det(T_opt[:3, :3]):.6f}  (should be 1.0)")
        print(f"  Local bbox        : {bbox_str(local_pts)}")
        print(f"  Global bbox       : {bbox_str(global_pts)}")
        nan_frac = (~np.isfinite(global_pts).all(axis=1)).mean()
        print(f"  NaN/Inf fraction  : {nan_frac:.4f}")

        # Save per-submap PLYs
        write_ply(str(out / f"submap_{sm.idx:02d}_local.ply"), local_pts, local_col)
        write_ply(str(out / f"submap_{sm.idx:02d}_global.ply"), global_pts, global_col)

        all_global_pts.append(global_pts)
        all_global_col.append(global_col)

    # Save combined global PLY
    print("\n" + "─" * 70)
    combined_pts = np.concatenate(all_global_pts)
    combined_col = np.concatenate(all_global_col)
    print(f"Combined: {len(combined_pts):,} pts")
    print(f"Combined global bbox: {bbox_str(combined_pts)}")
    write_ply(str(out / "map_combined.ply"), combined_pts, combined_col)

    # Also print per-submap alignments
    print("\n" + "─" * 70)
    print("  INTER-SUBMAP ALIGNMENTS (world_b_to_world_a)")
    print("─" * 70)
    from da3_slam.backend.processing.alignment import SubmapAligner
    aligner = SubmapAligner()
    for i in range(1, len(result.submaps)):
        sm_a = result.submaps[i - 1]
        sm_b = result.submaps[i]
        al = aligner.align(sm_a, sm_b)
        print(f"  Submap {i-1} → {i}:  "
              f"rot={al.rotation_angle_deg:.2f}°  "
              f"|t|={np.linalg.norm(al.translation):.4f}m  "
              f"t={al.translation}")

    # Per-frame extrinsic check (for convention diagnosis)
    print("\n" + "─" * 70)
    print("  EXTRINSIC CONVENTION CHECK (first submap)")
    print("─" * 70)
    sm0 = result.submaps[0]
    for i, frame in enumerate(sm0.frames[:4]):
        E = frame.extrinsic
        print(f"  Frame {i} (seq {frame.seq_idx}):")
        print(f"    extrinsic[:3,3] (translation col) = {E[:3, 3]}")
        print(f"    det(R) = {np.linalg.det(E[:3, :3]):.6f}")
        print(f"    cam_to_world[:3,3] = {frame.cam_to_world[:3, 3]}")

    print(f"\nAll outputs in: {out}/")


if __name__ == "__main__":
    main()
