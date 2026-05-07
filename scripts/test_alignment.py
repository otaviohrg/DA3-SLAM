"""
Test da3_slam.alignment.

Builds two consecutive submaps with a shared anchor frame, computes the
alignment transform, and verifies that the anchor's world position is
identical after mapping submap B into submap A's frame.

Usage:
    python scripts/test_alignment.py --image_dir data/video1_30fps
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--submap_size", type=int, default=8)
    parser.add_argument("--confidence_percentile", type=float, default=40.0)
    parser.add_argument("--save_ply", default=None,
                        help="Save merged two-submap point cloud to this .ply")
    return parser.parse_args()


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"FAIL: {label}")


def main():
    args = parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )
    # Need at least 2 full submaps
    needed = args.submap_size * 2 - 1
    check(f"At least {needed} images available", len(all_paths) >= needed)
    all_paths = all_paths[:needed]

    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    from da3_slam.backend.inference.submap import SubmapBuilder
    from da3_slam.backend.processing.alignment import SubmapAligner

    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator, confidence_percentile=args.confidence_percentile)
    aligner = SubmapAligner()

    # ── build sequence ────────────────────────────────────────────────────────
    header("SubmapBuilder.build_sequence()")
    submaps = builder.build_sequence(all_paths, submap_size=args.submap_size)
    check("2 submaps built", len(submaps) == 2)

    sm_a, sm_b = submaps[0], submaps[1]
    print(f"  Submap A: {sm_a.n_frames} frames  seq_idx {sm_a.frames[0].seq_idx}–{sm_a.frames[-1].seq_idx}")
    print(f"  Submap B: {sm_b.n_frames} frames  seq_idx {sm_b.frames[0].seq_idx}–{sm_b.frames[-1].seq_idx}")

    anchor_idx_a = sm_a.frames[-1].seq_idx
    anchor_idx_b = sm_b.frames[0].seq_idx
    check("anchor frame seq_idx matches", anchor_idx_a == anchor_idx_b)

    # ── alignment ─────────────────────────────────────────────────────────────
    header("SubmapAligner.align()")
    result = aligner.align(sm_a, sm_b)
    print(f"  Method:           {result.method}")
    print(f"  Rotation angle:   {result.rotation_angle_deg:.4f} deg")
    print(f"  Translation:      {result.translation.round(4)}")
    print(f"  T_a_from_b:\n{result.T_a_from_b.round(4)}")

    check("T shape (4,4)", result.T_a_from_b.shape == (4, 4))
    check("T bottom row = [0,0,0,1]",
          np.allclose(result.T_a_from_b[3], [0, 0, 0, 1], atol=1e-5))
    check("det(R) ≈ 1.0",
          abs(np.linalg.det(result.rotation) - 1.0) < 1e-4)

    # ── anchor frame consistency ───────────────────────────────────────────────
    header("Anchor frame consistency")
    # The anchor frame's camera position in world_A (from submap A directly)
    pos_a = sm_a.frames[-1].position_world

    # The anchor frame's camera position in world_B, mapped to world_A via T
    pos_b_in_a = aligner.transform_points(
        sm_b.frames[0].position_world[None], result.T_a_from_b
    )[0]

    err = float(np.linalg.norm(pos_a - pos_b_in_a))
    print(f"  Anchor pos in world_A:        {pos_a.round(5)}")
    print(f"  Anchor pos in world_B → A:    {pos_b_in_a.round(5)}")
    print(f"  Position error:               {err:.2e} m")
    check(f"Anchor position error < 1e-4 m", err < 1e-4)

    # ── apply_to_submap ───────────────────────────────────────────────────────
    header("apply_to_submap()")
    sm_b_aligned = aligner.apply_to_submap(sm_b, result.T_a_from_b)

    check("aligned submap has same n_frames", sm_b_aligned.n_frames == sm_b.n_frames)
    check("aligned points shape unchanged",
          sm_b_aligned.points_world.shape == sm_b.points_world.shape)

    # Anchor camera position should now match submap A's anchor position
    anchor_pos_aligned = sm_b_aligned.frames[0].position_world
    err2 = float(np.linalg.norm(pos_a - anchor_pos_aligned))
    print(f"  Aligned anchor pos:           {anchor_pos_aligned.round(5)}")
    print(f"  Position error after apply:   {err2:.2e} m")
    check("apply_to_submap anchor error < 1e-4 m", err2 < 1e-4)

    # ── full trajectory ───────────────────────────────────────────────────────
    header("Merged trajectory (world_A frame)")
    pos_a_all = sm_a.positions_world
    pos_b_all = sm_b_aligned.positions_world

    print("  Submap A camera positions:")
    for i, pos in enumerate(pos_a_all):
        print(f"    frame {sm_a.frames[i].seq_idx:03d}  [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    print("  Submap B (aligned) camera positions:")
    for i, pos in enumerate(pos_b_all):
        marker = " ← anchor" if i == 0 else ""
        print(f"    frame {sm_b.frames[i].seq_idx:03d}  [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]{marker}")

    # Check trajectory is continuous at the junction (anchor)
    junction_gap = float(np.linalg.norm(pos_a_all[-1] - pos_b_all[0]))
    inter_a = float(np.linalg.norm(np.diff(pos_a_all, axis=0), axis=1).mean())
    print(f"\n  Mean inter-frame dist in A:   {inter_a:.4f} m")
    print(f"  Junction gap (A[-1]↔B[0]):    {junction_gap:.2e} m")
    check("Junction gap < 1e-3 m (continuous trajectory)", junction_gap < 1e-3)

    # ── optional PLY ─────────────────────────────────────────────────────────
    if args.save_ply:
        header(f"Saving merged PLY → {args.save_ply}")
        pts = np.concatenate([sm_a.points_world, sm_b_aligned.points_world])
        cols = np.concatenate([sm_a.colors, sm_b_aligned.colors])
        _save_ply(pts, cols, args.save_ply)
        print(f"  Saved {len(pts):,} points.")

    header("All checks passed")


def _save_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
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
