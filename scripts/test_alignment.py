"""
Smoke test for da3_slam.backend.processing.alignment (requires GPU + DA3).

Builds two consecutive submaps that share an anchor frame (last of A =
first of B), computes the anchor-based alignment, and verifies that the
anchor's camera position is identical after mapping submap B's world frame
into submap A's.

Usage:
    python scripts/test_alignment.py --image_dir data/video1_30fps
"""

import argparse

import numpy as np

from smoke_test_utils import header, check, list_images, load_rgb_images, save_ascii_ply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--submap_size", type=int, default=8)
    parser.add_argument("--confidence_percentile", type=float, default=40.0)
    parser.add_argument("--save_ply", default=None,
                        help="Save merged two-submap point cloud to this .ply")
    return parser.parse_args()


def main():
    args = parse_args()

    # Two submaps of submap_size frames sharing one anchor frame
    needed = args.submap_size * 2 - 1
    all_paths = list_images(args.image_dir)
    check(f"At least {needed} images available", len(all_paths) >= needed)
    all_paths = all_paths[:needed]

    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    from da3_slam.backend.inference.submap import SubmapBuilder, transform_points
    from da3_slam.backend.processing.alignment import SubmapAligner

    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator, confidence_percentile=args.confidence_percentile)
    aligner = SubmapAligner()

    # ── build two overlapping submaps ─────────────────────────────────────────
    header("Building 2 submaps with shared anchor frame")
    images = load_rgb_images(all_paths)
    split = args.submap_size
    # Submap A: frames [0, split); submap B: frames [split-1, end) — the
    # 1-frame overlap mirrors the batching in DA3SLAM._frontend.
    sm_a = builder.build(all_paths[:split], images[:split],
                         list(range(split)), submap_idx=0)
    sm_b = builder.build(all_paths[split - 1:], images[split - 1:],
                         list(range(split - 1, needed)), submap_idx=1)

    print(f"  Submap A: {sm_a.n_frames} frames  "
          f"seq_idx {sm_a.frames[0].seq_idx}–{sm_a.frames[-1].seq_idx}")
    print(f"  Submap B: {sm_b.n_frames} frames  "
          f"seq_idx {sm_b.frames[0].seq_idx}–{sm_b.frames[-1].seq_idx}")
    check("anchor frame seq_idx matches",
          sm_a.frames[-1].seq_idx == sm_b.frames[0].seq_idx)

    # ── alignment ─────────────────────────────────────────────────────────────
    header("SubmapAligner.align()")
    result = aligner.align(sm_a, sm_b)
    print(f"  Rotation angle:   {result.rotation_angle_deg:.4f} deg")
    print(f"  Translation:      {result.translation.round(4)}")
    print(f"  world_b_to_world_a:\n{result.world_b_to_world_a.round(4)}")

    check("T shape (4,4)", result.world_b_to_world_a.shape == (4, 4))
    check("T bottom row = [0,0,0,1]",
          np.allclose(result.world_b_to_world_a[3], [0, 0, 0, 1], atol=1e-5))
    check("det(R) ≈ 1.0",
          abs(np.linalg.det(result.rotation) - 1.0) < 1e-4)

    # ── anchor frame consistency ──────────────────────────────────────────────
    header("Anchor frame consistency")
    # The anchor frame's camera position in world_A (from submap A directly)
    pos_a = sm_a.frames[-1].position_world

    # The anchor frame's camera position in world_B, mapped to world_A via T
    pos_b_in_a = transform_points(
        sm_b.frames[0].position_world[None].astype(np.float64),
        result.world_b_to_world_a.astype(np.float64),
    )[0]

    err = float(np.linalg.norm(pos_a - pos_b_in_a))
    print(f"  Anchor pos in world_A:        {pos_a.round(5)}")
    print(f"  Anchor pos in world_B → A:    {pos_b_in_a.round(5)}")
    print(f"  Position error:               {err:.2e} m")
    check("Anchor position error < 1e-4 m", err < 1e-4)

    # ── merged trajectory continuity ──────────────────────────────────────────
    header("Merged trajectory (world_A frame)")
    pos_a_all = sm_a.positions_world
    pos_b_all = transform_points(
        sm_b.positions_world.astype(np.float64),
        result.world_b_to_world_a.astype(np.float64),
    )

    print("  Submap A camera positions:")
    for i, pos in enumerate(pos_a_all):
        print(f"    frame {sm_a.frames[i].seq_idx:03d}  "
              f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print("  Submap B (aligned) camera positions:")
    for i, pos in enumerate(pos_b_all):
        marker = " ← anchor" if i == 0 else ""
        print(f"    frame {sm_b.frames[i].seq_idx:03d}  "
              f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]{marker}")

    junction_gap = float(np.linalg.norm(pos_a_all[-1] - pos_b_all[0]))
    inter_a = float(np.linalg.norm(np.diff(pos_a_all, axis=0), axis=1).mean())
    print(f"\n  Mean inter-frame dist in A:   {inter_a:.4f} m")
    print(f"  Junction gap (A[-1]↔B[0]):    {junction_gap:.2e} m")
    check("Junction gap < 1e-3 m (continuous trajectory)", junction_gap < 1e-3)

    # ── optional PLY ──────────────────────────────────────────────────────────
    if args.save_ply:
        header(f"Saving merged PLY → {args.save_ply}")
        pts_b_aligned = transform_points(
            sm_b.points_world.astype(np.float64),
            result.world_b_to_world_a.astype(np.float64),
        )
        pts = np.concatenate([sm_a.points_world, pts_b_aligned])
        cols = np.concatenate([sm_a.colors, sm_b.colors])
        save_ascii_ply(pts, cols, args.save_ply)
        print(f"  Saved {len(pts):,} points.")

    header("All checks passed")


if __name__ == "__main__":
    main()
