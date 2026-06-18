"""
Smoke test for the graph-building path used by DA3SLAM._processing
(requires GPU + DA3, and a GTSAM build with SL4 support).

Builds consecutive submaps from real images exactly like the pipeline
(1-frame anchor overlap), inserts them into the SL(4) PoseGraph with
between-factors, optimises, and verifies that the optimised poses are
consistent with DA3's relative poses.

Usage:
    python scripts/test_factor_graph.py --image_dir data/video1_30fps
"""

import argparse

import numpy as np

from smoke_test_utils import header, check, list_images, load_rgb_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--n_submaps", type=int, default=3,
                        help="Number of consecutive submaps to build")
    parser.add_argument("--submap_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    # n submaps of submap_size frames, each sharing 1 anchor with the next
    needed = args.submap_size * args.n_submaps - (args.n_submaps - 1)
    all_paths = list_images(args.image_dir)
    check(f"At least {needed} images available", len(all_paths) >= needed)
    all_paths = all_paths[:needed]

    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    from da3_slam.backend.inference.submap import SubmapBuilder
    from da3_slam.backend.processing.factor_graph import PoseGraph
    from da3_slam.config import load_slam_config

    slam_cfg = load_slam_config(submap_size=args.submap_size)
    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator)

    # ── build submaps with 1-frame anchor overlap ─────────────────────────────
    header(f"Building {args.n_submaps} submaps (anchor overlap)")
    images = load_rgb_images(all_paths)
    submaps = []
    start = 0
    for submap_idx in range(args.n_submaps):
        end = start + args.submap_size
        seq_indices = list(range(start, end))
        submap = builder.build(all_paths[start:end], images[start:end],
                               seq_indices, submap_idx=submap_idx)
        submaps.append(submap)
        print(f"  Submap {submap.idx}: seq_idx "
              f"{submap.frames[0].seq_idx}–{submap.frames[-1].seq_idx}")
        start = end - 1  # next submap starts at this submap's last frame

    for prev, curr in zip(submaps, submaps[1:]):
        check(f"submaps {prev.idx}/{curr.idx} share anchor",
              prev.frames[-1].seq_idx == curr.frames[0].seq_idx)

    # ── build the graph the way DA3SLAM._processing does ──────────────────────
    header("Building SL(4) pose graph")
    graph = PoseGraph(slam_cfg.noise)

    # First submap: frames at DA3 poses, prior on frame 0.
    # (Scale handling is identity here — single-scale check only; the full
    # cross-submap scale logic is exercised by the end-to-end pipeline.)
    for frame in submaps[0].frames:
        graph.add_frame(frame.seq_idx, frame.cam_to_world.astype(np.float64))
    graph.add_prior(submaps[0].frames[0].seq_idx)

    # Later submaps: place frames via the shared anchor
    for submap in submaps[1:]:
        anchor_global = graph.get_pose(submap.frames[0].seq_idx)
        anchor_local_w2c = submap.frames[0].extrinsic.astype(np.float64)
        for frame in submap.frames[1:]:
            rel = anchor_local_w2c @ frame.cam_to_world.astype(np.float64)
            graph.add_frame(frame.seq_idx, anchor_global @ rel)

    # Between-factors for consecutive frames within each submap
    for submap in submaps:
        for prev, curr in zip(submap.frames, submap.frames[1:]):
            rel = (prev.extrinsic.astype(np.float64)
                   @ curr.cam_to_world.astype(np.float64))
            graph.add_between(prev.seq_idx, curr.seq_idx, rel)

    n_keyframes = needed
    n_factors_expected = 1 + args.n_submaps * (args.submap_size - 1)
    check(f"n_nodes = {n_keyframes}", graph.n_nodes == n_keyframes)
    check(f"n_factors = {n_factors_expected} (1 prior + betweens)",
          graph.n_factors == n_factors_expected)

    # ── optimise ──────────────────────────────────────────────────────────────
    header("Optimizing (Levenberg-Marquardt)")
    result = graph.optimize(verbose=True)
    print(f"  Final error: {result.final_error:.6f}  iters: {result.iterations}")
    check("final error is finite", np.isfinite(result.final_error))

    # ── consistency checks ────────────────────────────────────────────────────
    header("Optimized pose consistency")
    first = submaps[0].frames[0]
    err0 = float(np.linalg.norm(
        result.pose(first.seq_idx) - first.cam_to_world.astype(np.float64)
    ))
    check(f"frame 0 pinned by prior  err={err0:.2e}", err0 < 1e-3)

    # With consistent measurements the optimised relative poses must match
    # the DA3 relative poses within each submap.
    worst = 0.0
    for submap in submaps:
        for prev, curr in zip(submap.frames, submap.frames[1:]):
            measured = (prev.extrinsic.astype(np.float64)
                        @ curr.cam_to_world.astype(np.float64))
            optimised = (np.linalg.inv(result.pose(prev.seq_idx))
                         @ result.pose(curr.seq_idx))
            worst = max(worst, float(np.linalg.norm(measured - optimised)))
    print(f"  Worst relative-pose deviation: {worst:.2e}")
    check("optimised relatives match measurements (err < 1e-3)", worst < 1e-3)

    header("All checks passed")


if __name__ == "__main__":
    main()
