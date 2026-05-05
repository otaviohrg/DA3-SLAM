"""
Test da3_slam.factor_graph.

Builds a short sequence of submaps, constructs a pose graph, optimizes,
and verifies the result is consistent with the anchor-based alignment.

Usage:
    python scripts/test_factor_graph.py --image_dir data/video1_30fps
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--n_submaps", type=int, default=3,
                        help="Number of consecutive submaps to build (default: 3)")
    parser.add_argument("--submap_size", type=int, default=8)
    parser.add_argument("--save_ply", default=None)
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
    all_paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )
    check("At least 1 image available", len(all_paths) >= 1)

    from da3_slam.depth_estimator import DepthEstimator
    from da3_slam.submap import SubmapBuilder
    from da3_slam.alignment import SubmapAligner
    from da3_slam.factor_graph import PoseGraph
    from da3_slam.keyframe_selector import KeyframeSelector
    from da3_slam.config import load_slam_config

    slam_cfg = load_slam_config(submap_size=args.submap_size)

    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator)
    aligner = SubmapAligner()

    # ── keyframe selection ────────────────────────────────────────────────────
    header("Keyframe selection")
    selector = KeyframeSelector(slam_cfg.keyframe)
    kf_result = selector.select_paths(all_paths)
    # Limit to enough keyframes for the requested number of submaps
    max_kf = args.submap_size * args.n_submaps - (args.n_submaps - 1)
    kf_indices = kf_result.indices[:max_kf]
    print(f"  Selected {len(kf_indices)} keyframes from {len(all_paths)} frames")
    check("Enough keyframes for requested submaps", len(kf_indices) >= max_kf)

    # ── build submaps ─────────────────────────────────────────────────────────
    header(f"Building {args.n_submaps} submaps")
    submaps = builder.build_sequence(all_paths, kf_indices, submap_size=args.submap_size)
    check(f"{args.n_submaps} submaps built", len(submaps) == args.n_submaps)
    for sm in submaps:
        print(f"  Submap {sm.idx}: frames {sm.frames[0].seq_idx}–{sm.frames[-1].seq_idx}")

    # ── compute alignments ────────────────────────────────────────────────────
    header("Computing alignments")
    alignments = []
    for i in range(len(submaps) - 1):
        a = aligner.align(submaps[i], submaps[i + 1])
        alignments.append(a)
        print(f"  {i}→{i+1}  rot={a.rotation_angle_deg:.3f}°  "
              f"t={a.translation.round(4)}")

    # ── build factor graph ────────────────────────────────────────────────────
    header("Building factor graph")
    graph = PoseGraph(slam_cfg.noise)
    graph.add_submap(submaps[0])
    for i, (sm, alignment) in enumerate(zip(submaps[1:], alignments)):
        graph.add_submap(sm, alignment)

    check(f"n_nodes = {args.n_submaps}", graph.n_nodes == args.n_submaps)
    # prior + between-factors
    check(f"n_factors = {args.n_submaps}", graph.n_factors == args.n_submaps)
    print(f"  Nodes:          {graph.n_nodes}")
    print(f"  Factors:        {graph.n_factors}")
    print(f"  Initial error:  {graph.initial_error():.6f}")

    # ── optimize ──────────────────────────────────────────────────────────────
    header("Optimizing (Levenberg-Marquardt)")
    result = graph.optimize(verbose=True)

    print(f"  Final error:    {result.final_error:.6f}")
    print(f"  Iterations:     {result.iterations}")
    check("Final error < initial error",
          result.final_error <= graph.initial_error() + 1e-9)

    # ── inspect optimized poses ───────────────────────────────────────────────
    header("Optimized submap poses")
    for idx in sorted(result.poses):
        T = result.pose(idx)
        R, t = T[:3, :3], T[:3, 3]
        det = np.linalg.det(R)
        print(f"  Submap {idx}  t={t.round(4)}  det(R)={det:.6f}")
        check(f"Submap {idx} det(R) ≈ 1.0", abs(det - 1.0) < 1e-4)

    # ── verify poses match pre-optimization estimates ─────────────────────────
    header("Pose consistency with anchor alignment")
    # Submap 0 should be at identity (prior fixes it)
    T0 = result.pose(0)
    check("Submap 0 ≈ identity",
          np.allclose(T0, np.eye(4), atol=1e-4))

    # Each subsequent pose should match the accumulated alignment
    accumulated = np.eye(4)
    for i, alignment in enumerate(alignments):
        accumulated = accumulated @ alignment.T_a_from_b
        T_opt = result.pose(i + 1)
        err = np.linalg.norm(T_opt - accumulated)
        print(f"  Submap {i+1}: alignment vs optimized error = {err:.2e}")
        check(f"Submap {i+1} pose matches alignment (err < 1e-3)", err < 1e-3)

    # ── apply optimized poses to rebuild global point cloud ───────────────────
    if args.save_ply:
        header(f"Saving global map → {args.save_ply}")
        all_points, all_colors = [], []
        for sm in submaps:
            T_opt = result.pose(sm.idx)
            # If this is submap 0, T_opt = identity; otherwise apply correction
            sm_aligned = aligner.apply_to_submap(sm, T_opt) if sm.idx > 0 \
                else sm
            all_points.append(sm_aligned.points_world)
            all_colors.append(sm_aligned.colors)

        pts = np.concatenate(all_points)
        cols = np.concatenate(all_colors)
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
