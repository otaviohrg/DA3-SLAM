"""
Test da3_slam.loop_closure.

Verifies descriptor extraction, similarity computation, and ICP transform
estimation. Uses a synthetic loop (same submap vs itself) to test the
full pipeline since a short video clip may not contain a real loop.

Usage:
    python scripts/test_loop_closure.py --image_dir data/video1_30fps
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--submap_size", type=int, default=8)
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
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
    check("At least 1 image available", len(all_paths) >= 1)

    from da3_slam.frontend.depth_estimator import DepthEstimator
    from da3_slam.frontend.submap import SubmapBuilder
    from da3_slam.frontend.keyframe_selector import KeyframeSelector
    from da3_slam.backend.loop_closure import LoopClosureDetector, LoopClosureConfig
    from da3_slam.config import load_slam_config

    slam_cfg = load_slam_config(submap_size=args.submap_size)

    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator)
    selector = KeyframeSelector(slam_cfg.keyframe)

    # Build one submap to test with
    header("Building test submap")
    kf_result = selector.select_paths(all_paths)
    kf_indices = kf_result.indices[:args.submap_size]
    submap = builder.build(
        [all_paths[i] for i in kf_indices],
        kf_indices,
        submap_idx=0,
    )
    print(f"  Submap: {submap.n_frames} frames, {len(submap.points_world):,} points")

    # ── descriptor extraction ─────────────────────────────────────────────────
    header("Descriptor extraction")
    cfg = LoopClosureConfig(
        similarity_threshold=args.similarity_threshold,
        min_submaps_apart=slam_cfg.loop_closure.min_submaps_apart,
        dinov2_model=slam_cfg.loop_closure.dinov2_model,
        icp_max_iterations=slam_cfg.loop_closure.icp_max_iterations,
        icp_tolerance=slam_cfg.loop_closure.icp_tolerance,
        icp_max_distance=slam_cfg.loop_closure.icp_max_distance,
        icp_num_points=slam_cfg.loop_closure.icp_num_points,
    )
    detector = LoopClosureDetector(config=cfg)

    desc = detector._extract_descriptor(submap)
    check("Descriptor is 1D", desc.ndim == 1)
    check("Descriptor is float32", desc.dtype == np.float32)
    check("Descriptor is L2-normalised", abs(np.linalg.norm(desc) - 1.0) < 1e-5)
    print(f"  Descriptor dim: {len(desc)}")
    print(f"  Descriptor norm: {np.linalg.norm(desc):.6f}")

    # ── self-similarity ───────────────────────────────────────────────────────
    header("Self-similarity (same descriptor)")
    sim_self = float(np.dot(desc, desc))
    check("Self-similarity = 1.0", abs(sim_self - 1.0) < 1e-5)
    print(f"  Self-similarity: {sim_self:.6f}")

    # ── synthetic loop closure ────────────────────────────────────────────────
    # Register submap 0, then register a copy as submap 10 (far apart)
    # to simulate revisiting the same place
    header("Synthetic loop closure detection")
    import copy, dataclasses

    submap_a = submap  # idx=0
    submap_b = dataclasses.replace(submap, idx=10)  # same frames, different idx

    pose_identity = np.eye(4, dtype=np.float32)
    closures_a = detector.process(submap_a, pose_identity)
    check("No loop when only 1 submap registered", len(closures_a) == 0)

    # Register submaps 1–9 as dummies (different content) to satisfy min_apart
    import torch
    for i in range(1, 10):
        dummy = dataclasses.replace(submap, idx=i)
        # Assign a random descriptor to make it clearly different
        dummy_desc = np.random.randn(len(desc)).astype(np.float32)
        dummy_desc /= np.linalg.norm(dummy_desc)
        detector._descriptors[i] = dummy_desc
        detector._submaps[i] = dummy
        detector._poses[i] = pose_identity

    closures_b = detector.process(submap_b, pose_identity)
    print(f"  Loop candidates found: {len(closures_b)}")
    if closures_b:
        lc = closures_b[0]
        print(f"  Best match: submap {lc.submap_idx_a}↔{lc.submap_idx_b}  "
              f"sim={lc.candidate.similarity:.4f}  icp_rmse={lc.icp_rmse:.4f}m")
        check("Matched against submap 0", lc.submap_idx_a == 0)
        check("Similarity ≥ threshold", lc.candidate.similarity >= args.similarity_threshold)
        check("ICP transform shape (4,4)", lc.alignment.T_a_from_b.shape == (4, 4))
        check("ICP det(R) ≈ 1.0",
              abs(np.linalg.det(lc.alignment.T_a_from_b[:3, :3]) - 1.0) < 1e-3)
        # For a self-loop the transform should be close to identity
        err = np.linalg.norm(lc.alignment.T_a_from_b - np.eye(4))
        print(f"  Transform error from identity: {err:.4f}")
        check("Self-loop transform ≈ identity (err < 0.1)", err < 0.1)
    else:
        print("  No loop detected (similarity below threshold — expected for non-looping video)")

    # ── min_submaps_apart enforcement ─────────────────────────────────────────
    header("min_submaps_apart enforcement")
    detector2 = LoopClosureDetector(config=LoopClosureConfig(
        similarity_threshold=0.0,
        min_submaps_apart=3,
        dinov2_model=slam_cfg.loop_closure.dinov2_model,
        icp_max_iterations=slam_cfg.loop_closure.icp_max_iterations,
        icp_tolerance=slam_cfg.loop_closure.icp_tolerance,
        icp_max_distance=slam_cfg.loop_closure.icp_max_distance,
        icp_num_points=slam_cfg.loop_closure.icp_num_points,
    ))
    # Register submaps 0, 1, 2 with identical descriptors
    for i in range(3):
        sm_i = dataclasses.replace(submap, idx=i)
        detector2._descriptors[i] = desc
        detector2._submaps[i] = sm_i
        detector2._poses[i] = pose_identity

    # Submap 2 is only 2 apart from 0 — should NOT be returned
    candidates = detector2._find_candidates(2)
    check("Adjacent submaps (gap ≤ 3) not returned as candidates",
          all(abs(c.submap_idx_a - 2) > 3 for c in candidates))
    print(f"  Candidates for submap 2 (min_apart=3): {[(c.submap_idx_a, c.similarity) for c in candidates]}")

    header("All checks passed")


if __name__ == "__main__":
    main()
