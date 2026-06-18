"""
Smoke test for da3_slam.backend.processing.loop_closure
(requires GPU + DA3 + DINO-SALAD).

Verifies per-frame descriptor extraction, candidate matching, and DA3
re-inference verification.  Uses a synthetic loop (a copy of the same
submap registered with a distant index) since a short clip may not contain
a real loop.

Usage:
    python scripts/test_loop_closure.py --image_dir data/video1_30fps
"""

import argparse
import dataclasses

import numpy as np

from smoke_test_utils import header, check, list_images, load_rgb_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--submap_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    all_paths = list_images(args.image_dir, limit=args.submap_size)
    check(f"At least {args.submap_size} images available",
          len(all_paths) >= args.submap_size)

    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    from da3_slam.backend.inference.submap import SubmapBuilder
    from da3_slam.backend.processing.loop_closure import (
        LoopClosureDetector, LoopClosureConfig, LoopMatchQueue, LoopCandidate,
    )

    estimator = DepthEstimator()
    builder = SubmapBuilder(estimator)

    # ── build one test submap ─────────────────────────────────────────────────
    header("Building test submap")
    images = load_rgb_images(all_paths)
    submap = builder.build(all_paths, images, list(range(len(all_paths))),
                           submap_idx=0)
    print(f"  Submap: {submap.n_frames} frames, {len(submap.points_world):,} points")

    config = LoopClosureConfig(
        distance_threshold=0.45,
        min_submaps_apart=3,
        max_loop_closures=1,
        min_confidence_ratio=0.0,  # accept everything — we test the mechanics
    )
    detector = LoopClosureDetector(config, builder=builder)

    # ── descriptor extraction ─────────────────────────────────────────────────
    header("Per-frame descriptor extraction")
    descriptors = detector._extract_per_frame_descriptors(submap)
    check("one descriptor per frame", len(descriptors) == submap.n_frames)
    for d in descriptors:
        check("descriptor is 1D float32", d.ndim == 1 and d.dtype == np.float32)
    norms = [float(np.linalg.norm(d)) for d in descriptors]
    check("descriptors are L2-normalised",
          all(abs(n - 1.0) < 1e-4 for n in norms))
    print(f"  Descriptor dim: {len(descriptors[0])}")

    # ── no loop with a single registered submap ───────────────────────────────
    header("No loop with a single submap")
    closures = detector.process(submap)
    check("no closures with only 1 submap", len(closures) == 0)
    check("retrieval vectors stored on frames",
          all(f.retrieval_vector is not None for f in submap.frames))

    # ── synthetic loop: same content registered far away ──────────────────────
    header("Synthetic loop closure (copy of submap at idx=10)")
    submap_copy = dataclasses.replace(submap, idx=10)
    closures = detector.process(submap_copy)
    print(f"  Verified closures: {len(closures)}")
    check("at least one closure found", len(closures) >= 1)

    lc = closures[0]
    check("matched against submap 0", lc.candidate.submap_idx_a == 0)
    check("distance below threshold",
          lc.candidate.distance < config.distance_threshold)
    check("LC submap has 2 frames", lc.lc_submap.n_frames == 2)
    check("LC submap flagged", lc.lc_submap.is_lc_submap)
    check("relative_b_to_a shape (4,4)", lc.relative_b_to_a.shape == (4, 4))
    # Same image pair → the relative pose should be near identity
    identity_err = float(np.linalg.norm(lc.relative_b_to_a - np.eye(4)))
    print(f"  |relative_b_to_a − I| = {identity_err:.4f}  "
          f"confidence = {lc.lc_confidence:.3f}")
    check("self-loop transform ≈ identity (err < 0.1)", identity_err < 0.1)

    # ── min_submaps_apart enforcement ─────────────────────────────────────────
    header("min_submaps_apart enforcement")
    near_copy = dataclasses.replace(submap, idx=12)  # only 2 from idx=10
    candidates = detector._find_candidates(near_copy)
    check("gap-2 submap (idx 10) excluded as candidate",
          all(c.submap_idx_a != 10 for c in candidates))

    # ── LoopMatchQueue (pure data structure) ──────────────────────────────────
    header("LoopMatchQueue keeps the K best candidates")
    queue = LoopMatchQueue(max_size=2)
    for i, dist in enumerate([0.4, 0.1, 0.3, 0.2]):
        queue.push(LoopCandidate(0, 0, 5, i, dist))
    best = queue.get_best()
    check("queue keeps 2 best", [c.distance for c in best] == [0.1, 0.2])

    header("All checks passed")


if __name__ == "__main__":
    main()
