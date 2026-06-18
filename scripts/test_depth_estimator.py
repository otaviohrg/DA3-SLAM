"""
Test da3_slam.depth_estimator.

Usage:
    python scripts/test_depth_estimator.py --image_dir data/video1_5fps --max_frames 8
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--confidence_percentile", type=float, default=40.0)
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

    # ── load image paths ──────────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )[: args.max_frames]
    check(f"Found {len(paths)} images", len(paths) > 0)

    # ── run inference ─────────────────────────────────────────────────────────
    header("DepthEstimator.infer()")
    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    estimator = DepthEstimator()
    pred = estimator.infer(paths)
    N = len(paths)

    # ── shapes ────────────────────────────────────────────────────────────────
    header("Output shapes")
    check(f"depth shape    = ({N}, H, W)", pred.depth.ndim == 3 and pred.depth.shape[0] == N)
    check(f"confidence shape = ({N}, H, W)", pred.confidence.shape == pred.depth.shape)
    check(f"extrinsics shape = ({N}, 4, 4)", pred.extrinsics.shape == (N, 4, 4))
    check(f"intrinsics shape = ({N}, 3, 3)", pred.intrinsics.shape == (N, 3, 3))
    check(f"n_frames property = {N}", pred.n_frames == N)

    # ── value ranges ──────────────────────────────────────────────────────────
    header("Value ranges")
    check("depth > 0 everywhere", pred.depth.min() > 0)
    check("confidence in [0, 1]", pred.confidence.min() >= 0.0 and pred.confidence.max() <= 1.0)

    for i in range(N):
        R = pred.extrinsics[i, :3, :3]
        det = np.linalg.det(R)
        check(f"frame {i:02d} extrinsics det(R) ≈ 1.0  (got {det:.6f})", abs(det - 1.0) < 1e-3)

    bottom_row = pred.extrinsics[:, 3, :]
    expected = np.array([0, 0, 0, 1], dtype=np.float32)
    check("extrinsics homogeneous row = [0,0,0,1]",
          np.allclose(bottom_row, expected[None], atol=1e-6))

    # ── confidence_mask ───────────────────────────────────────────────────────
    header(f"confidence_mask(percentile={args.confidence_percentile})")
    mask = pred.confidence_mask(args.confidence_percentile)
    check("mask shape matches depth", mask.shape == pred.depth.shape)
    check("mask dtype is bool", mask.dtype == bool)
    for i in range(N):
        kept = mask[i].sum()
        total = mask[i].size
        ratio = kept / total
        check(
            f"frame {i:02d}: {kept}/{total} pixels kept ({ratio:.1%})",
            ratio > 0.0 and ratio < 1.0,
        )

    # ── to_pointcloud ─────────────────────────────────────────────────────────
    header("to_pointcloud()")
    threshold = pred.confidence_threshold(args.confidence_percentile)
    print(f"  global threshold at p{args.confidence_percentile:g}: {threshold:.4f}")
    for i in range(N):
        points, pc_mask = pred.to_pointcloud(i, threshold)
        check(f"frame {i:02d}: points shape = (M, 3)", points.ndim == 2 and points.shape[1] == 3)
        check(f"frame {i:02d}: z > 0 (all points in front of camera)", (points[:, 2] > 0).all())
        print(f"           {len(points):,} points — "
              f"z range [{points[:,2].min():.2f}, {points[:,2].max():.2f}] m")

    header("All checks passed")


if __name__ == "__main__":
    main()
