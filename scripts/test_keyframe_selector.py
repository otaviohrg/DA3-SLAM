"""
Test da3_slam.keyframe_selector.

Usage:
    python scripts/test_keyframe_selector.py --image_dir data/video1_5fps
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--min_disparity_frac", type=float, default=0.15,
                        help="Min disparity as fraction of image width (default: 0.15)")
    parser.add_argument("--max_submap_size", type=int, default=8)
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
    )
    check(f"Found {len(paths)} images", len(paths) > 0)

    from da3_slam.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig

    cfg = KeyframeSelectorConfig(
        min_disparity_frac=args.min_disparity_frac,
        max_submap_size=args.max_submap_size,
    )
    selector = KeyframeSelector(cfg)

    # ── select from paths ─────────────────────────────────────────────────────
    header("select_paths()")
    result = selector.select_paths(paths)

    check("frame 0 is always a keyframe", result.indices[0] == 0)
    check("at least 1 keyframe", result.n_keyframes >= 1)
    check("indices are sorted", result.indices == sorted(result.indices))
    check("indices within bounds", max(result.indices) < len(paths))
    check("disparity length == n_frames", len(result.disparities) == len(paths))
    check("first disparity is 0.0", result.disparities[0] == 0.0)
    check("non-keyframe disparities >= 0", all(d >= 0.0 for d in result.disparities))

    # ── per-frame disparity table ─────────────────────────────────────────────
    header("Per-frame disparity")
    for i, (disp, path) in enumerate(zip(result.disparities, paths)):
        is_kf = i in result.indices
        tag = " ← keyframe" if is_kf else ""
        print(f"  frame {i:03d}  disparity={disp:6.2f}px{tag}")

    header("Summary")
    print(f"  Total frames:    {len(paths)}")
    print(f"  Keyframes:       {result.n_keyframes}")
    print(f"  Keyframe ratio:  {result.n_keyframes / len(paths):.1%}")
    print(f"  Keyframe idx:    {result.indices}")
    print(f"  Min disparity:   {args.min_disparity_frac} × W")
    print(f"  Max submap size: {args.max_submap_size}")

    # ── select from numpy arrays ──────────────────────────────────────────────
    header("select() from numpy arrays")
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    result_np = selector.select(images)
    check("same keyframe indices as path-based", result_np.indices == result.indices)

    # ── edge cases ────────────────────────────────────────────────────────────
    header("Edge cases")
    single = selector.select([images[0]])
    check("single frame → 1 keyframe", single.n_keyframes == 1)
    check("single frame index = [0]", single.indices == [0])

    empty = selector.select([])
    check("empty input → 0 keyframes", empty.n_keyframes == 0)

    # ── max_submap_size enforcement ───────────────────────────────────────────
    header("max_submap_size enforcement")
    cfg_tight = KeyframeSelectorConfig(min_disparity_frac=999.0, max_submap_size=3)
    result_tight = KeyframeSelector(cfg_tight).select(images)
    gaps = [result_tight.indices[i+1] - result_tight.indices[i]
            for i in range(len(result_tight.indices) - 1)]
    check("all gaps <= max_submap_size=3", all(g <= 3 for g in gaps))
    print(f"  Gaps between keyframes: {gaps}")

    header("All checks passed")


if __name__ == "__main__":
    main()
