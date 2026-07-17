"""
Smoke test for da3_slam.frontend.keyframe_selector (no GPU required).

Usage:
    python scripts/test_keyframe_selector.py --image_dir data/video1_5fps
"""

import argparse

from da3_slam.config import load_slam_config
from da3_slam.frontend.keyframe_selector import (
    KeyframeSelector,
    OnlineKeyframeSelector,
    SegmentKeyframeSelector,
)
from smoke_test_utils import header, check, list_images, load_rgb_images


def run_segment_selector(cfg_seg, images, threshold):
    """Run SegmentKeyframeSelector over `images` with the given threshold.

    Returns (keyframes, boundaries): the emitted (label, image, seq_idx)
    tuples and the seq_idx of the last frame of each emitted segment.
    """
    cfg_seg.segment_disparity_threshold = threshold
    selector = SegmentKeyframeSelector(cfg_seg)
    keyframes, boundaries = [], []
    for i, image in enumerate(images):
        emitted = selector.step(image, seq_idx=i, label=f"f{i}")
        if emitted:
            keyframes.extend(emitted)
            boundaries.append(emitted[-1][2])
    tail = selector.flush()
    if tail:
        keyframes.extend(tail)
        boundaries.append(tail[-1][2])
    return keyframes, boundaries


def parse_args():
    cfg = load_slam_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--min_disparity_fraction", type=float,
                        default=cfg.keyframe.min_disparity_fraction)
    parser.add_argument("--max_submap_size", type=int,
                        default=cfg.keyframe.max_submap_size)
    return parser.parse_args()


def main():
    args = parse_args()

    paths = list_images(args.image_dir)
    check(f"Found {len(paths)} images", len(paths) > 0)

    cfg = load_slam_config(submap_size=args.max_submap_size).keyframe
    cfg.min_disparity_fraction = args.min_disparity_fraction
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
    check("disparities are non-negative", all(d >= 0.0 for d in result.disparities))

    # ── per-frame disparity table ─────────────────────────────────────────────
    header("Per-frame disparity")
    keyframe_set = set(result.indices)
    for i, disp in enumerate(result.disparities):
        tag = " ← keyframe" if i in keyframe_set else ""
        print(f"  frame {i:03d}  disparity={disp:6.2f}px{tag}")

    header("Summary")
    print(f"  Total frames:    {len(paths)}")
    print(f"  Keyframes:       {result.n_keyframes}")
    print(f"  Keyframe ratio:  {result.n_keyframes / len(paths):.1%}")
    print(f"  Keyframe idx:    {result.indices}")
    print(f"  Min disparity:   {args.min_disparity_fraction} × W")
    print(f"  Max submap size: {args.max_submap_size}")

    # ── select from numpy arrays ──────────────────────────────────────────────
    header("select() from numpy arrays")
    images = load_rgb_images(paths)
    result_np = selector.select(images)
    check("same keyframe indices as path-based", result_np.indices == result.indices)

    # ── online selector agrees with batch selector ────────────────────────────
    header("OnlineKeyframeSelector.step() agrees with batch select()")
    online = OnlineKeyframeSelector(cfg)
    online_indices = [i for i, img in enumerate(images) if online.step(img)]
    check("online indices match batch indices", online_indices == result.indices)

    # ── edge cases ────────────────────────────────────────────────────────────
    header("Edge cases")
    single = selector.select([images[0]])
    check("single frame → 1 keyframe", single.n_keyframes == 1)
    check("single frame index = [0]", single.indices == [0])

    empty = selector.select([])
    check("empty input → 0 keyframes", empty.n_keyframes == 0)

    # ── max_submap_size enforcement ───────────────────────────────────────────
    header("max_submap_size enforcement")
    cfg_tight = load_slam_config(submap_size=3).keyframe
    cfg_tight.min_disparity_fraction = 999.0
    result_tight = KeyframeSelector(cfg_tight).select(images)
    gaps = [result_tight.indices[i + 1] - result_tight.indices[i]
            for i in range(len(result_tight.indices) - 1)]
    check("all gaps <= max_submap_size=3", all(g <= 3 for g in gaps))
    print(f"  Gaps between keyframes: {gaps}")

    # ── segment-level selector ────────────────────────────────────────────────
    header("SegmentKeyframeSelector (segment-level density control)")
    seg_len = 8
    cfg_seg = load_slam_config().keyframe
    cfg_seg.selection_mode = "segment"
    cfg_seg.segment_length = seg_len
    cfg_seg.segment_strides = (2, 4)

    # Very high threshold → never dense → sparse stride (b=4) on full segments.
    kfs_sparse, bounds_sparse = run_segment_selector(cfg_seg, images, threshold=1e12)
    # Very low threshold → always dense → dense stride (a=2) on full segments.
    kfs_dense, _ = run_segment_selector(cfg_seg, images, threshold=-1.0)

    sparse_idx = [seq for _, _, seq in kfs_sparse]
    dense_idx = [seq for _, _, seq in kfs_dense]

    check("segment keyframe indices are sorted & unique",
          sparse_idx == sorted(set(sparse_idx)))
    check("segment indices within bounds",
          all(0 <= s < len(images) for s in sparse_idx))
    n_full = len(images) // seg_len
    if n_full >= 1:
        # Last frame of each full segment must be selected (anchor/bridge).
        last_frames = [(k + 1) * seg_len - 1 for k in range(n_full)]
        check("last frame of each full segment is a keyframe",
              all(lf in sparse_idx for lf in last_frames))
        check("full-segment boundaries recorded", all(lf in bounds_sparse
                                                       for lf in last_frames))
    check("dense stride yields >= as many keyframes as sparse",
          len(dense_idx) >= len(sparse_idx))
    print(f"  Frames: {len(images)}  segment_length: {seg_len}")
    print(f"  Sparse (stride 4) keyframes: {len(sparse_idx)} -> {sparse_idx}")
    print(f"  Dense  (stride 2) keyframes: {len(dense_idx)} -> {dense_idx}")

    header("All checks passed")


if __name__ == "__main__":
    main()
