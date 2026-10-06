"""
Smoke test for the encoder-token tap (requires GPU + DA3) — Branch C, Step 0.

This is the gate for the whole temporal-redundancy study (hypothesis H0): if
the tokens we capture are not the real encoder output, every redundancy number
measured from them is meaningless.  Four checks:

  0. JITTER BAND        two identical forwards differ by some amount (bf16 is
                        GPU-nondeterministic).  Every "equal" below means
                        "equal within this band", so it is measured first.
  1. ROUND TRIP         capture a batch's tokens, re-run with all of them
                        injected, and require the depth/pose to match the plain
                        forward.  (H0 as written in the plan.)
  2. BATCH INVARIANCE   the same frame's tokens, captured once in batch A and
                        once in batch B with a different frame count and a
                        different position, must agree.  This is what licenses
                        caching a frame's tokens ACROSS submaps — the exact
                        overlap-frame reuse of Step 2 rests on it.
  3. PARTIAL INJECTION  inject one frame, encode the rest — the mixed path Step
                        2 actually uses.

It also reports the encoder prefix's share of backbone wall-clock (capture-only
vs fully-injected forward), i.e. the ceiling on everything temporal reuse can
save in this backbone.

Usage:
    python scripts/test_token_hook.py --image_dir data/tum/rgbd_dataset_freiburg1_desk/rgb
"""

import argparse
import time

import numpy as np
import torch

from smoke_test_utils import header, check, list_images, load_rgb_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--n_frames", type=int, default=8,
                        help="frames in the test batch")
    parser.add_argument("--depth_model", default=None,
                        help="HuggingFace model id (default: DepthEstimator's)")
    parser.add_argument("--resolution", type=int, default=504)
    return parser.parse_args()


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def token_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(max absolute difference, max cosine distance) between two token grids."""
    with torch.inference_mode():
        af, bf = a.float(), b.float()
        max_abs = float((af - bf).abs().max())
        cos = torch.nn.functional.cosine_similarity(af, bf, dim=-1)
        max_cos_dist = float((1.0 - cos).max())
    return max_abs, max_cos_dist


def main():
    args = parse_args()

    paths = list_images(args.image_dir, limit=args.n_frames + 1)
    check(f"Found {len(paths)} images", len(paths) >= args.n_frames + 1)
    images = load_rgb_images(paths)

    from da3_slam.backend.inference.depth_estimator import DepthEstimator

    kwargs = {"process_resolution": args.resolution}
    if args.depth_model:
        kwargs["model_id"] = args.depth_model
    estimator = DepthEstimator(**kwargs)

    batch_a = images[:args.n_frames]          # the reference batch
    batch_b = images[1:args.n_frames + 1]     # shifted: shares frames 1..n-1

    # ── tap description ───────────────────────────────────────────────────────
    header("Tapped backbone")
    tap = estimator.token_tap()
    info = tap.info
    print(f"  {info.describe()}")
    check("prefix is a non-empty proper prefix",
          0 < info.prefix_len < info.n_blocks)

    # ── 0. nondeterminism band ────────────────────────────────────────────────
    header("0. Jitter band (two identical plain forwards)")
    plain_1 = estimator.infer(batch_a)
    plain_2 = estimator.infer(batch_a)
    depth_jitter = max_abs_diff(plain_1.depth, plain_2.depth)
    pose_jitter = max_abs_diff(plain_1.extrinsics, plain_2.extrinsics)
    depth_scale = float(np.median(np.abs(plain_1.depth)))
    print(f"  depth: max |Δ| = {depth_jitter:.3e} m  (median depth {depth_scale:.3f} m)")
    print(f"  pose:  max |Δ| = {pose_jitter:.3e}")
    # Tolerances: the run-to-run band, with a floor so a perfectly deterministic
    # GPU does not set an impossible threshold.
    depth_tol = max(depth_jitter * 2.0, 1e-4 * max(depth_scale, 1e-3))
    pose_tol = max(pose_jitter * 2.0, 1e-5)
    print(f"  → tolerances: depth {depth_tol:.3e} m, pose {pose_tol:.3e}")

    # ── 1. round trip ─────────────────────────────────────────────────────────
    header("1. Round trip: capture → inject all → same output")
    t0 = time.perf_counter()
    captured = estimator.infer(batch_a, capture_tokens=True)
    capture_seconds = time.perf_counter() - t0
    tokens_a = captured.tokens
    check(f"captured {len(tokens_a)} token grids", len(tokens_a) == len(batch_a))
    n_tokens, dim = tokens_a[0].shape
    print(f"  tokens per frame: {n_tokens} x {dim}  ({tokens_a[0].dtype})")
    check("token dim matches the backbone", dim == info.embed_dim)
    check("tokens are finite", bool(torch.isfinite(tokens_a[0].float()).all()))

    check(f"capture leaves depth unchanged "
          f"(max |Δ| = {max_abs_diff(captured.depth, plain_1.depth):.3e})",
          max_abs_diff(captured.depth, plain_1.depth) <= depth_tol)

    inject_all = {i: t for i, t in enumerate(tokens_a)}
    t0 = time.perf_counter()
    injected = estimator.infer(batch_a, inject_tokens=inject_all)
    inject_seconds = time.perf_counter() - t0

    depth_delta = max_abs_diff(injected.depth, plain_1.depth)
    pose_delta = max_abs_diff(injected.extrinsics, plain_1.extrinsics)
    print(f"  depth: max |Δ| = {depth_delta:.3e} m")
    print(f"  pose:  max |Δ| = {pose_delta:.3e}")
    check(f"injected depth == plain depth (tol {depth_tol:.3e})",
          depth_delta <= depth_tol)
    check(f"injected pose == plain pose (tol {pose_tol:.3e})",
          pose_delta <= pose_tol)

    # ── 2. batch-composition invariance ───────────────────────────────────────
    header("2. Batch invariance: same frame, different batch")
    captured_b = estimator.infer(batch_b, capture_tokens=True)
    tokens_b = captured_b.tokens
    # frame i of batch_a (i >= 1) is frame i-1 of batch_b
    worst_abs, worst_cos, worst_frame = 0.0, 0.0, -1
    for i in range(1, len(batch_a)):
        abs_d, cos_d = token_diff(tokens_a[i], tokens_b[i - 1])
        if abs_d > worst_abs:
            worst_abs, worst_frame = abs_d, i
        worst_cos = max(worst_cos, cos_d)
    token_scale = float(tokens_a[0].float().abs().median())
    print(f"  worst frame {worst_frame}: max |Δ| = {worst_abs:.3e} "
          f"(median |token| {token_scale:.3e}), max cosine distance = {worst_cos:.3e}")
    check("tokens are batch-composition independent (cosine distance < 1e-3)",
          worst_cos < 1e-3)

    # Sanity the other way: DIFFERENT frames must NOT look identical, else the
    # comparison above is vacuous (e.g. the tap grabbed a constant tensor).
    other_abs, other_cos = token_diff(tokens_a[0], tokens_a[-1])
    print(f"  different frames differ: max |Δ| = {other_abs:.3e}, "
          f"max cosine distance = {other_cos:.3e}")
    check("distinct frames give distinct tokens", other_cos > worst_cos * 10)

    # ── 3. partial injection ──────────────────────────────────────────────────
    header("3. Partial injection: reuse frame 0, encode the rest")
    partial = estimator.infer(batch_a, inject_tokens={0: tokens_a[0]})
    partial_delta = max_abs_diff(partial.depth, plain_1.depth)
    print(f"  depth: max |Δ| = {partial_delta:.3e} m")
    check(f"partially injected depth == plain depth (tol {depth_tol:.3e})",
          partial_delta <= depth_tol)

    # Cross-batch reuse — the Step 2 overlap-frame case: feed batch_b using the
    # tokens frame 1 got while it was part of batch_a.
    cross = estimator.infer(batch_b, inject_tokens={0: tokens_a[1]})
    cross_delta = max_abs_diff(cross.depth, captured_b.depth)
    print(f"  cross-batch reuse: max |Δ| = {cross_delta:.3e} m")
    check(f"cross-batch injected depth == plain depth (tol {depth_tol:.3e})",
          cross_delta <= depth_tol)

    # ── headroom ──────────────────────────────────────────────────────────────
    header("Encoder prefix share of backbone wall-clock")
    print(f"  full forward     : {capture_seconds:.3f} s")
    print(f"  prefix skipped   : {inject_seconds:.3f} s")
    saved = 1.0 - inject_seconds / capture_seconds if capture_seconds > 0 else 0.0
    print(f"  → prefix is ~{100 * saved:.1f}% of the backbone forward "
          f"({info.prefix_len}/{info.n_blocks} blocks of the {info.branch} branch; "
          "the metric branch is untapped and still runs in full)")
    print("  (single-shot timings — treat as an order of magnitude, not a benchmark)")

    header("All checks passed")


if __name__ == "__main__":
    main()
