"""
Smoke test for cross-view token merging (da3_slam/backend/inference/token_merge.py).

Two tiers, so most of it runs without the model:

  KERNEL CHECKS (torch only, CPU)   — the merge/unmerge algebra: shapes, the
      identity case, and the structural guarantee the SLAM side depends on —
      that frame 0 (DA3's reference view) and every per-frame camera token are
      merge *destinations*, i.e. never absorbed into another token.  Note that
      destination does NOT mean untouched: ToMe's merge is a symmetric mean, so
      a destination that absorbs sources becomes the mean of itself and them.

  MODEL CHECKS (needs DA3 + GPU)    — the wiring: which blocks get wrapped,
      that merging OFF restores the backbone to within its bf16 noise band,
      that ON changes the answer well beyond that band, and that the min_frames
      gate keeps small (loop-closure) batches on the exact path.

      DA3's forward is bf16 + fused kernels and is NOT bit-reproducible, so the
      noise band is measured from two unmerged runs rather than assumed to be
      zero.

Usage:
    python scripts/test_token_merge.py                       # kernel checks only
    python scripts/test_token_merge.py --image_dir data/tum/<seq>/rgb   # + model

The model checks need a backbone that HAS cross-view attention.  The YAML
default (DA3METRIC-LARGE) is configured with alt_start=-1 and has none, so
--depth_model defaults to nested-giant, matching the benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from smoke_test_utils import check, header, list_images

from da3_slam.backend.inference._tome import token_merge_bipartite2d


# ── kernel checks (no model, no GPU) ──────────────────────────────────────────

def _build(n_frames, w, h, n_special, dim=32, ratio=0.9, protect=True, seed=33):
    tokens_per_img = w * h + n_special
    n = n_frames * tokens_per_img
    torch.manual_seed(0)
    x = torch.randn(1, n, dim)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    merge, unmerge = token_merge_bipartite2d(
        x, w=w, h=h, sx=2, sy=2, r=int(n * ratio), no_rand=False,
        generator=generator, enable_protection=protect,
        n_special=n_special, protect_ratio=0.1,
    )
    return x, merge, unmerge, tokens_per_img


def kernel_checks() -> None:
    header("Kernel: merge / unmerge algebra")

    # DA3 at process_res 504 on 4:3 input: 36x27 patch grid, 1 special token.
    # The height is deliberately ODD — that is the real DA3 geometry, and the
    # destination grid does not tile it evenly.
    n_frames, w, h, n_special = 16, 36, 27, 1
    x, merge, unmerge, tokens_per_img = _build(n_frames, w, h, n_special)
    n = x.shape[1]

    merged = merge(x, mode="mean")
    check(f"merge shortens the sequence ({n} -> {merged.shape[1]})",
          merged.shape[1] < n)
    check("merged token ratio is a real reduction (< 0.5x)",
          merged.shape[1] / n < 0.5)

    restored = unmerge(merged)
    check(f"unmerge restores full length ({restored.shape[1]} == {n})",
          restored.shape[1] == n)
    check("unmerge preserves dtype and dim",
          restored.dtype == x.dtype and restored.shape[-1] == x.shape[-1])
    check("every restored row is populated (no zero rows left behind)",
          bool((restored.abs().sum(-1) > 0).all()))

    # Some patch token outside frame 0 really did get merged, or the whole
    # exercise is a no-op.
    tail = slice(tokens_per_img, n)
    check("patch tokens outside frame 0 are actually merged",
          not torch.allclose(restored[:, tail], x[:, tail], atol=1e-6))

    # --- who is a destination -------------------------------------------------
    # NOTE: being a destination means "never absorbed into another token", NOT
    # "left untouched".  ToMe's merge is a symmetric mean, so a destination that
    # absorbs sources becomes the mean of itself and them.  That matters for
    # DA3: the per-frame camera token is a destination, so it is never deleted,
    # but its q/k/v ARE averaged with whatever patch tokens land on it — which
    # is exactly why the pose head has to be measured, not assumed (plan R1).
    #
    # The observable signature of a destination is therefore uniqueness, not
    # equality: absorbed tokens are given their destination's output verbatim,
    # so any two tokens sharing an output value are in the same group, and two
    # destinations can never share one.  Route a payload through the merge and
    # look for collisions.  The payload must be *random* — an index payload
    # produces group means that collide with plain indices by arithmetic
    # accident (mean(788, 1126) == 957), which says nothing about grouping.
    torch.manual_seed(1)
    payload = torch.randn(1, n, 1, dtype=torch.float64)
    grouped = unmerge(merge(payload, mode="mean")).reshape(-1)

    def all_distinct(indices) -> bool:
        values = grouped[list(indices)]
        return len(torch.unique(values)) == len(values)

    frame0 = list(range(tokens_per_img))
    check("all of frame 0 is held out as destination (the anchor view)",
          all_distinct(frame0))

    special = [i * tokens_per_img + j
               for i in range(n_frames) for j in range(n_special)]
    check("every per-frame special (camera) token is a destination",
          all_distinct(special))
    check("no camera token is grouped with a frame-0 token",
          all_distinct(sorted(set(special) | set(frame0))))

    # Without protection every merged slot yields exactly one output value.
    x_np, merge_np2, unmerge_np2, _ = _build(n_frames, w, h, n_special,
                                             protect=False)
    grouped_np = unmerge_np2(merge_np2(payload, mode="mean")).reshape(-1)
    check("distinct output values == merged length (group bookkeeping is sound)",
          len(torch.unique(grouped_np))
          == merge_np2(payload, mode="mean").shape[1])

    # With protection the count is *lower*, because a protected token that is
    # also a destination sits in the merged sequence twice — once averaged,
    # once pure — and the pure copy wins on unmerge.  That double-representation
    # is FastVGGT's own behaviour (plan risk R4), reproduced deliberately.
    n_groups = len(torch.unique(grouped))
    check("protection only ever collapses groups, never invents them",
          n_groups <= merged.shape[1])
    print(f"  [info] protection double-counts {merged.shape[1] - n_groups} "
          f"of {merged.shape[1]} merged slots (plan risk R4)")

    # Protected tokens are the one set that really does pass through untouched:
    # they are gathered aside, never reduced, and scattered back last.
    num_protected = int(n * 0.1)
    step = max(1, n // num_protected)
    protected = torch.arange(0, n, step)[:num_protected]
    check("protected tokens round-trip exactly",
          torch.allclose(restored[:, protected], x[:, protected], atol=1e-6))

    header("Kernel: degenerate and multi-tensor cases")

    _, merge0, unmerge0, _ = _build(n_frames, w, h, n_special, ratio=0.0)
    check("r <= 0 is a no-op (do_nothing passthrough)",
          torch.allclose(merge0(x), x))

    x2, merge2, _, _ = _build(n_frames, w, h, n_special)
    k = torch.randn_like(x2)
    v = torch.randn_like(x2)
    qm, km, vm = merge2(x2, mode="mean", extra_tensors=k, extra_tensors_2=v)
    check("q/k/v merge to the same length (the attention path)",
          qm.shape == km.shape == vm.shape)

    _, merge_np, unmerge_np, _ = _build(n_frames, w, h, n_special, protect=False)
    check("protect=False still round-trips to full length",
          unmerge_np(merge_np(x)).shape[1] == n)

    header("Kernel: guards")

    try:
        _build(n_frames, 37, h, n_special)   # width not divisible by sx=2
        raised = False
    except AssertionError:
        raised = True
    check("indivisible grid width is rejected (would mis-scatter dst)", raised)

    torch.manual_seed(0)
    bad = torch.randn(1, 12345, 32)
    try:
        token_merge_bipartite2d(bad, w=36, h=27, sx=2, sy=2, r=100,
                                generator=None, n_special=1)
        raised = False
    except AssertionError:
        raised = True
    check("token count that does not factor into frames is rejected", raised)


# ── model checks (needs DA3 + GPU) ────────────────────────────────────────────

def model_checks(image_dir: str, n_frames: int, model: str) -> None:
    from da3_runner import resolve_model_alias
    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    from da3_slam.backend.inference.token_merge import TokenMerger
    from da3_slam.config import TokenMergingConfig, load_slam_config

    config = load_slam_config()
    images = list_images(image_dir, limit=n_frames)
    if len(images) < n_frames:
        raise SystemExit(f"need >= {n_frames} images in {image_dir}")

    # NOTE: config/default.yaml ships DA3METRIC-LARGE, which is configured with
    # alt_start = -1 — it has no cross-view attention, so there is nothing to
    # merge and TokenMerger refuses it.  The benchmarks run nested-giant, whose
    # anyview branch has 14 global blocks; that is what this test exercises.
    estimator = DepthEstimator(
        model_id=resolve_model_alias(model),
        process_resolution=config.depth_model_resolution,
    )

    header("Model: block discovery")
    merger = TokenMerger(estimator.model)
    print(f"  {merger.describe()}")
    vit = merger._vit
    check("global blocks are the odd indices from alt_start",
          all(i >= vit.alt_start and i % 2 == 1 for i in merger.global_blocks))
    check("at least one global block was found", len(merger.global_blocks) > 0)

    header("Model: OFF restores the backbone, ON changes the answer")
    # DA3's forward is bf16 + fused kernels and therefore NOT bit-reproducible,
    # so "identical" has to mean "within the run-to-run noise band".  Measure
    # that band first from two unmerged runs — without it, a detach check either
    # fails spuriously or passes vacuously.
    estimator.set_token_merging(None)
    baseline = estimator.infer(images)
    repeat = estimator.infer(images)
    noise = float(abs(repeat.depth - baseline.depth).max())
    noise_pose = float(abs(repeat.extrinsics - baseline.extrinsics).max())
    print(f"  [info] bf16 noise floor over two unmerged runs: "
          f"depth {noise:.2e} m, pose {noise_pose:.2e}")

    estimator.set_token_merging(TokenMergingConfig(enable=True, min_frames=n_frames))
    merged = estimator.infer(images)

    estimator.set_token_merging(None)          # detach
    restored = estimator.infer(images)

    d_restored = float(abs(restored.depth - baseline.depth).max())
    d_merged = float(abs(merged.depth - baseline.depth).max())
    check(f"after detach, depth is back inside the noise band "
          f"({d_restored:.2e} <= {noise:.2e})",
          d_restored <= max(noise, 1e-12))
    check(f"merging ON changes the prediction well beyond the noise "
          f"({d_merged:.2e} >> {noise:.2e})",
          d_merged > 10 * max(noise, 1e-12))

    p_merged = float(abs(merged.extrinsics - baseline.extrinsics).max())
    print(f"  [info] merged-vs-baseline deltas: depth {d_merged:.2e} m, "
          f"pose {p_merged:.2e} (vs pose noise {noise_pose:.2e}) — "
          "plan Step 1 turns exactly this into T1")

    header("Model: the min_frames gate")
    estimator.set_token_merging(
        TokenMergingConfig(enable=True, min_frames=n_frames + 1))
    merger_on = estimator._merger
    merger_on.reset_stats()
    gated = estimator.infer(images)
    check("a batch below min_frames is never merged",
          merger_on.stats.calls_merged == 0)
    check("...and its output is back inside the noise band",
          float(abs(gated.depth - baseline.depth).max()) <= max(noise, 1e-12))

    estimator.set_token_merging(TokenMergingConfig(enable=True, min_frames=n_frames))
    merger_on = estimator._merger
    merger_on.reset_stats()
    estimator.infer(images)
    stats = merger_on.stats
    print(f"  {stats.describe()}")
    check("every global block merged on a full-size batch",
          stats.calls_merged == len(merger.global_blocks))
    check("realised token ratio is a real reduction", stats.token_ratio < 0.5)

    estimator.set_token_merging(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", default=None,
                        help="Run the model checks too (needs DA3 + GPU)")
    parser.add_argument("--n_frames", type=int, default=16,
                        help="Frames per batch for the model checks")
    parser.add_argument("--depth_model", default="nested-giant",
                        help="Model alias or HF ID for the model checks.  Must "
                             "have cross-view attention: the YAML default "
                             "DA3METRIC-LARGE has alt_start=-1 and cannot merge")
    args = parser.parse_args()

    kernel_checks()
    if args.image_dir:
        model_checks(args.image_dir, args.n_frames, args.depth_model)
    else:
        print("\n(skipping model checks — pass --image_dir to run them)")

    print("\nAll checks passed.\n")


if __name__ == "__main__":
    main()
