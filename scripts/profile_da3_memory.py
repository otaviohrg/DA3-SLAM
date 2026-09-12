"""
Attribute DA3's peak GPU memory to a stage of the forward pass.

WHY
---
Peak memory is what gates submap size — it is why submap 64 OOMs on a 20 GB
card — and the token-merging experiment showed it is NOT set by cross-view
attention (merging cut the attention token count to 0.39x and moved peak memory
by 1 MB out of 12.4 GB; both arms OOM at exactly the same submap size).  So
something else owns the high-water mark.  This script finds out what.

HOW
---
Two views, from one forward pass:

  TIMELINE — `torch.cuda.max_memory_allocated()` is monotonic between resets,
  so sampling it at every module boundary localises the peak exactly: the stage
  during which the sampled value last rose to its final value is the stage that
  set the peak.  No estimation, no attribution heuristics.  Each stage also
  reports what it *keeps* (allocated at exit − allocated at entry), which
  separates transient working memory from resident tensors.

  SCALING — peak memory as a function of frame count and processing resolution.
  The exponents say which term dominates without needing to read the model:
  ViT activations scale with (frames × tokens) = frames × (res/14)^2, and so
  does a DPT head, but the head works at *pixel* resolution rather than 1/14 of
  it, so its constant is ~200x larger.  A peak that scales with frames×res^2 and
  is far too large for the token tensors is the head.

Optionally dumps a `torch.cuda.memory._snapshot()` pickle for the official
viewer (https://docs.pytorch.org/memory_viz) — that gives per-allocation stack
traces if the stage-level answer is not specific enough.

Usage:
    python scripts/profile_da3_memory.py --image_dir data/tum/<seq>/rgb
    python scripts/profile_da3_memory.py --image_dir DIR --n_frames 16 --scaling
    python scripts/profile_da3_memory.py --image_dir DIR --snapshot mem.pickle
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from smoke_test_utils import list_images

MB = 1 / (1024 * 1024)


@dataclass
class Event:
    """One module boundary, with the allocator state observed at it."""

    name: str
    phase: str          # "enter" | "exit"
    depth: int
    allocated: int      # currently resident bytes
    peak: int           # high-water bytes so far (monotonic between resets)
    order: int


@dataclass
class Profiler:
    """Samples the CUDA allocator at every hooked module boundary."""

    device: torch.device
    max_depth: int = 3
    events: list[Event] = field(default_factory=list)
    _handles: list = field(default_factory=list)
    _order: int = 0

    def _sample(self, name: str, phase: str, depth: int) -> None:
        # synchronize() so the allocator numbers correspond to work that has
        # actually happened, not to queued kernels.
        torch.cuda.synchronize(self.device)
        self.events.append(Event(
            name=name, phase=phase, depth=depth,
            allocated=torch.cuda.memory_allocated(self.device),
            peak=torch.cuda.max_memory_allocated(self.device),
            order=self._order))
        self._order += 1

    def attach(self, model: torch.nn.Module, include: tuple[str, ...] = ()) -> None:
        """Hook every module at or above `max_depth`, plus any name substring
        in `include` (used to drill into one stage without hooking all 40 ViT
        blocks)."""
        for name, module in model.named_modules():
            depth = name.count(".") + 1 if name else 0
            wanted = depth <= self.max_depth or any(i in name for i in include)
            if not wanted:
                continue
            label = name or "<root>"
            self._handles.append(module.register_forward_pre_hook(
                lambda m, a, _l=label, _d=depth: self._sample(_l, "enter", _d)))
            self._handles.append(module.register_forward_hook(
                lambda m, a, o, _l=label, _d=depth: self._sample(_l, "exit", _d)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ── reporting ─────────────────────────────────────────────────────────────

    def stages(self) -> list[dict]:
        """Pair enter/exit events into stages with their memory behaviour."""
        open_stack: list[Event] = []
        out: list[dict] = []
        for event in self.events:
            if event.phase == "enter":
                open_stack.append(event)
                continue
            while open_stack and open_stack[-1].name != event.name:
                open_stack.pop()          # a child that never fired its exit
            if not open_stack:
                continue
            start = open_stack.pop()
            out.append({
                "name": event.name,
                "depth": event.depth,
                "order": start.order,
                "kept_mb": (event.allocated - start.allocated) * MB,
                "peak_at_entry_mb": start.peak * MB,
                "peak_at_exit_mb": event.peak * MB,
                "raised_peak_mb": (event.peak - start.peak) * MB,
                "resident_at_exit_mb": event.allocated * MB,
            })
        return sorted(out, key=lambda s: s["order"])

    def peak_setter(self) -> tuple[Event, Event] | None:
        """The two consecutive samples between which the global peak was hit.

        Because `max_memory_allocated` only ever rises, the last increase
        pinpoints the peak in execution order — no guessing.
        """
        if not self.events:
            return None
        final_peak = max(e.peak for e in self.events)
        for previous, current in zip(self.events, self.events[1:]):
            if previous.peak < final_peak <= current.peak:
                return previous, current
        return None


def profile_once(estimator, images: list[str], max_depth: int,
                 include: tuple[str, ...]) -> Profiler:
    device = estimator.device
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    profiler = Profiler(device=device, max_depth=max_depth)
    profiler.attach(estimator.model, include=include)
    try:
        estimator.infer(images)
    finally:
        profiler.detach()
    return profiler


def report_weights(model: torch.nn.Module, device) -> float:
    """Resident bytes before any activation exists — i.e. the parameters.

    Reported first because it turns out to be most of the peak, and unlike
    activations it does not shrink with submap size: it is a floor under every
    configuration.  The dtype breakdown is the actionable part — DA3 runs its
    heads under `torch.autocast(..., enabled=False)`, so fp32 parameters are a
    deliberate choice, not an oversight, but they set that floor.
    """
    torch.cuda.synchronize(device)
    resident = torch.cuda.memory_allocated(device) * MB

    by_dtype: dict[torch.dtype, list[int]] = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        entry = by_dtype.setdefault(tensor.dtype, [0, 0])
        entry[0] += tensor.numel()
        entry[1] += tensor.numel() * tensor.element_size()

    print(f"\n{'─' * 78}\n  PARAMETERS (the frame-independent floor)\n{'─' * 78}")
    print(f"  resident on device before the forward: {resident:.0f} MB "
          f"({resident / 1024:.2f} GB)")
    print(f"  {'dtype':<16} {'params':>14} {'MB':>10}")
    total_mb = 0.0
    for dtype, (count, nbytes) in sorted(by_dtype.items(), key=lambda kv: -kv[1][1]):
        total_mb += nbytes * MB
        print(f"  {str(dtype):<16} {count:>14,} {nbytes * MB:>10.0f}")
    print(f"  {'total':<16} {'':>14} {total_mb:>10.0f}")

    fp32 = by_dtype.get(torch.float32)
    if fp32:
        print(f"\n  → casting the fp32 parameters to bf16 would free "
              f"~{fp32[1] * MB / 2:.0f} MB.")
    return resident


def report(profiler: Profiler, n_frames: int,
           weights_mb: float = 0.0) -> None:
    stages = profiler.stages()
    if not stages:
        print("  no stages recorded (are the hooks attached to the right model?)")
        return

    peak_mb = max(s["peak_at_exit_mb"] for s in stages)
    print(f"\n{'─' * 78}")
    print(f"  PEAK: {peak_mb:.0f} MB ({peak_mb / 1024:.2f} GB) over "
          f"{n_frames} frames")
    if weights_mb:
        print(f"        of which {weights_mb:.0f} MB "
              f"({100 * weights_mb / peak_mb:.0f}%) is parameters, "
              f"{peak_mb - weights_mb:.0f} MB is activations")
    print(f"{'─' * 78}")

    setter = profiler.peak_setter()
    if setter:
        before, after = setter
        print(f"  Peak reached between  {before.name} [{before.phase}]"
              f"  →  {after.name} [{after.phase}]")

    print(f"\n  Stages that RAISED the high-water mark "
          f"(the ones that actually cost memory):")
    print(f"  {'stage':<44} {'raised':>9} {'kept':>9} {'peak@exit':>10}")
    raisers = [s for s in stages if s["raised_peak_mb"] > 1.0]
    raisers.sort(key=lambda s: -s["raised_peak_mb"])
    for stage in raisers[:18]:
        indent = "  " * min(stage["depth"], 3)
        name = (indent + stage["name"])[:44]
        print(f"  {name:<44} {stage['raised_peak_mb']:>8.0f}M "
              f"{stage['kept_mb']:>8.0f}M {stage['peak_at_exit_mb']:>9.0f}M")
    if not raisers:
        print("    (none above 1 MB — the peak is set outside the hooked "
              "modules, e.g. in preprocessing)")

    print(f"\n  Top-level stages in execution order:")
    print(f"  {'stage':<44} {'raised':>9} {'kept':>9} {'peak@exit':>10}")
    for stage in stages:
        if stage["depth"] > 2:
            continue
        indent = "  " * min(stage["depth"], 3)
        name = (indent + stage["name"])[:44]
        print(f"  {name:<44} {stage['raised_peak_mb']:>8.0f}M "
              f"{stage['kept_mb']:>8.0f}M {stage['peak_at_exit_mb']:>9.0f}M")


def cast_backbones_bf16(model) -> int:
    """Cast the ViT backbones to bf16, reconciling the head dtype boundary.

    The backbones already run under autocast bf16, so their fp32 master copies
    are storage the forward never benefits from — autocast re-casts them on
    every matmul anyway.  But DA3 runs its heads under
    `torch.autocast(..., enabled=False)`, so nothing converts the (now bf16)
    backbone output back: a bare `.to(bfloat16)` dies with "mat1 and mat2 must
    have the same dtype" inside cam_dec.  A forward hook on each backbone casts
    its output tree back to fp32 at exactly that boundary.

    Returns the number of backbones cast.  EXPERIMENTAL — this changes numerics
    (measured at ~300x the bf16 run-to-run noise floor on poses), so it is a
    profiling knob, not a supported configuration.
    """
    from da3_slam.backend.inference.token_tap import _resolve_net
    nets = [_resolve_net(model, "anyview")]
    try:
        nets.append(_resolve_net(model, "metric"))
    except (ValueError, TypeError):
        pass                                   # not a nested model

    def to_f32(x):
        if torch.is_tensor(x):
            return x.float()
        if isinstance(x, (list, tuple)):
            return type(x)(to_f32(v) for v in x)
        return x

    count = 0
    for net in nets:
        net.backbone.to(torch.bfloat16)
        net.backbone.register_forward_hook(lambda m, a, o: to_f32(o))
        count += 1
    return count


def _run_once(estimator, images, device) -> tuple[float | None, float | None, float | None]:
    """(peak MB, backbone seconds, merged token ratio) for one forward, or
    (None, None, None) on OOM — with the allocator reset so the next cell is
    not poisoned by the failed one."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    estimator.reset_stats()
    merger = getattr(estimator, "_merger", None)
    if merger is not None:
        merger.reset_stats()
    try:
        estimator.infer(images)
    except Exception as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return None, None, None
    torch.cuda.synchronize(device)
    ratio = merger.stats.token_ratio if merger is not None else None
    return (torch.cuda.max_memory_allocated(device) * MB,
            estimator.backbone_seconds, ratio)


def scaling(estimator, all_images: list[str], frame_counts: list[int],
            resolutions: list[int], arms: list[str]) -> None:
    """Peak memory and latency vs sequence length, for each requested variant.

    Variants are run fp32-first because casting the backbones to bf16 cannot be
    undone in-process without reloading the checkpoint.
    """
    from da3_slam.config import TokenMergingConfig
    device = estimator.device
    base_res = estimator.process_resolution
    rows: dict[tuple[str, int], tuple] = {}

    def measure(label: str, merge: bool) -> None:
        estimator.set_token_merging(
            TokenMergingConfig(enable=True, min_frames=4) if merge else None)
        estimator.infer([all_images[0]] * 4)          # warm this configuration
        for n in frame_counts:
            images = [all_images[i % len(all_images)] for i in range(n)]
            rows[(label, n)] = _run_once(estimator, images, device)
            peak, secs, ratio = rows[(label, n)]
            state = f"{peak:8.0f} MB {secs:7.2f} s" if peak else "     OOM"
            print(f"    {label:<14} N={n:<4} {state}")

    print(f"\n{'─' * 78}\n  SCALING vs SEQUENCE LENGTH (resolution {base_res})"
          f"\n{'─' * 78}")
    if "fp32" in arms:
        measure("fp32", merge=False)
    if "fp32+merge" in arms:
        measure("fp32+merge", merge=True)

    if any(a.startswith("bf16") for a in arms):
        n_cast = cast_backbones_bf16(estimator.model)
        print(f"\n  [cast {n_cast} ViT backbone(s) to bf16 — EXPERIMENTAL, "
              f"changes numerics]")
        if "bf16" in arms:
            measure("bf16", merge=False)
        if "bf16+merge" in arms:
            measure("bf16+merge", merge=True)

    estimator.set_token_merging(None)
    estimator.process_resolution = base_res

    # ── the table ─────────────────────────────────────────────────────────────
    labels = [a for a in ("fp32", "fp32+merge", "bf16", "bf16+merge")
              if any(k[0] == a for k in rows)]
    print(f"\n  PEAK MEMORY (MB)")
    _grid(rows, labels, frame_counts, index=0, fmt="{:.0f}")
    print(f"\n  BACKBONE LATENCY (s)")
    _grid(rows, labels, frame_counts, index=1, fmt="{:.2f}")

    # Fit peak = constant + slope*N per variant: the split between the
    # frame-independent floor and the per-frame cost is the whole story of what
    # gates submap size.
    print(f"\n  FIT  peak(MB) = constant + slope x frames"
          f"   (and the length at which they cross)")
    print(f"  NOTE: 'max N' extrapolates the linear fit and is OPTIMISTIC — the"
          f" real wall\n        arrives earlier from fragmentation and "
          f"transient spikes.  Trust the OOM\n        column above for where it"
          f" actually is.")
    print(f"  {'variant':<14} {'constant MB':>12} {'MB/frame':>10} "
          f"{'floor=activations at N':>24} {'max N in 20GB':>14}")
    import numpy as np
    for label in labels:
        pts = [(n, rows[(label, n)][0]) for n in frame_counts
               if rows.get((label, n)) and rows[(label, n)][0]]
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        const, slope = np.linalg.lstsq(
            np.vstack([np.ones_like(xs), xs]).T, ys, rcond=None)[0]
        cross = const / slope if slope > 0 else float("inf")
        budget = 20 * 1024 * 0.95            # leave headroom for fragmentation
        max_n = (budget - const) / slope if slope > 0 else float("inf")
        print(f"  {label:<14} {const:>12.0f} {slope:>10.1f} "
              f"{cross:>24.0f} {max_n:>14.0f}")

    if resolutions:
        print(f"\n  Peak vs resolution ({frame_counts[0]} frames, last variant):")
        print(f"  {'res':>7} {'tokens/frame':>13} {'peak MB':>10} {'vs 1st':>8}")
        images = [all_images[i % len(all_images)] for i in range(frame_counts[0])]
        first = None
        for res in resolutions:
            estimator.process_resolution = res
            peak, _, _ = _run_once(estimator, images, device)
            if peak is None:
                print(f"  {res:>7} {'':>13} {'OOM':>10}")
                continue
            first = first or peak
            print(f"  {res:>7} {(res // 14) ** 2:>13} {peak:>10.0f} "
                  f"{peak / first:>7.2f}x")
        estimator.process_resolution = base_res


def _grid(rows: dict, labels: list[str], frame_counts: list[int],
          index: int, fmt: str) -> None:
    print("  " + f"{'N':>6}" + "".join(f"{l:>14}" for l in labels))
    for n in frame_counts:
        line = f"  {n:>6}"
        for label in labels:
            value = rows.get((label, n))
            cell = fmt.format(value[index]) if value and value[index] else "OOM"
            line += f"{cell:>14}"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_dir", required=True, help="Directory of frames")
    p.add_argument("--n_frames", type=int, default=16,
                   help="Frames in the profiled batch")
    p.add_argument("--depth_model", default="nested-giant",
                   help="Model alias or HF ID")
    p.add_argument("--resolution", type=int, default=504,
                   help="DA3 processing resolution")
    p.add_argument("--max_depth", type=int, default=3,
                   help="Hook modules down to this depth in the module tree")
    p.add_argument("--include", nargs="*", default=[],
                   help="Extra name substrings to hook regardless of depth "
                        "(e.g. 'head.' to drill into the DPT head)")
    p.add_argument("--scaling", action="store_true",
                   help="Also sweep peak memory vs frames and vs resolution")
    p.add_argument("--frame_counts", nargs="+", type=int,
                   default=[4, 8, 16, 32],
                   help="Sequence lengths (submap sizes) to sweep")
    p.add_argument("--arms", nargs="+",
                   default=["fp32"],
                   choices=["fp32", "fp32+merge", "bf16", "bf16+merge"],
                   help="Variants to sweep.  bf16* casts the ViT backbones to "
                        "bf16 (EXPERIMENTAL, changes numerics) and is applied "
                        "after every fp32 arm, since it cannot be undone "
                        "in-process")
    p.add_argument("--resolutions", nargs="+", type=int,
                   default=[280, 392, 504])
    p.add_argument("--snapshot", default=None,
                   help="Also dump a torch memory snapshot pickle here "
                        "(view at https://docs.pytorch.org/memory_viz)")
    args = p.parse_args()

    from da3_runner import resolve_model_alias
    from da3_slam.backend.inference.depth_estimator import DepthEstimator

    images_all = list_images(args.image_dir)
    if len(images_all) < 1:
        raise SystemExit(f"no images in {args.image_dir}")
    images = [images_all[i % len(images_all)] for i in range(args.n_frames)]

    estimator = DepthEstimator(model_id=resolve_model_alias(args.depth_model),
                               process_resolution=args.resolution)
    weights_mb = report_weights(estimator.model, estimator.device)

    if args.snapshot:
        try:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        except Exception as exc:
            print(f"  [warn] could not start memory history: {exc}")
            args.snapshot = None

    estimator.infer(images)      # warm up: allocator pools + any lazy init
    profiler = profile_once(estimator, images, args.max_depth,
                            tuple(args.include))
    report(profiler, args.n_frames, weights_mb)

    if args.snapshot:
        try:
            torch.cuda.memory._dump_snapshot(args.snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"\n  snapshot → {args.snapshot}  "
                  f"(open at https://docs.pytorch.org/memory_viz)")
        except Exception as exc:
            print(f"  [warn] snapshot dump failed: {exc}")

    if args.scaling:
        scaling(estimator, images_all, args.frame_counts, args.resolutions,
                args.arms)


if __name__ == "__main__":
    main()
