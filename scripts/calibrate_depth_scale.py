"""
Calibrate DA3 depth scale against TUM RGB-D sensor depth.

For each frame, computes per-pixel ratio:
    ratio = DA3_depth / sensor_depth

at high-confidence, sensor-valid pixels. If ratios cluster tightly around
a constant across frames and sequences, a global scale correction factor
can be applied to DA3 outputs to eliminate systematic scale drift.

Interpretation:
    ratio > 1  →  DA3 overestimates depth  →  correction = 1 / ratio (< 1)
    ratio < 1  →  DA3 underestimates depth →  correction = 1 / ratio (> 1)
    CV < 0.05  →  bias is systematic, global correction is reliable
    CV > 0.15  →  bias is scene-dependent, per-boundary correction needed

Usage:
    # Single sequence
    python scripts/calibrate_depth_scale.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_desk

    # All sequences
    python scripts/calibrate_depth_scale.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_* \\
        --max_frames 50

    # Save plots
    python scripts/calibrate_depth_scale.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_* \\
        --max_frames 50 --plot calibration.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ── TUM helpers ───────────────────────────────────────────────────────────────

def parse_tum_file(path: Path) -> list[tuple[float, str]]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return sorted(entries)


def associate(
    entries_a: list[tuple[float, str]],
    entries_b: list[tuple[float, str]],
    max_diff: float = 0.02,
) -> list[tuple[int, int]]:
    """Associate two timestamp lists by nearest match (max_diff seconds)."""
    stamps_b = np.array([t for t, _ in entries_b])
    pairs = []
    used: set[int] = set()
    for ia, (ta, _) in enumerate(entries_a):
        ib = int(np.argmin(np.abs(stamps_b - ta)))
        if abs(stamps_b[ib] - ta) <= max_diff and ib not in used:
            pairs.append((ia, ib))
            used.add(ib)
    return pairs


def load_sensor_depth(path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    """
    Load a TUM 16-bit depth PNG and rescale to metres.
    Resizes to target_shape (H, W) using nearest-neighbour to avoid
    interpolating depth values across object boundaries.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if raw is None:
        return None
    depth = raw.astype(np.float32) / 5000.0  # TUM convention: value / 5000 = metres
    if depth.shape != target_shape:
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return depth


# ── per-frame ratio computation ───────────────────────────────────────────────

def depth_scale_ratios(
    da3_depth: np.ndarray,
    sensor_depth: np.ndarray,
    da3_confidence: np.ndarray,
    confidence_percentile: float,
) -> np.ndarray | None:
    """
    Return per-pixel DA3/sensor depth ratios at valid, high-confidence pixels.

    Valid means:
      - sensor depth > 0.1 m  (sensor returns 0 for missing measurements)
      - sensor depth < 10.0 m (TUM fr1 range limit)
      - DA3 depth   > 0.0 m
      - DA3 confidence in top (100 - confidence_percentile)% of the frame
    """
    confidence_threshold = float(np.percentile(da3_confidence, confidence_percentile))
    high_confidence = da3_confidence >= confidence_threshold
    valid_sensor    = (sensor_depth > 0.1) & (sensor_depth < 10.0)
    valid_da3       = da3_depth > 0.0
    mask = high_confidence & valid_sensor & valid_da3

    if mask.sum() < 50:
        return None

    return (da3_depth[mask] / sensor_depth[mask]).astype(np.float32)


# ── sequence calibration ──────────────────────────────────────────────────────

def calibrate_sequence(
    seq_dir: Path,
    estimator,
    max_frames: int | None,
    confidence_percentile: float,
) -> dict | None:
    rgb_txt   = seq_dir / "rgb.txt"
    depth_txt = seq_dir / "depth.txt"

    if not rgb_txt.exists() or not depth_txt.exists():
        print(f"  [SKIP] rgb.txt or depth.txt missing in {seq_dir.name}")
        return None

    rgb_entries   = parse_tum_file(rgb_txt)
    depth_entries = parse_tum_file(depth_txt)
    pairs = associate(rgb_entries, depth_entries)

    if not pairs:
        print(f"  [SKIP] No matching RGB–depth pairs in {seq_dir.name}")
        return None

    # Subsample evenly across the sequence so we cover it uniformly
    if max_frames and len(pairs) > max_frames:
        indices = np.linspace(0, len(pairs) - 1, max_frames, dtype=int)
        pairs = [pairs[i] for i in indices]

    print(f"  {seq_dir.name}: {len(pairs)} frames")

    all_ratios: list[np.ndarray] = []
    per_frame_medians: list[float] = []

    for ia, ib in pairs:
        rgb_path   = seq_dir / rgb_entries[ia][1]
        depth_path = seq_dir / depth_entries[ib][1]

        try:
            prediction = estimator.infer([str(rgb_path)])
        except Exception as exc:
            print(f"    [warn] DA3 failed on {rgb_path.name}: {exc}")
            continue

        da3_depth      = prediction.depth[0]       # (H, W) float32, metres
        da3_confidence = prediction.confidence[0]  # (H, W) float32, [0, 1]
        H, W           = da3_depth.shape

        sensor_depth = load_sensor_depth(depth_path, (H, W))
        if sensor_depth is None:
            print(f"    [warn] Could not load sensor depth: {depth_path.name}")
            continue

        ratios = depth_scale_ratios(da3_depth, sensor_depth,
                                    da3_confidence, confidence_percentile)
        if ratios is None:
            continue

        all_ratios.append(ratios)
        per_frame_medians.append(float(np.median(ratios)))

    if not all_ratios:
        print(f"  [SKIP] No valid frames produced for {seq_dir.name}")
        return None

    flat = np.concatenate(all_ratios)
    medians = np.array(per_frame_medians)

    return {
        "seq":               seq_dir.name,
        "n_frames":          len(medians),
        "ratios":            flat,
        "per_frame_medians": medians,
        "mean":              float(np.mean(flat)),
        "median":            float(np.median(flat)),
        "std":               float(np.std(flat)),
        # Coefficient of variation of per-frame medians —
        # low CV means the bias is stable across frames → global correction works
        "cv":                float(np.std(medians) / np.mean(medians)),
    }


# ── reporting ─────────────────────────────────────────────────────────────────

_TUM_PREFIX = "rgbd_dataset_freiburg1_"


def _strip_tum_prefix(name: str) -> str:
    return name[len(_TUM_PREFIX):] if name.startswith(_TUM_PREFIX) else name


def print_summary(results: list[dict], overall_median: float) -> None:
    correction = 1.0 / overall_median

    seq_col = max(len(r["seq"]) for r in results)

    header = (f"{'Sequence':<{seq_col}}  "
              f"{'Median ratio':>13}  {'Std':>8}  {'CV':>6}  {'Frames':>6}")
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        print(f"  {_strip_tum_prefix(r['seq']):<{seq_col-2}}  "
              f"{r['median']:>13.4f}  "
              f"{r['std']:>8.4f}  "
              f"{r['cv']:>6.3f}  "
              f"{r['n_frames']:>6}")

    print(sep)
    overall_cv = float(np.std([r["median"] for r in results])
                       / np.mean([r["median"] for r in results]))
    print(f"  {'overall':<{seq_col-2}}  {overall_median:>13.4f}  "
          f"{'':>8}  {overall_cv:>6.3f}")
    print(sep)

    print(f"\n  DA3/sensor median ratio : {overall_median:.4f}")
    print(f"  Recommended correction  : × {correction:.4f}  "
          f"({'scale DA3 depth DOWN' if correction < 1 else 'scale DA3 depth UP'})")
    print(f"  Cross-sequence CV       : {overall_cv:.3f}")

    if overall_cv < 0.05:
        print("\n  → Bias is highly systematic. A global correction factor is reliable.")
    elif overall_cv < 0.15:
        print("\n  → Bias is moderately systematic. Global correction will help "
              "but some scene-dependent residual remains.")
    else:
        print("\n  → Bias varies significantly across scenes. "
              "Per-boundary scale estimation (strategy 2) is needed.")


def plot_results(results: list[dict], overall_median: float, save_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    prefix = "rgbd_dataset_freiburg1_"
    names  = [r["seq"][len(prefix):] if r["seq"].startswith(prefix) else r["seq"]
              for r in results]

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)

    # ── left: per-frame median over time, one line per sequence ───────────────
    ax_left = fig.add_subplot(gs[0])
    for r, name in zip(results, names):
        ax_left.plot(r["per_frame_medians"], label=name, linewidth=1.2)
    ax_left.axhline(overall_median, color="black", linewidth=1.5,
                    linestyle="--", label=f"overall median ({overall_median:.3f})")
    ax_left.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_left.set_xlabel("Frame index (subsampled)")
    ax_left.set_ylabel("DA3 / sensor depth ratio (per-frame median)")
    ax_left.set_title("Depth scale ratio over time")
    ax_left.legend(fontsize=8, ncol=2)
    ax_left.grid(True, alpha=0.3)

    # ── right: box plot per sequence ──────────────────────────────────────────
    ax_right = fig.add_subplot(gs[1])
    data = [r["per_frame_medians"] for r in results]
    bp   = ax_right.boxplot(data, vert=True, patch_artist=True,
                             medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec6e8")
    ax_right.axhline(overall_median, color="black", linewidth=1.5,
                     linestyle="--", label=f"overall ({overall_median:.3f})")
    ax_right.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_right.set_xticks(range(1, len(names) + 1))
    ax_right.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax_right.set_ylabel("DA3 / sensor depth ratio")
    ax_right.set_title("Per-sequence spread")
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3, axis="y")

    plt.suptitle("DA3 depth scale calibration vs TUM sensor depth", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate DA3 depth scale against TUM sensor depth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="TUM sequence directory/directories")
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Frames to sample per sequence (evenly spaced)")
    parser.add_argument("--confidence_percentile", type=float, default=50.0,
                        help="DA3 confidence percentile threshold for valid pixels")
    parser.add_argument("--plot", default=None, metavar="PATH",
                        help="Save plots to this path (e.g. calibration.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from da3_slam.backend.inference.depth_estimator import DepthEstimator
    from da3_slam.config import load_slam_config

    config    = load_slam_config()
    estimator = DepthEstimator(
        model_id=config.depth_model,
        process_resolution=config.depth_model_resolution,
    )

    results = []
    for seq_path in args.seq_dir:
        seq_dir = Path(seq_path)
        print(f"\n[{seq_dir.name}]")
        result = calibrate_sequence(
            seq_dir, estimator, args.max_frames, args.confidence_percentile
        )
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo valid sequences processed.")
        sys.exit(1)

    all_medians    = np.concatenate([r["per_frame_medians"] for r in results])
    overall_median = float(np.median(all_medians))

    print_summary(results, overall_median)

    if args.plot:
        try:
            plot_results(results, overall_median, args.plot)
        except ImportError:
            print("  [warn] matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
