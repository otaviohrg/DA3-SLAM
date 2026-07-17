"""
Keyframe-density × submap-size grid study for DA3-SLAM.

Sweeps the two axes that govern the speed/ATE trade-off jointly, because they
interact through the submap-boundary scale chaining:

  * keyframe stride s   — one keyframe every s input frames (fixed-stride
                          segment mode: segment_strides=(s, s), so density is
                          deterministic and independent of scene motion).
                          Total DA3 cost is ~linear in 1/s.
  * submap_size n       — keyframes per DA3 batch.  Larger n means fewer
                          submap boundaries (less compounding scale drift) but
                          a longer per-batch inference latency.

The physical extent covered by one DA3 batch is s·(n−1) input frames, so the
grid disentangles whether ATE tracks keyframe count, boundary count, or
per-submap coverage.  Each cell records Sim3/SE3 ATE, RPE, wall time, fps and
per-submap inference latency, averaged across the given sequences.

The DA3 model is loaded once and reused across every cell
(SharedSLAM.reconfigure swaps the cheap components only).  Results are
checkpointed to JSON after every cell (--resume skips completed cells) and
rendered as CSV + heatmaps + a speed-vs-ATE Pareto chart (grid.png).

Sequences may be UAS ROS-bag dirs (sensors_only.bag + .tum ground truth;
frames are extracted and fisheye-undistorted once, then cached) or TUM RGB-D
dirs (rgb.txt + groundtruth.txt) — the layout is auto-detected per directory.

Usage (fast-motion UAS sequences, inside the container):

    python scripts/kf_submap_grid.py \\
        --seq_dir data/UAS/fyllingsdalen_tunnel data/UAS/runehamar_tunnel/hornbill \\
        --out_dir outputs/kf_submap_grid \\
        --no_loop_closure

--no_loop_closure is recommended for the factorial study: it isolates the
keyframe/submap effect (loop closures fire more often when there are more
submaps, confounding the two axes) and skips reloading SALAD per cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import traceback
from pathlib import Path

import numpy as np

import benchmark_common as bc
import uas_common as uas
from da3_runner import run_da3
from da3_slam.config import load_slam_config
from tum_eval_common import SharedSLAM


def _mean(xs) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _std(xs) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _seq_mean(per_seq: dict[str, dict], key: str) -> float:
    """Mean of one metric across a cell's per-sequence records."""
    return _mean([m[key] for m in per_seq.values()])


def _seq_std(per_seq: dict[str, dict], key: str) -> float:
    """Sample std of one metric across a cell's per-sequence records."""
    return _std([m[key] for m in per_seq.values()])


# ── sequence loading (layout auto-detected, cached across cells) ──────────────

def load_sequence(
    seq_dir: Path, cache_root: Path, args,
) -> tuple[list[str], list[float], list[tuple[float, np.ndarray]]]:
    """Return (image_paths, timestamps_s, gt_poses) for a TUM or UAS sequence.

    UAS bags are extracted/undistorted once into cache_root/<seq>/ and reused
    by every grid cell (extract_bag_frames + undistort_frames are idempotent).
    """
    if (seq_dir / "rgb.txt").exists():  # TUM RGB-D layout
        paths, ts = bc.load_rgb_list(seq_dir, args.max_frames)
        gt = bc.load_groundtruth(seq_dir / "groundtruth.txt")
        return paths, ts, gt

    bag = uas.find_bag(seq_dir)
    if bag is None:
        raise FileNotFoundError(f"{seq_dir}: no rgb.txt and no *.bag — "
                                f"not a TUM or UAS sequence")
    gt_txt = uas.find_gt(seq_dir)
    if gt_txt is None:
        raise FileNotFoundError(f"{seq_dir}: no ground-truth .tum file")

    topic, calib_path = uas.resolve_camera(
        seq_dir, args.topic, args.calib, args.calib_dir)
    calibration = uas.load_calibration(calib_path)
    cache = cache_root / seq_dir.name
    paths, ts = uas.extract_bag_frames(
        bag, topic, cache / "frames", max_frames=args.max_frames)
    if args.undistort:
        paths = uas.undistort_frames(paths, calibration, cache / "undistorted")
    gt = bc.load_groundtruth(gt_txt)
    return paths, ts, gt


# ── live viewer (optional) ────────────────────────────────────────────────────

def add_viewer_cli(ap) -> None:
    """Add the live-Rerun-viewer flags (mirrors run_slam.py / run_realsense.py)."""
    ap.add_argument("--viewer", choices=["connect", "serve", "spawn", "none"],
                    default="none",
                    help="Live Rerun view of every run (one recording per "
                         "cell × sequence). In Docker the container needs host "
                         "networking to reach the host viewer — run via the "
                         "viz service, e.g. `make scale-sweep SERVICE=viz ...`")
    ap.add_argument("--viewer_addr",
                    default="rerun+http://127.0.0.1:9876/proxy",
                    help="Address of the host Rerun viewer (mode=connect)")
    ap.add_argument("--viewer_max_points", type=int, default=60_000,
                    help="Max points logged per submap (subsampled for speed)")


def make_viewer(args, app_id: str):
    """Build a LiveViewer for one run, or None when the viewer is disabled.

    A fresh viewer (= a fresh Rerun recording, keyed by app_id) is created per
    cell × sequence so runs appear side by side in the viewer instead of
    overlaying into one cloud.
    """
    if getattr(args, "viewer", "none") == "none":
        return None
    from live_viewer import LiveViewer
    return LiveViewer(
        mode=args.viewer,
        addr=args.viewer_addr,
        max_points_per_submap=args.viewer_max_points,
        app_id=app_id,
    )


# ── per-cell configuration ─────────────────────────────────────────────────────

def build_config(stride: int, submap_size: int, args):
    """Base config + fixed-stride keyframe density + submap size for one cell."""
    config = load_slam_config(
        submap_size=submap_size,
        confidence_percentile=args.confidence_percentile,
    )
    keyframe = config.keyframe
    keyframe.selection_mode = "segment"
    # Equal strides = deterministic density (1 keyframe per `stride` frames)
    # and the selector's fixed-stride fast path (no optical flow).
    keyframe.segment_strides = (int(stride), int(stride))
    if args.segment_length is not None:
        keyframe.segment_length = args.segment_length
    if args.no_loop_closure:
        config.enable_loop_closure = False
    # Point clouds are only consumed by a live viewer here (no PLY export);
    # lean frames cut resident memory ~3x — long LC-enabled runs OOM without.
    config.build_pointclouds = getattr(args, "viewer", "none") != "none"
    return config


# ── scoring ────────────────────────────────────────────────────────────────────

def score_trajectory(
    est_ts_to_pose: dict[float, np.ndarray],
    gt_all: list[tuple[float, np.ndarray]],
    max_diff: float,
) -> dict | None:
    """Associate estimate ↔ GT and return compact ATE/RPE metrics."""
    gt_stamps = [e[0] for e in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())
    pairs = bc.associate(est_stamps, gt_stamps, max_diff=max_diff)
    if len(pairs) < 3:
        return None
    gt_matched = [gt_all[ib][1] for _, ib in pairs]
    est_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

    ate_se3 = bc.compute_ate(gt_matched, est_matched, align="se3")
    ate_sim3 = bc.compute_ate(gt_matched, est_matched, align="sim3")
    rpe = bc.compute_rpe(gt_matched, est_matched, delta=1)
    return {
        "ate_sim3": ate_sim3["rmse"],
        "ate_se3": ate_se3["rmse"],
        "sim3_scale": ate_sim3["scale"],
        "rpe_trans": rpe.get("trans_rmse", float("nan")),
        "rpe_rot_deg": rpe.get("rot_rmse_deg", float("nan")),
        "n_matched": len(pairs),
    }


# ── grid execution ─────────────────────────────────────────────────────────────

def run_cell(
    slam: SharedSLAM,
    stride: int,
    submap_size: int,
    sequences: dict[str, tuple],
    out_root: Path,
    args,
) -> dict | None:
    """Run one (stride, submap_size) cell over all sequences; return a record."""
    tag = f"s{stride:02d}_n{submap_size:02d}"
    per_seq: dict[str, dict] = {}

    for seq_name, (paths, timestamps, gt_all) in sequences.items():
        try:
            viewer = make_viewer(args, app_id=f"kf-grid-{seq_name}-{tag}")
            est, timings, counts = run_da3(paths, timestamps, slam, args,
                                           on_update=viewer)
        except Exception as exc:
            print(f"    [ERROR] {seq_name} {tag}: {exc}")
            traceback.print_exc()
            continue

        cell_dir = out_root / tag / seq_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        bc.save_tum_trajectory(est, cell_dir / "trajectory_est.txt")

        metrics = score_trajectory(est, gt_all, args.max_diff)
        if metrics is None:
            print(f"    [SKIP] {seq_name} {tag}: too few GT matches")
            continue

        n_submaps = max(counts["n_submaps"], 1)
        per_seq[seq_name] = {
            **metrics,
            "n_keyframes": counts["n_keyframes"],
            "n_submaps": counts["n_submaps"],
            "n_loop_closures": counts["n_loop_closures"],
            "wall_s": timings["total_s"],
            "fps": timings["fps"],
            # Per-batch DA3 inference time = the real-time lag a live consumer
            # sees between submap updates (grows with submap_size).
            "submap_latency_s": timings.get("submap_building", 0.0) / n_submaps,
        }

    if not per_seq:
        return None

    return {
        "stride": int(stride),
        "submap_size": int(submap_size),
        # Input frames spanned by one DA3 batch (anchor to anchor).
        "frames_per_submap": int(stride) * (int(submap_size) - 1),
        "n_sequences": len(per_seq),
        "n_keyframes_mean": _seq_mean(per_seq, "n_keyframes"),
        "n_submaps_mean": _seq_mean(per_seq, "n_submaps"),
        "n_loop_closures_mean": _seq_mean(per_seq, "n_loop_closures"),
        "ate_sim3_mean": _seq_mean(per_seq, "ate_sim3"),
        "ate_sim3_std": _seq_std(per_seq, "ate_sim3"),
        "ate_se3_mean": _seq_mean(per_seq, "ate_se3"),
        "rpe_trans_mean": _seq_mean(per_seq, "rpe_trans"),
        "rpe_rot_deg_mean": _seq_mean(per_seq, "rpe_rot_deg"),
        "wall_s_mean": _seq_mean(per_seq, "wall_s"),
        "fps_mean": _seq_mean(per_seq, "fps"),
        "submap_latency_s_mean": _seq_mean(per_seq, "submap_latency_s"),
        "per_seq": per_seq,
    }


def run_grid(args) -> list[dict]:
    """Execute the full grid: load sequences once, then run every
    (stride, submap_size) cell, checkpointing grid.json after each so a
    crashed/interrupted study can --resume."""
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading sequences (bag extraction is cached across cells)...")
    sequences: dict[str, tuple] = {}
    for seq in args.seq_dir:
        seq_dir = Path(seq)
        paths, ts, gt = load_sequence(seq_dir, out_root / "_frames", args)
        sequences[seq_dir.name] = (paths, ts, gt)
        print(f"  {seq_dir.name}: {len(paths)} frames, {len(gt)} GT poses")

    cells = [(s, n) for s in args.strides for n in args.submap_sizes]
    print(f"\nGrid plan: {len(args.strides)} stride(s) × "
          f"{len(args.submap_sizes)} submap size(s) × "
          f"{len(sequences)} sequence(s) = "
          f"{len(cells) * len(sequences)} SLAM runs")

    # ── resume: reload completed cells and skip them ──────────────────────────
    records: list[dict] = []
    done: set[tuple[int, int]] = set()
    ckpt = out_root / "grid.json"
    if args.resume and ckpt.exists():
        with open(ckpt) as f:
            records = json.load(f)
        done = {(r["stride"], r["submap_size"]) for r in records}
        print(f"Resuming: {len(done)} cell(s) already done in {ckpt}")

    print("Loading DA3 model (loaded once, reused across all cells)...")
    slam = SharedSLAM(build_config(cells[0][0], cells[0][1], args))

    for stride, submap_size in cells:
        if (stride, submap_size) in done:
            print(f"  [skip] stride={stride} submap_size={submap_size}: done")
            continue
        print(f"\n{'═'*60}\n  Cell: stride={stride}  submap_size={submap_size}  "
              f"(~{stride*(submap_size-1)} frames per submap)\n{'═'*60}")
        slam.reconfigure(build_config(stride, submap_size, args))

        record = run_cell(slam, stride, submap_size, sequences, out_root, args)
        if record is None:
            print(f"  [SKIP] stride={stride} submap_size={submap_size}: "
                  f"no successful sequences")
            continue
        records.append(record)
        done.add((stride, submap_size))
        save_data(records, out_root, quiet=True)  # checkpoint (crash-safe)
        print(f"  stride={stride:<3} submap_size={submap_size:<3} "
              f"Sim3 ATE={record['ate_sim3_mean']:.4f}±{record['ate_sim3_std']:.4f} m  "
              f"fps={record['fps_mean']:.1f}  "
              f"KFs={record['n_keyframes_mean']:.0f}  "
              f"submaps={record['n_submaps_mean']:.0f}  "
              f"batch latency={record['submap_latency_s_mean']:.2f}s")
    return records


# ── outputs ────────────────────────────────────────────────────────────────────

def save_data(records: list[dict], out_dir: Path, quiet: bool = False) -> None:
    """Write grid.json (full records) and grid.csv (scalar columns only,
    sorted by cell) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "grid.json", "w") as f:
        json.dump(records, f, indent=2)
    if records:
        scalar_keys = [k for k, v in records[0].items()
                       if not isinstance(v, (list, dict))]
        with open(out_dir / "grid.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(records,
                               key=lambda r: (r["stride"], r["submap_size"])))
    if not quiet:
        print(f"\nData saved → {out_dir}/grid.{{json,csv}}")


def _metric_grid(
    records: list[dict], strides: list[int], sizes: list[int], key: str,
) -> np.ndarray:
    """(len(sizes), len(strides)) array of one metric; NaN for missing cells."""
    grid = np.full((len(sizes), len(strides)), np.nan)
    for r in records:
        grid[sizes.index(r["submap_size"]), strides.index(r["stride"])] = r[key]
    return grid


def _draw_heatmap(
    fig, ax, grid: np.ndarray, strides: list[int], sizes: list[int],
    label: str, fmt: str, cmap: str,
) -> None:
    """One annotated stride × submap-size heatmap panel."""
    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(strides)), [str(s) for s in strides])
    ax.set_yticks(range(len(sizes)), [str(n) for n in sizes])
    ax.set_xlabel("keyframe stride (frames per keyframe)")
    ax.set_ylabel("submap size (keyframes per batch)")
    ax.set_title(label)
    finite = grid[np.isfinite(grid)]
    mid = (finite.min() + finite.max()) / 2 if len(finite) else 0
    for i in range(len(sizes)):
        for j in range(len(strides)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                        fontsize=9,
                        color="white" if grid[i, j] < mid else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)


def make_chart(records: list[dict], out_path: Path, scene: str = "") -> None:
    """2×2 figure: ATE / fps / batch-latency heatmaps + speed-ATE Pareto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strides = sorted({r["stride"] for r in records})
    sizes = sorted({r["submap_size"] for r in records})

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    title = "Keyframe stride × submap size"
    if scene:
        title += f" — {scene}"
    fig.suptitle(title, fontsize=14)

    panels = [
        (axes[0][0], "ate_sim3_mean", "Sim3 ATE (m) — lower is better",
         "{:.3f}", "viridis_r"),
        (axes[0][1], "fps_mean", "throughput (fps) — higher is better",
         "{:.1f}", "viridis"),
        (axes[1][0], "submap_latency_s_mean",
         "per-submap DA3 latency (s) — live update lag", "{:.2f}", "viridis_r"),
    ]
    for ax, key, label, fmt, cmap in panels:
        grid = _metric_grid(records, strides, sizes, key)
        _draw_heatmap(fig, ax, grid, strides, sizes, label, fmt, cmap)

    ax = axes[1][1]
    cmap = plt.get_cmap("viridis")
    denom = max(len(sizes) - 1, 1)
    for r in sorted(records, key=lambda r: (r["stride"], r["submap_size"])):
        color = cmap(sizes.index(r["submap_size"]) / denom)
        ax.scatter(r["fps_mean"], r["ate_sim3_mean"], s=60, color=color,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(f"s{r['stride']}/n{r['submap_size']}",
                    (r["fps_mean"], r["ate_sim3_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("throughput (fps)")
    ax.set_ylabel("Sim3 ATE (m)")
    ax.set_title("speed–accuracy Pareto (lower-right is better)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _ints(s: str) -> list[int]:
    """argparse type: comma-separated ints, e.g. "4,8,16"."""
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Keyframe-density × submap-size grid study for DA3-SLAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seq_dir", nargs="+", required=True,
                    help="Sequence dir(s): UAS (sensors_only.bag + .tum) or "
                         "TUM (rgb.txt + groundtruth.txt); auto-detected")
    ap.add_argument("--out_dir", default="outputs/kf_submap_grid",
                    help="Output directory for runs, data, and chart")
    ap.add_argument("--strides", type=_ints, default=[4, 8, 16],
                    help="Comma-separated keyframe strides (1 = every frame)")
    ap.add_argument("--submap_sizes", type=_ints, default=[8, 16, 32],
                    help="Comma-separated submap sizes (keyframes per DA3 batch)")
    ap.add_argument("--segment_length", type=int, default=None,
                    help="Segment length override (frames buffered per segment)")
    ap.add_argument("--confidence_percentile", type=float, default=None)
    ap.add_argument("--no_loop_closure", action="store_true",
                    help="Disable loop closure (recommended: isolates the "
                         "keyframe/submap effect)")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Cap frames per sequence (for quick tests)")
    ap.add_argument("--max_diff", type=float, default=0.02,
                    help="Estimate↔GT association tolerance in seconds")
    # UAS-specific I/O (ignored for TUM-layout sequences)
    ap.add_argument("--topic", default=None,
                    help="UAS camera topic override (else inferred per sequence)")
    ap.add_argument("--calib", default=None,
                    help="UAS calibration YAML override (else inferred)")
    ap.add_argument("--calib_dir", type=Path, default=None,
                    help="UAS calibration/ folder (else auto-discovered)")
    ap.add_argument("--no_undistort", dest="undistort", action="store_false",
                    help="Skip fisheye undistortion (UAS sequences)")
    add_viewer_cli(ap)
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already in <out_dir>/grid.json")
    ap.add_argument("--chart_only", type=Path, default=None,
                    help="Skip running; render the chart from an existing "
                         "grid.json at this path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for stride in args.strides:
        if stride < 1:
            raise SystemExit(f"stride must be >= 1, got {stride}")
    for size in args.submap_sizes:
        if size < 2:
            raise SystemExit(f"submap_size must be >= 2, got {size}")

    out_dir = Path(args.out_dir)
    scene = (Path(args.seq_dir[0]).name if len(args.seq_dir) == 1
             else f"{len(args.seq_dir)} sequences")

    if args.chart_only is not None:
        with open(args.chart_only) as f:
            records = json.load(f)
        make_chart(records, out_dir / "grid.png", scene)
        return

    records = run_grid(args)
    if not records:
        print("No results produced; nothing to save or plot.")
        return
    save_data(records, out_dir)
    make_chart(records, out_dir / "grid.png", scene)


if __name__ == "__main__":
    main()
