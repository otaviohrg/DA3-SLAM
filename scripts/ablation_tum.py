"""
Ablation study for DA3-SLAM on TUM RGB-D sequences.

Sweeps key hyperparameters one-at-a-time (OFAT) around a baseline and
reports average ATE across all sequences in a ranked comparison table.

Parameters swept:
    submap_size              — keyframes per submap (including 1-frame anchor)
    min_disparity_fraction   — keyframe selection aggressiveness (optical flow)
    confidence_percentile    — point cloud filtering threshold
    loop_threshold           — DINOv2 cosine similarity for loop closure

Usage:
    python scripts/ablation_tum.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_xyz \\
                  data/tum/rgbd_dataset_freiburg1_desk \\
                  data/tum/rgbd_dataset_freiburg1_room \\
        --out_dir outputs/ablation

    # Limit frames for a quick sanity pass (not representative):
    python scripts/ablation_tum.py --seq_dir ... --max_frames 300

    # Skip already-computed configs (reads ablation_results.json):
    python scripts/ablation_tum.py --seq_dir ... --resume

Outputs:
    <out_dir>/ablation_results.json    all per-sequence metrics per config
    <out_dir>/<config_name>/           per-sequence trajectory + plots
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# ── sweep specification (round 2) ────────────────────────────────────────────
#
# Round 1 findings:
#   - submap_size=20 is clearly best; updated as new YAML default.
#   - confidence_percentile=65 is a stable local optimum; not swept again.
#   - min_disparity_fraction was confounded at sub=8 (max_submap_size cap was
#     always the binding constraint → all disp values gave identical KF counts).
#     Properly tested here at sub=20.
#   - Loop closure produced 0 LCs at sub=20 with thr≥0.85 (too few submaps to
#     find candidates). Threshold lowered to exercise loop closure.
#
# Baseline: current config/default.yaml values.
BASELINE: dict[str, Any] = {
    "submap_size":            20,
    "min_disparity_fraction": 0.10,
    "confidence_percentile":  65.0,
    "loop_threshold":         0.90,
}

# Each entry: (param_key, display_short, values_to_sweep)
# The baseline value must appear somewhere in each sweep so the baseline row
# is included once and naturally compared to every variant.
#
# Parameter guidance (round 2):
#   submap_size:            Fine sweep around the round-1 optimum (20).
#                           DA3 joint estimation improves with more frames per
#                           batch, but very large values leave too few submaps
#                           for the pose graph to correct drift.
#   min_disparity_fraction: At sub=20, max_submap_size=20 is no longer always
#                           the binding constraint. Lower values (≥0.10) select
#                           more keyframes → more context per submap.  Higher
#                           values select fewer, sparser keyframes.
#   loop_threshold:         At sub=20 the graph has 2–4 nodes per sequence;
#                           loop closure requires lower similarity thresholds to
#                           fire at all. Sweep 0.70–0.90 to find whether any
#                           loop closures actually help at this submap size.
#   confidence_percentile:  Not swept — well characterised in round 1 (65 optimal).
SWEEPS: list[tuple[str, str, list]] = [
    ("submap_size",            "sub",  [14, 16, 18, 20, 22, 25]),
    ("min_disparity_fraction", "disp", [0.10, 0.20, 0.30, 0.40, 0.50]),
    ("loop_threshold",         "thr",  [0.70, 0.75, 0.80, 0.85, 0.90]),
]

# Specific multi-parameter combos (optional — set to [] to skip)
COMBO_CONFIGS: list[dict[str, Any]] = [
    # Example: pair best submap_size with best disparity found from OFAT sweeps
    # Fill in after a first OFAT pass.
]


# ── config generation ─────────────────────────────────────────────────────────

def _config_label(params: dict[str, Any]) -> str:
    """Short human-readable label for a parameter dict."""
    parts = []
    for key, val in params.items():
        short = {
            "submap_size":            "sub",
            "min_disparity_fraction": "disp",
            "confidence_percentile":  "conf",
            "loop_threshold":         "thr",
        }.get(key, key)
        parts.append(f"{short}={val}")
    return "  ".join(parts)


def generate_configs() -> list[dict[str, Any]]:
    """
    Generate OFAT configs: for each sweep axis, vary the target parameter
    while holding all others at their baseline value.

    The baseline itself appears exactly once (deduplicated by param dict).
    Additional combo configs are appended at the end.
    """
    seen: list[dict] = []

    # OFAT: vary one param at a time
    for sweep_key, _, values in SWEEPS:
        for val in values:
            cfg = dict(BASELINE)
            cfg[sweep_key] = val
            if cfg not in seen:
                seen.append(cfg)

    # Explicit combos
    for extra in COMBO_CONFIGS:
        cfg = dict(BASELINE)
        cfg.update(extra)
        if cfg not in seen:
            seen.append(cfg)

    return seen


# ── benchmark helpers (mirrored from benchmark_tum.py) ───────────────────────

def _import_benchmark():
    """Import utility functions from benchmark_tum.py (same directory)."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import benchmark_tum as bm
    return bm


# ── per-config runner ─────────────────────────────────────────────────────────

class SharedSLAM:
    """
    Wraps DA3SLAM so the expensive DepthEstimator (GPU model) is loaded
    once and reused across all configs.  Only the cheap components
    (SubmapBuilder, LoopClosureDetector) are rebuilt per config.
    """

    def __init__(self, base_config):
        from da3_slam.slam import DA3SLAM
        # Load model once
        self._slam = DA3SLAM(base_config)

    def reconfigure(self, config) -> None:
        """Swap in new config without reloading the depth model."""
        from da3_slam.backend.inference.submap import SubmapBuilder
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector

        self._slam.config  = config
        self._slam.builder = SubmapBuilder(
            self._slam.estimator,
            confidence_percentile=config.confidence_percentile,
        )
        self._slam.detector = (
            LoopClosureDetector(config.loop_closure)
            if config.enable_loop_closure else None
        )

    def run(self, image_paths: list[str]):
        # Reset detector before each sequence so embeddings from a previous
        # sequence don't produce cross-sequence loop closures with stale indices.
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector
        cfg = self._slam.config
        if cfg.enable_loop_closure:
            self._slam.detector = LoopClosureDetector(cfg.loop_closure)
        return self._slam.run(image_paths)


def run_config(
    params: dict[str, Any],
    seq_dirs: list[Path],
    out_root: Path,
    max_frames: int | None,
    shared_slam: SharedSLAM,
    bm,
) -> dict:
    """
    Run all sequences for one parameter configuration.
    Returns a summary dict with per-sequence metrics and averages.
    """
    from da3_slam.config import load_slam_config

    label = _config_label(params)
    cfg_name = label.replace(" ", "").replace("=", "").replace(".", "p")

    config = load_slam_config(
        submap_size=params["submap_size"],
        confidence_percentile=params["confidence_percentile"],
    )
    # Patch sub-level fields not reachable via load_slam_config overrides
    config.keyframe.min_disparity_fraction = params["min_disparity_fraction"]
    config.keyframe.max_submap_size        = params["submap_size"]
    config.loop_closure.similarity_threshold = params["loop_threshold"]

    shared_slam.reconfigure(config)

    seq_metrics: list[dict] = []
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        out_dir  = out_root / cfg_name / seq_name

        rgb_txt = seq_dir / "rgb.txt"
        gt_txt  = seq_dir / "groundtruth.txt"
        if not rgb_txt.exists() or not gt_txt.exists():
            print(f"  [SKIP] {seq_name}: missing rgb.txt or groundtruth.txt")
            continue

        rgb_entries = bm.parse_tum_file(rgb_txt)
        image_paths = [str(seq_dir / e[1]) for e in rgb_entries]
        timestamps  = [e[0] for e in rgb_entries]

        if max_frames:
            image_paths = image_paths[:max_frames]
            timestamps  = timestamps[:max_frames]

        try:
            t0     = time.time()
            result = shared_slam.run(image_paths)
            wall   = time.time() - t0

            ts_map = {i: timestamps[i] for i in range(len(timestamps))}
            out_dir.mkdir(parents=True, exist_ok=True)
            est_tum = str(out_dir / "trajectory_est.txt")
            result.save_tum(est_tum, timestamps=ts_map)

            est_ts_to_pose: dict[float, np.ndarray] = {}
            for seq_idx, pose in result.keyframe_poses.items():
                if seq_idx in ts_map:
                    est_ts_to_pose[ts_map[seq_idx]] = pose

            gt_all     = bm.load_groundtruth(gt_txt)
            gt_stamps  = [e[0] for e in gt_all]
            est_stamps = sorted(est_ts_to_pose.keys())
            pairs      = bm.associate(est_stamps, gt_stamps, max_diff=0.02)

            if len(pairs) < 3:
                print(f"  [SKIP] {seq_name}: too few matched poses")
                continue

            gt_matched  = [gt_all[ib][1]                    for _, ib in pairs]
            est_matched = [est_ts_to_pose[est_stamps[ia]]   for ia, _ in pairs]

            ate_se3  = bm.compute_ate(gt_matched, est_matched, align="se3")
            ate_sim3 = bm.compute_ate(gt_matched, est_matched, align="sim3")
            rpe_1    = bm.compute_rpe(gt_matched, est_matched, delta=1)

            m = {
                "sequence":         seq_name,
                "ate_se3_rmse":     ate_se3["rmse"],
                "ate_sim3_rmse":    ate_sim3["rmse"],
                "rpe_trans_rmse":   rpe_1["trans_rmse"],
                "rpe_rot_rmse_deg": rpe_1["rot_rmse_deg"],
                "n_frames":         len(image_paths),
                "n_keyframes":      result.n_keyframes,
                "n_submaps":        len(result.submaps),
                "n_loop_closures":  len(result.loop_closures),
                "wall_seconds":     round(wall, 1),
            }
            seq_metrics.append(m)

            print(f"  {seq_name:<30}  ATE={ate_se3['rmse']:.4f}m  "
                  f"Sim3={ate_sim3['rmse']:.4f}m  "
                  f"KFs={result.n_keyframes}  LCs={len(result.loop_closures)}")

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {seq_name}: {exc}")
            traceback.print_exc()

    if not seq_metrics:
        return {"params": params, "label": label, "sequences": [], "avg": {}}

    avg_ate      = float(np.mean([m["ate_se3_rmse"]     for m in seq_metrics]))
    avg_ate_sim3 = float(np.mean([m["ate_sim3_rmse"]    for m in seq_metrics]))
    avg_rpe_t    = float(np.mean([m["rpe_trans_rmse"]   for m in seq_metrics]))
    avg_rpe_r    = float(np.mean([m["rpe_rot_rmse_deg"] for m in seq_metrics]))
    avg_lcs      = float(np.mean([m["n_loop_closures"]  for m in seq_metrics]))
    avg_kfs      = float(np.mean([m["n_keyframes"]      for m in seq_metrics]))
    avg_wall     = float(np.mean([m["wall_seconds"]     for m in seq_metrics]))
    # Normalized timing: compute per-sequence ratios then average (avoids
    # longer sequences dominating the mean).
    avg_s_per_frame = float(np.mean(
        [m["wall_seconds"] / m["n_frames"] for m in seq_metrics]
    ))
    avg_s_per_kf = float(np.mean(
        [m["wall_seconds"] / m["n_keyframes"] for m in seq_metrics if m["n_keyframes"] > 0]
    ))

    return {
        "params":    params,
        "label":     label,
        "sequences": seq_metrics,
        "avg": {
            "ate_se3_rmse":     avg_ate,
            "ate_sim3_rmse":    avg_ate_sim3,
            "rpe_trans_rmse":   avg_rpe_t,
            "rpe_rot_rmse_deg": avg_rpe_r,
            "n_loop_closures":  avg_lcs,
            "n_keyframes":      avg_kfs,
            "wall_seconds":     avg_wall,
            "s_per_frame":      avg_s_per_frame,
            "s_per_kf":         avg_s_per_kf,
        },
    }


# ── table printing ────────────────────────────────────────────────────────────

def print_table(results: list[dict], baseline_params: dict) -> None:
    """Print ranked ablation table sorted by avg ATE SE3 (ascending)."""

    completed = [r for r in results if r.get("avg")]
    if not completed:
        print("No completed configs to report.")
        return

    completed.sort(key=lambda r: r["avg"]["ate_se3_rmse"])

    # Column widths
    label_col = max(len(r["label"]) for r in completed) + 2

    header = (
        f"{'Configuration':<{label_col}}  "
        f"{'sub':>4}  {'disp':>5}  {'conf':>5}  {'thr':>5}  "
        f"{'avg ATE':>9}  {'avg Sim3':>9}  "
        f"{'avg RPE-t':>10}  {'avg RPE-r':>10}  "
        f"{'avg KFs':>8}  {'avg LCs':>8}  {'avg wall':>9}"
    )
    sep = "═" * (len(header) + 2)

    print(f"\n{sep}")
    print("  ABLATION STUDY  —  ranked by avg ATE RMSE (SE3 alignment)")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))

    for rank, r in enumerate(completed, 1):
        p   = r["params"]
        avg = r["avg"]
        is_baseline = (p == baseline_params)
        marker = " *" if is_baseline else "  "
        label  = r["label"]

        print(
            f"{marker}{label:<{label_col}}  "
            f"{p['submap_size']:>4}  "
            f"{p['min_disparity_fraction']:>5.2f}  "
            f"{p['confidence_percentile']:>5.1f}  "
            f"{p['loop_threshold']:>5.2f}  "
            f"{avg['ate_se3_rmse']:>9.4f}  "
            f"{avg['ate_sim3_rmse']:>9.4f}  "
            f"{avg['rpe_trans_rmse']:>10.4f}  "
            f"{avg['rpe_rot_rmse_deg']:>10.3f}  "
            f"{avg['n_keyframes']:>8.1f}  "
            f"{avg['n_loop_closures']:>8.1f}  "
            f"{avg['wall_seconds']:>9.1f}s"
        )

    print("  " + "─" * len(header))
    best = completed[0]
    print(f"  Best: {best['label']}  →  avg ATE {best['avg']['ate_se3_rmse']:.4f} m")
    print(f"  (* = baseline config)")
    print(sep)


# ── timing table ─────────────────────────────────────────────────────────────

def print_timing_table(results: list[dict], baseline_params: dict) -> None:
    """
    Second ablation table: wall-clock timing ranked by avg s/frame (ascending).

    Normalizing by sequence length (s/frame) removes the effect of different
    sequence durations and reflects per-frame processing cost.
    s/KF shows how much time each selected keyframe costs (inference-heavy).
    """
    completed = [r for r in results if r.get("avg") and "s_per_frame" in r["avg"]]
    if not completed:
        return

    completed.sort(key=lambda r: r["avg"]["s_per_frame"])

    label_col = max(len(r["label"]) for r in completed) + 2

    header = (
        f"{'Configuration':<{label_col}}  "
        f"{'sub':>4}  {'disp':>5}  {'conf':>5}  {'thr':>5}  "
        f"{'avg wall':>9}  {'s/frame':>8}  {'s/KF':>8}  "
        f"{'avg KFs':>8}  {'KF rate':>8}"
    )
    sep = "═" * (len(header) + 2)

    print(f"\n{sep}")
    print("  ABLATION STUDY  —  wall-clock timing (ranked by s/frame, ascending = fastest)")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))

    for r in completed:
        p   = r["params"]
        avg = r["avg"]
        is_baseline = (p == baseline_params)
        marker = " *" if is_baseline else "  "
        # KF rate: fraction of frames selected as keyframes
        kf_rate = avg["n_keyframes"] / (avg["wall_seconds"] / avg["s_per_frame"]) \
                  if avg["s_per_frame"] > 0 else 0.0

        print(
            f"{marker}{r['label']:<{label_col}}  "
            f"{p['submap_size']:>4}  "
            f"{p['min_disparity_fraction']:>5.2f}  "
            f"{p['confidence_percentile']:>5.1f}  "
            f"{p['loop_threshold']:>5.2f}  "
            f"{avg['wall_seconds']:>9.1f}s  "
            f"{avg['s_per_frame']:>7.3f}s  "
            f"{avg['s_per_kf']:>7.2f}s  "
            f"{avg['n_keyframes']:>8.1f}  "
            f"{kf_rate:>7.1%}"
        )

    print("  " + "─" * len(header))
    best = completed[0]
    print(f"  Fastest: {best['label']}  →  {best['avg']['s_per_frame']:.3f} s/frame")
    print(f"  (* = baseline config)")
    print(sep)


# ── per-axis per-sequence breakdown ──────────────────────────────────────────

def print_per_axis_breakdown(results: list[dict], baseline_params: dict) -> None:
    """For each sweep axis, print a mini-table showing how that parameter alone affects results."""
    completed = {tuple(sorted(r["params"].items())): r for r in results if r.get("avg")}

    for sweep_key, short, values in SWEEPS:
        print(f"\n  ── Sweep: {sweep_key} ({'  '.join(str(v) for v in values)}) ──")
        print(f"  {'value':>8}  {'avg ATE':>9}  {'avg Sim3':>9}  {'avg KFs':>8}  {'avg LCs':>8}  {'s/frame':>8}")
        print(f"  {'─'*64}")

        for val in values:
            cfg = dict(BASELINE)
            cfg[sweep_key] = val
            key = tuple(sorted(cfg.items()))
            if key not in completed:
                print(f"  {val:>8}  {'—':>9}  {'—':>9}  {'—':>8}  {'—':>8}  {'—':>8}")
                continue
            avg = completed[key]["avg"]
            marker = " *" if cfg == baseline_params else "  "
            s_per_frame = avg.get("s_per_frame", float("nan"))
            print(
                f"{marker}{val:>8}  "
                f"{avg['ate_se3_rmse']:>9.4f}  "
                f"{avg['ate_sim3_rmse']:>9.4f}  "
                f"{avg['n_keyframes']:>8.1f}  "
                f"{avg['n_loop_closures']:>8.1f}  "
                f"{s_per_frame:>7.3f}s"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for DA3-SLAM on TUM RGB-D sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="TUM sequence directories to benchmark")
    parser.add_argument("--out_dir", default="outputs/ablation",
                        help="Root output directory")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap frames per sequence for quick tests")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs already present in ablation_results.json")
    parser.add_argument("--table_only", action="store_true",
                        help="Only print table from saved ablation_results.json, no new runs")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    bm     = _import_benchmark()
    configs = generate_configs()

    out_root     = Path(args.out_dir)
    results_path = out_root / "ablation_results.json"
    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = [Path(p) for p in args.seq_dir]

    # ── load previous results ─────────────────────────────────────────────────
    all_results: list[dict] = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing result(s) from {results_path}")

    if args.table_only:
        print_table(all_results, BASELINE)
        print_timing_table(all_results, BASELINE)
        print_per_axis_breakdown(all_results, BASELINE)
        return

    # ── identify configs to run ───────────────────────────────────────────────
    completed_params = [r["params"] for r in all_results]
    pending = [c for c in configs if c not in completed_params] \
              if args.resume else configs

    if not pending:
        print("All configs already completed. Use --table_only to view results.")
        print_table(all_results, BASELINE)
        print_timing_table(all_results, BASELINE)
        print_per_axis_breakdown(all_results, BASELINE)
        return

    if args.resume and len(pending) < len(configs):
        print(f"Resuming: {len(configs) - len(pending)} configs already done, "
              f"{len(pending)} remaining.")

    print(f"\n{'═'*70}")
    print(f"  Ablation: {len(pending)} config(s) × {len(seq_dirs)} sequence(s)")
    if args.max_frames:
        print(f"  Capped at {args.max_frames} frames per sequence")
    print(f"  Output: {out_root}")
    print(f"{'═'*70}")

    # ── load model once ───────────────────────────────────────────────────────
    from da3_slam.config import load_slam_config
    base_config = load_slam_config()
    print("\n  Loading depth model (once for all configs)…")
    shared_slam = SharedSLAM(base_config)
    print("  Model ready.\n")

    # ── run ablation ──────────────────────────────────────────────────────────
    for ci, params in enumerate(pending, 1):
        label = _config_label(params)
        print(f"\n{'─'*70}")
        print(f"  Config {ci}/{len(pending)}: {label}")
        print(f"{'─'*70}")

        r = run_config(params, seq_dirs, out_root, args.max_frames, shared_slam, bm)

        # Merge with previous results (replace if re-running same config)
        all_results = [x for x in all_results if x["params"] != params]
        all_results.append(r)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        if r.get("avg"):
            avg = r["avg"]
            print(f"\n  → avg ATE {avg['ate_se3_rmse']:.4f} m  "
                  f"Sim3 {avg['ate_sim3_rmse']:.4f} m  "
                  f"RPE-t {avg['rpe_trans_rmse']:.4f} m  "
                  f"KFs {avg['n_keyframes']:.0f}  "
                  f"wall {avg['wall_seconds']:.0f}s  "
                  f"({avg['s_per_frame']:.3f} s/frame)")

    # ── final tables ──────────────────────────────────────────────────────────
    print_table(all_results, BASELINE)
    print_timing_table(all_results, BASELINE)
    print_per_axis_breakdown(all_results, BASELINE)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {results_path}")


if __name__ == "__main__":
    main()
