"""
Grid search for DA3-SLAM hyperparameters on TUM RGB-D sequences.

Evaluates every combination of the parameter values defined in GRID and
reports average ATE, keyframe count, submap count, loop closure count,
and processing time in ranked summary tables.

Usage — full grid on all fr1 sequences:
    python scripts/grid_search_tum.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_* \\
        --out_dir outputs/grid_search

Usage — quick pass (capped frames) to sanity-check the grid:
    python scripts/grid_search_tum.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_xyz \\
                  data/tum/rgbd_dataset_freiburg1_desk \\
        --out_dir outputs/grid_search_quick \\
        --max_frames 300

Resume an interrupted run (skips configs already saved to JSON):
    python scripts/grid_search_tum.py --seq_dir ... --out_dir ... --resume

Print tables from a previous run without running anything:
    python scripts/grid_search_tum.py --seq_dir ... --out_dir ... --table_only

Preview how many configs the current GRID generates:
    python scripts/grid_search_tum.py --seq_dir ... --dry_run
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# ── Search grid ───────────────────────────────────────────────────────────────
# Edit the lists to define the parameter space.
# Total configurations = product of all list lengths.
# At ~1–3 min per (config × sequence), estimate runtime before running.
#
# Current default: 3×3×3×3 = 81 configs.

GRID: dict[str, list] = {
    "submap_size":            [12, 16, 20],       # frames per submap (incl. anchor overlap)
    "confidence_percentile":  [50.0, 65.0, 75.0], # point-cloud confidence threshold
    "min_disparity_fraction": [0.20, 0.30, 0.40], # min mean optical-flow / image-width
    "lc_distance_threshold":  [0.35, 0.45, 0.55], # DINO-SALAD L2 distance for LC detection
}

# Values that match config/default.yaml — marked with * in tables.
BASELINE: dict[str, Any] = {
    "submap_size":            20,
    "confidence_percentile":  65.0,
    "min_disparity_fraction": 0.40,
    "lc_distance_threshold":  0.45,
}

# Short display names used in table columns and labels.
SHORT: dict[str, str] = {
    "submap_size":            "sub",
    "confidence_percentile":  "conf",
    "min_disparity_fraction": "disp",
    "lc_distance_threshold":  "lc_thr",
}


# ── Config generation ─────────────────────────────────────────────────────────

def generate_configs() -> list[dict[str, Any]]:
    """Return the Cartesian product of all GRID values as a list of param dicts."""
    keys  = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def config_label(params: dict[str, Any]) -> str:
    return "  ".join(f"{SHORT.get(k, k)}={v}" for k, v in params.items())


# ── SLAM wrapper ──────────────────────────────────────────────────────────────

class SharedSLAM:
    """
    Loads the DA3 depth model once and reuses it across all configurations.
    Only the cheap components (SubmapBuilder, LoopClosureDetector) are rebuilt
    for each config.
    """

    def __init__(self, base_config) -> None:
        from da3_slam.slam import DA3SLAM
        self._slam = DA3SLAM(base_config)

    def reconfigure(self, config) -> None:
        """Swap in a new config without reloading the depth model."""
        from da3_slam.backend.inference.submap import SubmapBuilder
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector

        self._slam.config  = config
        self._slam.builder = SubmapBuilder(
            self._slam.estimator,
            confidence_percentile=config.confidence_percentile,
        )
        self._slam.detector = (
            LoopClosureDetector(config.loop_closure, builder=self._slam.builder)
            if config.enable_loop_closure else None
        )

    def run(self, image_paths: list[str]):
        """Run SLAM, resetting the loop-closure detector between sequences."""
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector
        cfg = self._slam.config
        if cfg.enable_loop_closure:
            # Reset so embeddings from a previous sequence don't leak into this one.
            self._slam.detector = LoopClosureDetector(
                cfg.loop_closure, builder=self._slam.builder
            )
        return self._slam.run(image_paths)


def build_config(params: dict[str, Any]):
    """Construct a SLAMConfig from a params dict."""
    from da3_slam.config import load_slam_config

    config = load_slam_config(
        submap_size=params["submap_size"],
        confidence_percentile=params["confidence_percentile"],
    )
    config.keyframe.min_disparity_fraction = params["min_disparity_fraction"]
    config.keyframe.max_submap_size        = params["submap_size"]
    config.loop_closure.distance_threshold = params["lc_distance_threshold"]
    return config


# ── Per-config runner ─────────────────────────────────────────────────────────

def run_config(
    params:       dict[str, Any],
    seq_dirs:     list[Path],
    out_root:     Path,
    max_frames:   int | None,
    shared_slam:  SharedSLAM,
    bm,
) -> dict:
    """
    Run every sequence for one parameter configuration.
    Returns a result dict with per-sequence metrics and cross-sequence averages.
    """
    label    = config_label(params)
    cfg_name = label.replace(" ", "").replace("=", "").replace(".", "p")

    config = build_config(params)
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
            result.save_tum(str(out_dir / "trajectory_est.txt"), timestamps=ts_map)

            est_ts_to_pose: dict[float, np.ndarray] = {
                ts_map[seq_idx]: pose
                for seq_idx, pose in result.keyframe_poses.items()
                if seq_idx in ts_map
            }

            gt_all     = bm.load_groundtruth(gt_txt)
            gt_stamps  = [e[0] for e in gt_all]
            est_stamps = sorted(est_ts_to_pose.keys())
            pairs      = bm.associate(est_stamps, gt_stamps, max_diff=0.02)

            if len(pairs) < 3:
                print(f"  [SKIP] {seq_name}: too few matched poses ({len(pairs)})")
                continue

            gt_matched  = [gt_all[ib][1]                  for _, ib in pairs]
            est_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

            ate_se3  = bm.compute_ate(gt_matched, est_matched, align="se3")
            ate_sim3 = bm.compute_ate(gt_matched, est_matched, align="sim3")
            rpe_1    = bm.compute_rpe(gt_matched, est_matched, delta=1)

            n_submaps_real = len([s for s in result.submaps if not s.is_lc_submap])

            m: dict[str, Any] = {
                "sequence":         seq_name,
                "ate_se3_rmse":     ate_se3["rmse"],
                "ate_sim3_rmse":    ate_sim3["rmse"],
                "rpe_trans_rmse":   rpe_1["trans_rmse"],
                "rpe_rot_rmse_deg": rpe_1["rot_rmse_deg"],
                "n_frames":         len(image_paths),
                "n_keyframes":      result.n_keyframes,
                "n_submaps":        n_submaps_real,
                "n_loop_closures":  len(result.loop_closures),
                "wall_seconds":     round(wall, 1),
                "timings":          result.timings,
            }
            seq_metrics.append(m)

            short_seq = seq_name.replace("rgbd_dataset_freiburg1_", "")
            print(
                f"  {short_seq:<10}  "
                f"ATE={ate_se3['rmse']:.4f}m  Sim3={ate_sim3['rmse']:.4f}m  "
                f"KFs={result.n_keyframes}  sub={n_submaps_real}  "
                f"LCs={len(result.loop_closures)}  "
                f"wall={wall:.0f}s"
            )

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {seq_name}: {exc}")
            traceback.print_exc()

    if not seq_metrics:
        return {"params": params, "label": label, "sequences": [], "avg": {}}

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in seq_metrics]))

    avg_s_per_frame   = float(np.mean(
        [m["wall_seconds"] / m["n_frames"]               for m in seq_metrics]
    ))
    avg_s_per_submap  = float(np.mean(
        [m["wall_seconds"] / max(m["n_submaps"], 1)      for m in seq_metrics]
    ))

    _MODULES = ("keyframe_selection", "submap_building",
                "graph_building", "loop_closure", "optimization")
    module_ms_per_frame: dict[str, float] = {}
    for mod in _MODULES:
        vals = [
            m["timings"].get(mod, 0.0) / max(m["n_frames"], 1) * 1000
            for m in seq_metrics
        ]
        module_ms_per_frame[mod] = float(np.mean(vals))

    return {
        "params":    params,
        "label":     label,
        "sequences": seq_metrics,
        "avg": {
            "ate_se3_rmse":       _mean("ate_se3_rmse"),
            "ate_sim3_rmse":      _mean("ate_sim3_rmse"),
            "rpe_trans_rmse":     _mean("rpe_trans_rmse"),
            "rpe_rot_rmse_deg":   _mean("rpe_rot_rmse_deg"),
            "n_keyframes":        _mean("n_keyframes"),
            "n_submaps":          _mean("n_submaps"),
            "n_loop_closures":    _mean("n_loop_closures"),
            "wall_seconds":       _mean("wall_seconds"),
            "s_per_frame":        avg_s_per_frame,
            "s_per_submap":       avg_s_per_submap,
            "module_ms_per_frame": module_ms_per_frame,
        },
    }


# ── Table printing ────────────────────────────────────────────────────────────

def print_main_table(results: list[dict]) -> None:
    """Print all configs ranked by avg ATE SE3 (ascending = best first)."""
    completed = [r for r in results if r.get("avg") and r["avg"]]
    if not completed:
        print("No completed configurations to report.")
        return

    completed.sort(key=lambda r: r["avg"]["ate_se3_rmse"])

    # Dynamic column widths
    keys = list(GRID.keys())
    col_w = {k: max(len(SHORT[k]), 5) for k in keys}

    header = (
        "  "
        + "  ".join(f"{SHORT[k]:>{col_w[k]}}" for k in keys)
        + "  "
        + "  ".join([
            f"{'avg ATE':>9}",
            f"{'avg Sim3':>9}",
            f"{'RPE-t':>7}",
            f"{'RPE-r°':>7}",
            f"{'avg KFs':>8}",
            f"{'avg sub':>8}",
            f"{'avg LCs':>8}",
            f"{'s/frame':>8}",
            f"{'s/submap':>9}",
        ])
    )
    sep = "═" * len(header)

    print(f"\n{sep}")
    print("  GRID SEARCH — ranked by avg ATE RMSE (SE3 alignment, ↑ = better baseline)")
    print(sep)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in completed:
        p   = r["params"]
        avg = r["avg"]
        is_baseline = (p == BASELINE)
        marker = "* " if is_baseline else "  "
        param_cols = "  ".join(f"{p[k]:>{col_w[k]}}" for k in keys)
        print(
            f"{marker}"
            + param_cols
            + "  "
            + f"{avg['ate_se3_rmse']:>9.4f}"
            + f"  {avg['ate_sim3_rmse']:>9.4f}"
            + f"  {avg['rpe_trans_rmse']:>7.4f}"
            + f"  {avg['rpe_rot_rmse_deg']:>7.3f}"
            + f"  {avg['n_keyframes']:>8.1f}"
            + f"  {avg['n_submaps']:>8.1f}"
            + f"  {avg['n_loop_closures']:>8.1f}"
            + f"  {avg['s_per_frame']:>7.3f}s"
            + f"  {avg['s_per_submap']:>8.1f}s"
        )

    print("  " + "─" * (len(header) - 2))
    best = completed[0]
    print(f"  Best: {best['label']}  →  avg ATE {best['avg']['ate_se3_rmse']:.4f} m")
    print(f"  (* = baseline from config/default.yaml)")
    print(sep)


def print_per_param_breakdown(results: list[dict]) -> None:
    """
    For each grid axis, aggregate over all other dimensions and show the
    marginal effect of that parameter alone.

    Each row is the mean (over all configs sharing that parameter value)
    of avg ATE, avg Sim3, avg KFs, avg submaps, avg LCs, and s/frame.
    """
    completed = [r for r in results if r.get("avg") and r["avg"]]
    if not completed:
        return

    print(f"\n{'═' * 80}")
    print("  PER-PARAMETER MARGINAL EFFECT  (mean over all other parameter values)")
    print(f"{'═' * 80}")

    for key in GRID:
        short = SHORT[key]
        print(f"\n  ── {key} ({short}) ──")
        print(f"  {'value':>10}  {'count':>6}  {'avg ATE':>9}  {'avg Sim3':>9}  "
              f"{'avg KFs':>8}  {'avg sub':>8}  {'avg LCs':>8}  {'s/frame':>8}")
        print(f"  {'─' * 80}")

        for val in GRID[key]:
            matching = [r for r in completed if r["params"][key] == val]
            if not matching:
                print(f"  {val:>10}  {'—':>6}")
                continue

            ate_vals  = [r["avg"]["ate_se3_rmse"]   for r in matching]
            sim3_vals = [r["avg"]["ate_sim3_rmse"]   for r in matching]
            kf_vals   = [r["avg"]["n_keyframes"]     for r in matching]
            sub_vals  = [r["avg"]["n_submaps"]       for r in matching]
            lc_vals   = [r["avg"]["n_loop_closures"] for r in matching]
            spf_vals  = [r["avg"]["s_per_frame"]     for r in matching]

            is_baseline = (val == BASELINE.get(key))
            marker = " *" if is_baseline else "  "
            print(
                f"{marker}{val:>10}"
                f"  {len(matching):>6}"
                f"  {np.mean(ate_vals):>9.4f}"
                f"  {np.mean(sim3_vals):>9.4f}"
                f"  {np.mean(kf_vals):>8.1f}"
                f"  {np.mean(sub_vals):>8.1f}"
                f"  {np.mean(lc_vals):>8.1f}"
                f"  {np.mean(spf_vals):>7.3f}s"
            )

    print(f"\n  (* = baseline value from config/default.yaml)")
    print(f"{'═' * 80}")


def print_sequence_heatmap(results: list[dict]) -> None:
    """
    For each sequence, show the best config found and its ATE.
    Useful for spotting sequences where all configs struggle.
    """
    completed = [r for r in results if r.get("avg") and r["sequences"]]
    if not completed:
        return

    # Collect all sequence names
    all_seqs: list[str] = []
    for r in completed:
        for m in r["sequences"]:
            if m["sequence"] not in all_seqs:
                all_seqs.append(m["sequence"])
    all_seqs.sort()

    short_seqs = [s.replace("rgbd_dataset_freiburg1_", "") for s in all_seqs]
    col = max(len(s) for s in short_seqs) + 2

    print(f"\n{'═' * 70}")
    print("  PER-SEQUENCE BEST CONFIG  (lowest ATE RMSE across all grid configs)")
    print(f"{'═' * 70}")
    print(f"  {'sequence':<{col}}  {'best ATE':>9}  {'best config'}")
    print(f"  {'─' * 66}")

    for seq, short in zip(all_seqs, short_seqs):
        candidates = []
        for r in completed:
            for m in r["sequences"]:
                if m["sequence"] == seq:
                    candidates.append((m["ate_se3_rmse"], r["label"]))
        if not candidates:
            continue
        candidates.sort()
        best_ate, best_label = candidates[0]
        print(f"  {short:<{col}}  {best_ate:>9.4f}m  {best_label}")

    print(f"{'═' * 70}")


def print_timing_breakdown(results: list[dict]) -> None:
    """
    Show per-module average timing (ms/frame) for each configuration,
    ranked by total compute time (fastest first).

    Modules: keyframe_selection, submap_building, graph_building,
             loop_closure, optimization.
    """
    _MODULES = ("keyframe_selection", "submap_building",
                "graph_building", "loop_closure", "optimization")
    _SHORT_MOD = {
        "keyframe_selection": "KF-sel",
        "submap_building":    "sub-bld",
        "graph_building":     "graph",
        "loop_closure":       "LC",
        "optimization":       "optim",
    }

    completed = [
        r for r in results
        if r.get("avg") and r["avg"].get("module_ms_per_frame")
    ]
    if not completed:
        return

    completed.sort(key=lambda r: r["avg"]["s_per_frame"])

    keys = list(GRID.keys())
    col_w = {k: max(len(SHORT[k]), 5) for k in keys}

    mod_w = 9  # column width for each module (ms)

    header = (
        "  "
        + "  ".join(f"{SHORT[k]:>{col_w[k]}}" for k in keys)
        + "  "
        + f"{'total ms/f':>10}"
        + "  "
        + "  ".join(f"{_SHORT_MOD[m]:>{mod_w}}" for m in _MODULES)
    )
    sep = "═" * len(header)

    print(f"\n{sep}")
    print("  TIMING BREAKDOWN — ms per frame  (↓ = faster; fastest first)")
    print(sep)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in completed:
        p   = r["params"]
        avg = r["avg"]
        mod = avg.get("module_ms_per_frame", {})
        is_baseline = (p == BASELINE)
        marker = "* " if is_baseline else "  "
        param_cols = "  ".join(f"{p[k]:>{col_w[k]}}" for k in keys)
        total_ms = avg["s_per_frame"] * 1000
        mod_cols = "  ".join(
            f"{mod.get(m, 0.0):>{mod_w}.1f}" for m in _MODULES
        )
        print(f"{marker}{param_cols}  {total_ms:>10.1f}  {mod_cols}")

    print("  " + "─" * (len(header) - 2))
    best = completed[0]
    print(f"  Fastest: {best['label']}  →  {best['avg']['s_per_frame']*1000:.1f} ms/frame")
    print(f"  (* = baseline from config/default.yaml)")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for DA3-SLAM hyperparameters on TUM RGB-D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seq_dir", nargs="+", required=True,
        help="TUM sequence directory/directories to benchmark",
    )
    parser.add_argument(
        "--out_dir", default="outputs/grid_search",
        help="Root output directory (per-config sub-dirs are created automatically)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Cap frames per sequence (useful for quick sanity checks)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configurations already present in grid_results.json",
    )
    parser.add_argument(
        "--table_only", action="store_true",
        help="Print tables from saved grid_results.json without running any new configs",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the grid size and config list, then exit",
    )
    return parser.parse_args()


def _import_benchmark():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import benchmark_tum as bm
    return bm


def main() -> None:
    args    = parse_args()
    configs = generate_configs()
    n_seqs  = len(args.seq_dir)

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nGrid definition:")
        for k, vals in GRID.items():
            print(f"  {SHORT[k]:<8} = {vals}")
        print(f"\nTotal configurations : {len(configs)}")
        print(f"Sequences            : {n_seqs}")
        print(f"Total SLAM runs      : {len(configs) * n_seqs}")
        print(f"\nAll configs ({len(configs)}):")
        for i, cfg in enumerate(configs, 1):
            marker = " *" if cfg == BASELINE else "  "
            print(f"{marker}{i:>4}.  {config_label(cfg)}")
        return

    out_root     = Path(args.out_dir)
    results_path = out_root / "grid_results.json"
    out_root.mkdir(parents=True, exist_ok=True)

    # ── load previous results ─────────────────────────────────────────────────
    all_results: list[dict] = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing result(s) from {results_path}")

    bm = _import_benchmark()

    if args.table_only:
        print_main_table(all_results)
        print_per_param_breakdown(all_results)
        print_sequence_heatmap(all_results)
        print_timing_breakdown(all_results)
        return

    # ── select pending configs ────────────────────────────────────────────────
    completed_params = [r["params"] for r in all_results]
    pending = (
        [c for c in configs if c not in completed_params]
        if args.resume
        else configs
    )

    if not pending:
        print("All configurations already completed. Use --table_only to view results.")
        print_main_table(all_results)
        print_per_param_breakdown(all_results)
        print_sequence_heatmap(all_results)
        print_timing_breakdown(all_results)
        return

    if args.resume and len(pending) < len(configs):
        print(f"Resuming: {len(configs) - len(pending)} done, {len(pending)} remaining.")

    seq_dirs = [Path(p) for p in args.seq_dir]

    print(f"\n{'═' * 72}")
    print(f"  Grid search: {len(pending)} config(s) × {len(seq_dirs)} sequence(s)"
          f" = {len(pending) * len(seq_dirs)} run(s)")
    if args.max_frames:
        print(f"  Capped at {args.max_frames} frames per sequence")
    print(f"  Output: {out_root}")
    print(f"  Grid: " + "  ".join(f"{SHORT[k]}∈{GRID[k]}" for k in GRID))
    print(f"{'═' * 72}\n")

    # ── load model once ───────────────────────────────────────────────────────
    from da3_slam.config import load_slam_config
    print("  Loading depth model (once for all configs)…")
    shared_slam = SharedSLAM(load_slam_config())
    print("  Model ready.\n")

    # ── run grid ──────────────────────────────────────────────────────────────
    total_start = time.time()

    for ci, params in enumerate(pending, 1):
        label = config_label(params)
        is_baseline = (params == BASELINE)
        marker = " [baseline]" if is_baseline else ""
        print(f"\n{'─' * 72}")
        print(f"  Config {ci}/{len(pending)}{marker}: {label}")
        print(f"{'─' * 72}")

        r = run_config(params, seq_dirs, out_root, args.max_frames, shared_slam, bm)

        # Replace or append
        all_results = [x for x in all_results if x["params"] != params]
        all_results.append(r)

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        if r.get("avg") and r["avg"]:
            avg = r["avg"]
            elapsed = time.time() - total_start
            remaining = elapsed / ci * (len(pending) - ci)
            mod = avg.get("module_ms_per_frame", {})
            mod_str = "  ".join(
                f"{k.split('_')[0]}={v:.0f}ms"
                for k, v in mod.items() if v > 0
            )
            print(
                f"\n  → avg ATE {avg['ate_se3_rmse']:.4f} m  "
                f"Sim3 {avg['ate_sim3_rmse']:.4f} m  "
                f"KFs {avg['n_keyframes']:.0f}  "
                f"sub {avg['n_submaps']:.0f}  "
                f"LCs {avg['n_loop_closures']:.0f}  "
                f"{avg['s_per_frame']*1000:.1f} ms/frame"
                + (f"  [{mod_str}]" if mod_str else "")
                + f"  [ETA ~{remaining/60:.0f} min]"
            )

    # ── final tables ──────────────────────────────────────────────────────────
    print_main_table(all_results)
    print_per_param_breakdown(all_results)
    print_sequence_heatmap(all_results)
    print_timing_breakdown(all_results)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\nTotal wall time: {total_elapsed / 60:.1f} min")
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
