"""
Ablation study for DA3-SLAM on TUM RGB-D sequences.

Sweeps key hyperparameters one-at-a-time (OFAT) around a baseline and
reports average ATE across all sequences in a ranked comparison table.

Parameters swept (edit SWEEPS below to change the study):
    submap_size              — keyframes per submap (including 1-frame anchor)
    min_disparity_fraction   — keyframe selection aggressiveness (optical flow)
    lc_distance_threshold    — DINO-SALAD L2 distance for loop closure
                               candidates (lower = stricter)

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

Note: earlier revisions of this script swept a `loop_threshold` parameter
interpreted as a DINOv2 cosine similarity.  Loop closure detection now uses
DINO-SALAD L2 *distance*, so the sweep axis is `lc_distance_threshold` and
old ablation_results.json files (keyed on `loop_threshold`) are not
compatible with the current tables.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from tum_eval_common import SharedSLAM, evaluate_sequence, average_metrics

# ── sweep specification ───────────────────────────────────────────────────────
#
# Baseline: current config/default.yaml values.
BASELINE: dict[str, Any] = {
    "submap_size":            20,
    "min_disparity_fraction": 0.40,
    "confidence_percentile":  65.0,
    "lc_distance_threshold":  0.45,
}

# Each entry: (param_key, display_short, values_to_sweep)
# The baseline value must appear somewhere in each sweep so the baseline row
# is included once and naturally compared to every variant.
#
# confidence_percentile is not swept — well characterised in earlier rounds
# (65 was a stable local optimum).
SWEEPS: list[tuple[str, str, list]] = [
    ("submap_size",            "sub",    [14, 16, 18, 20, 22, 25]),
    ("min_disparity_fraction", "disp",   [0.10, 0.20, 0.30, 0.40, 0.50]),
    ("lc_distance_threshold",  "lc_thr", [0.35, 0.45, 0.55]),
]

# Short display names used in labels and table columns.
SHORT: dict[str, str] = {
    "submap_size":            "sub",
    "min_disparity_fraction": "disp",
    "confidence_percentile":  "conf",
    "lc_distance_threshold":  "lc_thr",
}

# Specific multi-parameter combos (optional — set to [] to skip)
COMBO_CONFIGS: list[dict[str, Any]] = []

# ── config generation ─────────────────────────────────────────────────────────

def config_label(params: dict[str, Any]) -> str:
    """Short human-readable label for a parameter dict."""
    return "  ".join(f"{SHORT.get(k, k)}={v}" for k, v in params.items())

def generate_configs() -> list[dict[str, Any]]:
    """
    Generate OFAT configs: for each sweep axis, vary the target parameter
    while holding all others at their baseline value.

    The baseline itself appears exactly once (deduplicated by param dict).
    Additional combo configs are appended at the end.
    """
    seen: list[dict] = []

    for sweep_key, _, values in SWEEPS:
        for val in values:
            cfg = dict(BASELINE)
            cfg[sweep_key] = val
            if cfg not in seen:
                seen.append(cfg)

    for extra in COMBO_CONFIGS:
        cfg = dict(BASELINE)
        cfg.update(extra)
        if cfg not in seen:
            seen.append(cfg)

    return seen

def build_config(params: dict[str, Any]):
    """Construct a SLAMConfig from an ablation params dict."""
    from da3_slam.config import load_slam_config

    config = load_slam_config(
        submap_size=params["submap_size"],
        confidence_percentile=params["confidence_percentile"],
    )
    # Nested fields not reachable via load_slam_config overrides
    config.keyframe.min_disparity_fraction = params["min_disparity_fraction"]
    config.keyframe.max_submap_size = params["submap_size"]
    config.loop_closure.distance_threshold = params["lc_distance_threshold"]
    return config

# ── per-config runner ─────────────────────────────────────────────────────────

def run_config(
    params: dict[str, Any],
    seq_dirs: list[Path],
    out_root: Path,
    max_frames: int | None,
    shared_slam: SharedSLAM,
) -> dict:
    """
    Run all sequences for one parameter configuration.
    Returns a summary dict with per-sequence metrics and averages.
    """
    label = config_label(params)
    cfg_name = label.replace(" ", "").replace("=", "").replace(".", "p")

    shared_slam.reconfigure(build_config(params))

    seq_metrics: list[dict] = []
    for seq_dir in seq_dirs:
        try:
            m = evaluate_sequence(
                shared_slam.run, seq_dir, out_root / cfg_name / seq_dir.name, max_frames
            )
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {seq_dir.name}: {exc}")
            traceback.print_exc()
            continue
        if m is None:
            continue
        seq_metrics.append(m)
        print(f"  {m['sequence']:<30}  ATE={m['ate_se3_rmse']:.4f}m  "
              f"Sim3={m['ate_sim3_rmse']:.4f}m  "
              f"KFs={m['n_keyframes']}  LCs={m['n_loop_closures']}")

    return {
        "params": params,
        "label": label,
        "sequences": seq_metrics,
        "avg": average_metrics(seq_metrics),
    }

# ── table printing ────────────────────────────────────────────────────────────

def print_table(results: list[dict], baseline_params: dict) -> None:
    """Print ranked ablation table sorted by avg ATE SE3 (ascending)."""
    completed = [r for r in results if r.get("avg")]
    if not completed:
        print("No completed configs to report.")
        return

    completed.sort(key=lambda r: r["avg"]["ate_se3_rmse"])

    label_col = max(len(r["label"]) for r in completed) + 2

    header = (
        f"{'Configuration':<{label_col}}  "
        f"{'sub':>4}  {'disp':>5}  {'conf':>5}  {'lc_thr':>6}  "
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

    for r in completed:
        p = r["params"]
        avg = r["avg"]
        marker = " *" if p == baseline_params else "  "
        print(
            f"{marker}{r['label']:<{label_col}}  "
            f"{p['submap_size']:>4}  "
            f"{p['min_disparity_fraction']:>5.2f}  "
            f"{p['confidence_percentile']:>5.1f}  "
            f"{p['lc_distance_threshold']:>6.2f}  "
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
    print("  (* = baseline config)")
    print(sep)

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
        f"{'sub':>4}  {'disp':>5}  {'conf':>5}  {'lc_thr':>6}  "
        f"{'avg wall':>9}  {'s/frame':>8}  {'s/KF':>8}  "
        f"{'avg KFs':>8}"
    )
    sep = "═" * (len(header) + 2)

    print(f"\n{sep}")
    print("  ABLATION STUDY  —  wall-clock timing (ranked by s/frame, ascending = fastest)")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))

    for r in completed:
        p = r["params"]
        avg = r["avg"]
        marker = " *" if p == baseline_params else "  "
        print(
            f"{marker}{r['label']:<{label_col}}  "
            f"{p['submap_size']:>4}  "
            f"{p['min_disparity_fraction']:>5.2f}  "
            f"{p['confidence_percentile']:>5.1f}  "
            f"{p['lc_distance_threshold']:>6.2f}  "
            f"{avg['wall_seconds']:>9.1f}s  "
            f"{avg['s_per_frame']:>7.3f}s  "
            f"{avg['s_per_kf']:>7.2f}s  "
            f"{avg['n_keyframes']:>8.1f}"
        )

    print("  " + "─" * len(header))
    best = completed[0]
    print(f"  Fastest: {best['label']}  →  {best['avg']['s_per_frame']:.3f} s/frame")
    print("  (* = baseline config)")
    print(sep)

def print_per_axis_breakdown(results: list[dict], baseline_params: dict) -> None:
    """For each sweep axis, print a mini-table of that parameter's effect alone."""
    completed = {tuple(sorted(r["params"].items())): r for r in results if r.get("avg")}

    for sweep_key, _, values in SWEEPS:
        print(f"\n  ── Sweep: {sweep_key} ({'  '.join(str(v) for v in values)}) ──")
        print(f"  {'value':>8}  {'avg ATE':>9}  {'avg Sim3':>9}  "
              f"{'avg KFs':>8}  {'avg LCs':>8}  {'s/frame':>8}")
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

def print_all_tables(results: list[dict]) -> None:
    print_table(results, BASELINE)
    print_timing_table(results, BASELINE)
    print_per_axis_breakdown(results, BASELINE)

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
    args = parse_args()
    configs = generate_configs()

    out_root = Path(args.out_dir)
    results_path = out_root / "ablation_results.json"
    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = [Path(p) for p in args.seq_dir]

    all_results: list[dict] = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing result(s) from {results_path}")

    if args.table_only:
        print_all_tables(all_results)
        return

    completed_params = [r["params"] for r in all_results]
    pending = [c for c in configs if c not in completed_params] \
        if args.resume else configs

    if not pending:
        print("All configs already completed. Use --table_only to view results.")
        print_all_tables(all_results)
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

    from da3_slam.config import load_slam_config
    print("\n  Loading depth model (once for all configs)…")
    shared_slam = SharedSLAM(load_slam_config())
    print("  Model ready.\n")

    for ci, params in enumerate(pending, 1):
        print(f"\n{'─'*70}")
        print(f"  Config {ci}/{len(pending)}: {config_label(params)}")
        print(f"{'─'*70}")

        r = run_config(params, seq_dirs, out_root, args.max_frames, shared_slam)

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

    print_all_tables(all_results)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {results_path}")

if __name__ == "__main__":
    main()
