"""
Print ablation tables from a saved ablation_results.json file.

Reads the JSON produced by ablation_tum.py and prints the accuracy table,
timing table, and per-axis breakdown — without running any new experiments.

Usage:
    python scripts/show_ablation.py
    python scripts/show_ablation.py --results outputs/ablation/ablation_results.json
    python scripts/show_ablation.py --sort sim3
    python scripts/show_ablation.py --axis submap_size
"""

import argparse
import json
import sys
from pathlib import Path

# Import table-printing functions from the ablation script.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from ablation_tum import (
    BASELINE,
    SWEEPS,
    print_table,
    print_timing_table,
    print_per_axis_breakdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print ablation tables from a saved ablation_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="outputs/ablation/ablation_results.json",
        help="Path to ablation_results.json",
    )
    parser.add_argument(
        "--sort",
        choices=["ate", "sim3", "rpe_t", "rpe_r", "wall", "s_per_frame"],
        default="ate",
        help="Column to sort accuracy table by (ascending)",
    )
    parser.add_argument(
        "--axis",
        default=None,
        help="Only show per-axis breakdown for this parameter "
             f"(choices: {[key for key, _, _ in SWEEPS]}). "
             "Omit to show all axes.",
    )
    parser.add_argument(
        "--no_timing",
        action="store_true",
        help="Skip the timing table",
    )
    parser.add_argument(
        "--no_breakdown",
        action="store_true",
        help="Skip the per-axis breakdown",
    )
    return parser.parse_args()


_SORT_KEYS = {
    "ate":        "ate_se3_rmse",
    "sim3":       "ate_sim3_rmse",
    "rpe_t":      "rpe_trans_rmse",
    "rpe_r":      "rpe_rot_rmse_deg",
    "wall":       "wall_seconds",
    "s_per_frame": "s_per_frame",
}


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        print("Run ablation_tum.py first, or pass --results <path>.")
        sys.exit(1)

    with open(results_path) as f:
        all_results = json.load(f)

    completed = [r for r in all_results if r.get("avg")]
    print(f"Loaded {len(completed)} completed config(s) from {results_path}")

    # ── accuracy table ────────────────────────────────────────────────────────

    sort_metric = _SORT_KEYS[args.sort]

    # Temporarily patch print_table's sort key if a non-default sort was chosen
    if args.sort != "ate":
        _print_accuracy_table_sorted(completed, BASELINE, sort_metric)
    else:
        print_table(all_results, BASELINE)

    # ── timing table ──────────────────────────────────────────────────────────

    if not args.no_timing:
        print_timing_table(all_results, BASELINE)

    # ── per-axis breakdown ────────────────────────────────────────────────────

    if not args.no_breakdown:
        if args.axis is not None:
            valid = [key for key, _, _ in SWEEPS]
            if args.axis not in valid:
                print(f"Unknown axis '{args.axis}'. Valid: {valid}")
                sys.exit(1)
            _print_single_axis(all_results, BASELINE, args.axis)
        else:
            print_per_axis_breakdown(all_results, BASELINE)


def _print_accuracy_table_sorted(
    completed: list[dict], baseline_params: dict, sort_metric: str
) -> None:
    """Re-implementation of print_table with a configurable sort key."""
    if not completed:
        print("No completed configs to report.")
        return

    completed = sorted(completed, key=lambda r: r["avg"].get(sort_metric, float("inf")))

    label_col = max(len(r["label"]) for r in completed) + 2
    header = (
        f"{'Configuration':<{label_col}}  "
        f"{'sub':>4}  {'disp':>5}  {'conf':>5}  {'thr':>5}  "
        f"{'avg ATE':>9}  {'avg Sim3':>9}  "
        f"{'avg RPE-t':>10}  {'avg RPE-r':>10}  "
        f"{'avg KFs':>8}  {'avg LCs':>8}  {'avg wall':>9}"
    )
    sep = "═" * (len(header) + 2)
    sort_label = sort_metric.replace("_", " ")
    print(f"\n{sep}")
    print(f"  ABLATION STUDY  —  ranked by {sort_label} (ascending)")
    print(sep)
    print("  " + header)
    print("  " + "─" * len(header))

    for r in completed:
        p   = r["params"]
        avg = r["avg"]
        marker = " *" if p == baseline_params else "  "
        print(
            f"{marker}{r['label']:<{label_col}}  "
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
    print(f"  Best: {best['label']}  →  {sort_label} {best['avg'].get(sort_metric, '?')}")
    print(f"  (* = baseline config)")
    print(sep)


def _print_single_axis(
    all_results: list[dict], baseline_params: dict, axis_key: str
) -> None:
    """Print per-axis breakdown for a single sweep axis."""
    from ablation_tum import SWEEPS as _SWEEPS

    completed = {tuple(sorted(r["params"].items())): r for r in all_results if r.get("avg")}

    for sweep_key, short, values in _SWEEPS:
        if sweep_key != axis_key:
            continue

        print(f"\n  ── Sweep: {sweep_key} ({'  '.join(str(v) for v in values)}) ──")
        print(f"  {'value':>8}  {'avg ATE':>9}  {'avg Sim3':>9}  "
              f"{'avg KFs':>8}  {'avg LCs':>8}  {'s/frame':>8}")
        print(f"  {'─'*64}")

        import numpy as np
        for val in values:
            cfg = dict(baseline_params)
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


if __name__ == "__main__":
    main()
