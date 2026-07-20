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

# Table-printing functions live in the ablation script.  Both scripts are run
# from the repo root as `python scripts/<name>.py`, which puts scripts/ on
# sys.path, so this import resolves without path manipulation.
from ablation_tum import (
    BASELINE,
    SWEEPS,
    print_table,
    print_timing_table,
    print_per_axis_breakdown,
)

# CLI sort name → key in each result's "avg" dict.
_SORT_KEYS = {
    "ate":         "ate_se3_rmse",
    "sim3":        "ate_sim3_rmse",
    "rpe_t":       "rpe_trans_rmse",
    "rpe_r":       "rpe_rot_rmse_deg",
    "wall":        "wall_seconds",
    "s_per_frame": "s_per_frame",
}


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
        choices=sorted(_SORT_KEYS),
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


def main() -> None:
    """Load a saved ablation_results.json and print the requested tables."""
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

    if args.sort == "ate":
        print_table(all_results, BASELINE)  # default SE3-ATE ranking
    else:
        sort_key = _SORT_KEYS[args.sort]
        print_table(all_results, BASELINE, sort_key=sort_key,
                    sort_label=sort_key.replace("_", " "))

    if not args.no_timing:
        print_timing_table(all_results, BASELINE)

    if not args.no_breakdown:
        if args.axis is not None:
            valid = [key for key, _, _ in SWEEPS]
            if args.axis not in valid:
                print(f"Unknown axis '{args.axis}'. Valid: {valid}")
                sys.exit(1)
            print_per_axis_breakdown(all_results, BASELINE, axes=[args.axis])
        else:
            print_per_axis_breakdown(all_results, BASELINE)


if __name__ == "__main__":
    main()
