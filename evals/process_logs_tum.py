"""Summarise a TUM eval-harness log CSV (per-run / per-dataset RMSE + wall time).

Reads the CSV written by evals/eval_tum.sh (columns: Run, Dataset, RMSE and
optionally WallTime) and prints per-run, per-dataset and overall averages.

Usage:
    python evals/process_logs_tum.py --submap_size 20 --model nested-giant
    python evals/process_logs_tum.py --log_path logs/tum_results_w20_nested-giant.txt
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TUM results")
    parser.add_argument("--submap_size", type=str, default="20",
                        help="submap size used during eval")
    parser.add_argument("--model", type=str, default="nested-giant",
                        help="model short name used during eval "
                             "(nested-giant, giant, large, base, small)")
    parser.add_argument("--log_path", type=str, default=None,
                        help="explicit path to log file "
                             "(overrides --submap_size / --model)")
    return parser.parse_args()


def print_group_means(df: pd.DataFrame, column: str, group: str,
                      title: str, fmt: str, key_prefix: str = "",
                      suffix: str = "") -> None:
    print(f"\n=== {title} ===")
    for key, value in df.groupby(group)[column].mean().items():
        print(f"{key_prefix}{key}: {value:{fmt}}{suffix}")


def main() -> None:
    args = parse_args()

    if args.log_path:
        log_path = Path(args.log_path)
    else:
        log_path = Path.cwd() / f"logs/tum_results_w{args.submap_size}_{args.model}.txt"

    df = pd.read_csv(log_path)
    df = df[df["Dataset"] != "Average"]
    df["Run"] = df["Run"].astype(int)

    print("=== Per-Experiment RMSE APE (Run x Dataset) ===")
    for run in sorted(df["Run"].unique()):
        print(f"\n--- Run {run} ---")
        for _, row in df[df["Run"] == run].iterrows():
            print(f"{row['Dataset']}: {row['RMSE']:.4f}")

    print_group_means(df, "RMSE", "Run", "Per-Run Average RMSE APE", ".4f",
                      key_prefix="Run ")
    print_group_means(df, "RMSE", "Dataset", "Per-Dataset Average RMSE APE", ".4f")
    print("\n=== Overall Average RMSE APE Across All Runs ===")
    print(f"Overall Average RMSE: {df['RMSE'].mean():.4f}")

    if "WallTime" in df.columns:
        print_group_means(df, "WallTime", "Dataset",
                          "Per-Dataset Average Wall Time", ".1f", suffix="s")
        print_group_means(df, "WallTime", "Run",
                          "Per-Run Average Wall Time", ".1f",
                          key_prefix="Run ", suffix="s")
        print("\n=== Overall Average Wall Time Across All Runs ===")
        print(f"Overall Average Wall Time: {df['WallTime'].mean():.1f}s")


if __name__ == "__main__":
    main()
