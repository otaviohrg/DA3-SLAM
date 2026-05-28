import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Process TUM results")
parser.add_argument("--submap_size", type=str, default="20", help="submap size used during eval")
parser.add_argument("--model", type=str, default="nested-giant",
                    help="model short name used during eval (nested-giant, giant, large, base, small)")
parser.add_argument("--log_path", type=str, default=None, help="explicit path to log file (overrides --submap_size / --model)")
args = parser.parse_args()

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
    run_df = df[df["Run"] == run]
    for _, row in run_df.iterrows():
        print(f"{row['Dataset']}: {row['RMSE']:.4f}")

print("\n=== Per-Run Average RMSE APE ===")
per_run_avg = df.groupby("Run")["RMSE"].mean()
for run, val in per_run_avg.items():
    print(f"Run {run}: {val:.4f}")

print("\n=== Per-Dataset Average RMSE APE ===")
per_dataset_avg = df.groupby("Dataset")["RMSE"].mean()
for dataset, val in per_dataset_avg.items():
    print(f"{dataset}: {val:.4f}")

overall_avg = df["RMSE"].mean()
print("\n=== Overall Average RMSE APE Across All Runs ===")
print(f"Overall Average RMSE: {overall_avg:.4f}")

if "WallTime" in df.columns:
    print("\n=== Per-Dataset Average Wall Time ===")
    per_dataset_time = df.groupby("Dataset")["WallTime"].mean()
    for dataset, val in per_dataset_time.items():
        print(f"{dataset}: {val:.1f}s")

    print("\n=== Per-Run Average Wall Time ===")
    per_run_time = df.groupby("Run")["WallTime"].mean()
    for run, val in per_run_time.items():
        print(f"Run {run}: {val:.1f}s")

    overall_avg_time = df["WallTime"].mean()
    print("\n=== Overall Average Wall Time Across All Runs ===")
    print(f"Overall Average Wall Time: {overall_avg_time:.1f}s")
