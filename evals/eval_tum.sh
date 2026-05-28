#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
submap_size=${1:-20} # Default to 20 (paper setting)
model_arg=${2:-nested-giant} # Default to best model. Options: nested-giant, giant, large, base, small, or full HF ID
use_ray_pose=${3:-0} # Set to 1 to use ray-based pose estimation instead of camera decoder

# Resolve short model aliases to HuggingFace model IDs
case "$model_arg" in
    nested-giant)
        model_id="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
        model_short="nested-giant"
        ;;
    giant)
        model_id="depth-anything/DA3-GIANT-1.1"
        model_short="giant"
        ;;
    large)
        model_id="depth-anything/DA3-LARGE-1.1"
        model_short="large"
        ;;
    base)
        model_id="depth-anything/DA3-BASE"
        model_short="base"
        ;;
    small)
        model_id="depth-anything/DA3-SMALL"
        model_short="small"
        ;;
    *)
        # Accept full HuggingFace model ID directly
        model_id="$model_arg"
        model_short=$(echo "$model_arg" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
        ;;
esac

# Build suffix and ray pose flag for filenames and CLI
if [ "$use_ray_pose" = "1" ]; then
    ray_suffix="_raydpose"
    ray_flag="--use_ray_pose"
else
    ray_suffix=""
    ray_flag=""
fi

dataset_path="${REPO_ROOT}/data/tum/"
gt_path="${REPO_ROOT}/data/tum/"
log_path="$(pwd)/logs/tum_results_w${submap_size}_${model_short}${ray_suffix}.txt"

echo "Model: $model_id"
echo "Submap size: $submap_size"
echo "Use ray pose: $use_ray_pose"
echo "Log: $log_path"

mkdir -p "$(pwd)/logs"

datasets=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

# Number of full runs
n=1  # <-- change as needed

# If file doesn't exist, write header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE,WallTime" > "$log_path"
fi

for run in $(seq 1 $n); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    total_time=0
    count=0
    declare -A dataset_times

    for dataset in "${datasets[@]}"; do
        echo "Running run_slam.py on $dataset (Run $run)"
        image_dir="${dataset_path}${dataset}/rgb"
        out_dir="$(pwd)/logs/${dataset}_run${run}_w${submap_size}_${model_short}${ray_suffix}"
        t_start=$(date +%s%3N)
        python "${REPO_ROOT}/scripts/run_slam.py" \
            --image_dir "$image_dir" \
            --out_dir "$out_dir" \
            --submap_size "$submap_size" \
            --depth_model "$model_id" \
            $ray_flag \
            --skip_ply
        t_end=$(date +%s%3N)
        dataset_times["$dataset"]=$(python3 -c "print(($t_end - $t_start) / 1000.0)")
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        est_path="$(pwd)/logs/${dataset}_run${run}_w${submap_size}_${model_short}${ray_suffix}/trajectory_tum.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        ape_result=$(evo_ape tum "$gt_file" "$est_path" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        rmse=${rmse:-0}

        wall_time=${dataset_times["$dataset"]}
        echo "$run,$dataset,$rmse,$wall_time" >> "$log_path"

        total_rmse=$(python3 -c "print($total_rmse + $rmse)")
        total_time=$(python3 -c "print($total_time + $wall_time)")
        count=$((count + 1))
    done

    avg_rmse=$(python3 -c "print($total_rmse / $count)")
    avg_time=$(python3 -c "print($total_time / $count)")
    echo "$run,Average,$avg_rmse,$avg_time" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_rmse"
    echo "Average wall time for run $run: ${avg_time}s"
done
