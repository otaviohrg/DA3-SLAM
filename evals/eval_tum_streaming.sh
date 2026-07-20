#!/bin/bash
# Evaluate DA3-Streaming on the TUM RGB-D freiburg1 benchmark.
# Mirrors the convention of eval_tum.sh: same dataset list, same evo_ape flags,
# same CSV log format (Run,Dataset,RMSE,WallTime).
#
# Usage:
#   ./eval_tum_streaming.sh [config] [n_runs]
#
#   config  - config name under da3_streaming/configs/ (default: tum)
#             or an absolute path to a .yaml file
#   n_runs  - number of full benchmark runs (default: 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

config_arg=${1:-tum}
n=${2:-1}

STREAMING_DIR="/opt/third_party/Depth-Anything-3/da3_streaming"

# Resolve config path
if [ -f "$config_arg" ]; then
    config_path="$(realpath "$config_arg")"
    config_short=$(basename "$config_arg" .yaml)
else
    config_path="${STREAMING_DIR}/configs/${config_arg}.yaml"
    config_short="$config_arg"
fi

if [ ! -f "$config_path" ]; then
    echo "ERROR: Config not found: $config_path"
    exit 1
fi

dataset_path="${REPO_ROOT}/data/tum/"
gt_path="${REPO_ROOT}/data/tum/"
log_path="$(pwd)/logs/tum_streaming_results_${config_short}.txt"

echo "Config:    $config_path"
echo "Datasets:  $dataset_path"
echo "Runs:      $n"
echo "Log:       $log_path"

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
        echo "Running DA3-Streaming on $dataset (Run $run)"
        image_dir="${dataset_path}${dataset}/rgb"
        out_dir="$(pwd)/logs/streaming_${dataset}_run${run}_${config_short}"
        t_start=$(date +%s%3N)
        (cd "$STREAMING_DIR" && python da3_streaming.py \
            --image_dir "$image_dir" \
            --config "$config_path" \
            --output_dir "$out_dir")
        t_end=$(date +%s%3N)
        dataset_times["$dataset"]=$(python3 -c "print(($t_end - $t_start) / 1000.0)")
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        image_dir="${dataset_path}${dataset}/rgb"
        out_dir="$(pwd)/logs/streaming_${dataset}_run${run}_${config_short}"
        est_path="${out_dir}/trajectory_tum.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        python3 "${SCRIPT_DIR}/convert_streaming_poses.py" \
            --poses_file "${out_dir}/camera_poses.txt" \
            --image_dir "$image_dir" \
            --output_file "$est_path"

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
