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
    echo "Run,Dataset,RMSE,WallTime,WallTotal,ModelLoad,Pipeline,FPS,KFSelMsPerFrame,SubmapSPerSubmap,LCSPerSubmap,KFSelection,SubmapBuilding,LoopClosure,Keyframes,Submaps,LoopClosures" > "$log_path"
fi

for run in $(seq 1 $n); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    total_wall=0
    total_wall_total=0
    total_model_load=0
    total_pipeline=0
    total_kf_sel=0
    total_submap=0
    total_lc=0
    total_fps=0
    total_kf_ms=0
    total_submap_s=0
    total_lc_s=0
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
        out_dir="$(pwd)/logs/${dataset}_run${run}_w${submap_size}_${model_short}${ray_suffix}"
        est_path="${out_dir}/trajectory_tum.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        ape_result=$(evo_ape tum "$gt_file" "$est_path" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        rmse=${rmse:-0}

        timings_json="${out_dir}/timings.json"
        read_timing() { python3 -c "import json; d=json.load(open('$timings_json')); print(d.get('$1', 0))"; }
        model_load=$(read_timing model_load)
        wall_total=$(read_timing wall_total)
        pipeline=$(read_timing pipeline)
        kf_sel=$(read_timing keyframe_selection)
        submap_build=$(read_timing submap_building)
        lc=$(read_timing loop_closure)
        keyframes=$(read_timing keyframes)
        submaps=$(read_timing submaps)
        loop_closures=$(read_timing loop_closures)
        fps=$(read_timing fps)
        kf_sel_ms_per_frame=$(read_timing kf_sel_ms_per_frame)
        submap_s_per_submap=$(read_timing submap_s_per_submap)
        lc_s_per_submap=$(read_timing lc_s_per_submap)
        wall_time=${dataset_times["$dataset"]}

        echo "$run,$dataset,$rmse,$wall_time,$wall_total,$model_load,$pipeline,$fps,$kf_sel_ms_per_frame,$submap_s_per_submap,$lc_s_per_submap,$kf_sel,$submap_build,$lc,$keyframes,$submaps,$loop_closures" >> "$log_path"

        total_rmse=$(python3 -c "print($total_rmse + $rmse)")
        total_wall=$(python3 -c "print($total_wall + $wall_time)")
        total_wall_total=$(python3 -c "print($total_wall_total + $wall_total)")
        total_model_load=$(python3 -c "print($total_model_load + $model_load)")
        total_pipeline=$(python3 -c "print($total_pipeline + $pipeline)")
        total_kf_sel=$(python3 -c "print($total_kf_sel + $kf_sel)")
        total_submap=$(python3 -c "print($total_submap + $submap_build)")
        total_lc=$(python3 -c "print($total_lc + $lc)")
        total_fps=$(python3 -c "print($total_fps + $fps)")
        total_kf_ms=$(python3 -c "print($total_kf_ms + $kf_sel_ms_per_frame)")
        total_submap_s=$(python3 -c "print($total_submap_s + $submap_s_per_submap)")
        total_lc_s=$(python3 -c "print($total_lc_s + $lc_s_per_submap)")
        count=$((count + 1))
    done

    avg_rmse=$(python3 -c "print($total_rmse / $count)")
    avg_wall=$(python3 -c "print($total_wall / $count)")
    avg_wall_total=$(python3 -c "print($total_wall_total / $count)")
    avg_model_load=$(python3 -c "print($total_model_load / $count)")
    avg_pipeline=$(python3 -c "print($total_pipeline / $count)")
    avg_kf_sel=$(python3 -c "print($total_kf_sel / $count)")
    avg_submap=$(python3 -c "print($total_submap / $count)")
    avg_lc=$(python3 -c "print($total_lc / $count)")
    avg_fps=$(python3 -c "print($total_fps / $count)")
    avg_kf_ms=$(python3 -c "print($total_kf_ms / $count)")
    avg_submap_s=$(python3 -c "print($total_submap_s / $count)")
    avg_lc_s=$(python3 -c "print($total_lc_s / $count)")
    echo "$run,Average,$avg_rmse,$avg_wall,$avg_wall_total,$avg_model_load,$avg_pipeline,$avg_fps,$avg_kf_ms,$avg_submap_s,$avg_lc_s,$avg_kf_sel,$avg_submap,$avg_lc,,," >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_rmse"
    echo "Average FPS for run $run: $avg_fps"
    echo "Average submap build time: ${avg_submap_s}s/submap"
    echo "Average wall time for run $run: ${avg_wall}s"
    echo "Average pipeline time for run $run: ${avg_pipeline}s"
done
