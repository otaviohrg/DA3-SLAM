#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
submap_size=${1:-20}
model_arg=${2:-nested-giant}
use_ray_pose=${3:-0}

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
        model_id="$model_arg"
        model_short=$(echo "$model_arg" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
        ;;
esac

if [ "$use_ray_pose" = "1" ]; then
    ray_suffix="_raydpose"
    ray_flag="--use_ray_pose"
else
    ray_suffix=""
    ray_flag=""
fi

dataset_path="${REPO_ROOT}/data/Replica/"
log_path="$(pwd)/logs/replica_results_w${submap_size}_${model_short}${ray_suffix}.txt"

echo "Model:      $model_id"
echo "Submap size: $submap_size"
echo "Use ray pose: $use_ray_pose"
echo "Log:        $log_path"

mkdir -p "$(pwd)/logs"

datasets=(
    room0
    room1
    room2
    office0
    office1
    office2
    office3
    office4
)

n=1  # <-- change as needed

if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE,WallTime,WallTotal,ModelLoad,Pipeline,FPS,KFSelMsPerFrame,SubmapSPerSubmap,LCSPerSubmap,KFSelection,SubmapBuilding,LoopClosure,Keyframes,Submaps,LoopClosures,PSNR,SSIM,LPIPS,DepthAbsRel,DepthRMSE,DepthDelta1,Chamfer,Accuracy,Completeness,FScore" > "$log_path"
fi

# Pre-generate GT TUM files (once per scene, reused across runs)
for dataset in "${datasets[@]}"; do
    gt_tum="${dataset_path}${dataset}/gt_tum.txt"
    if [ ! -f "$gt_tum" ]; then
        echo "Converting GT for $dataset..."
        python3 "${SCRIPT_DIR}/convert_replica_gt.py" \
            --traj_file "${dataset_path}${dataset}/traj.txt" \
            --output_file "$gt_tum"
    fi
done

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
    total_psnr=0
    total_ssim=0
    total_lpips=0
    total_depth_absrel=0
    total_depth_rmse=0
    total_depth_delta1=0
    total_chamfer=0
    total_accuracy=0
    total_completeness=0
    total_fscore=0
    count=0
    declare -A dataset_times

    for dataset in "${datasets[@]}"; do
        echo "Running run_slam.py on $dataset (Run $run)"
        image_dir="${dataset_path}${dataset}/results"
        out_dir="$(pwd)/logs/${dataset}_run${run}_w${submap_size}_${model_short}${ray_suffix}"
        t_start=$(date +%s%3N)
        python "${REPO_ROOT}/scripts/run_slam.py" \
            --image_dir "$image_dir" \
            --out_dir "$out_dir" \
            --submap_size "$submap_size" \
            --depth_model "$model_id" \
            $ray_flag
        t_end=$(date +%s%3N)
        dataset_times["$dataset"]=$(python3 -c "print(($t_end - $t_start) / 1000.0)")
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        out_dir="$(pwd)/logs/${dataset}_run${run}_w${submap_size}_${model_short}${ray_suffix}"
        est_path="${out_dir}/trajectory_tum.txt"
        gt_file="${dataset_path}${dataset}/gt_tum.txt"

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

        # ── mapping evaluation ────────────────────────────────────────────
        python3 "${SCRIPT_DIR}/eval_replica_mapping.py" \
            --out_dir   "$out_dir" \
            --scene_dir "${dataset_path}${dataset}" \
            --cam_params "${dataset_path}cam_params.json" || \
            echo "[warn] Mapping eval failed for $dataset — metrics will be empty"

        mapping_json="${out_dir}/mapping_metrics.json"
        read_mapping() {
            if [ -f "$mapping_json" ]; then
                python3 -c "import json; d=json.load(open('$mapping_json')); v=d.get('$1'); print(v if v is not None else '')"
            else
                echo ""
            fi
        }
        psnr_val=$(read_mapping psnr)
        ssim_val=$(read_mapping ssim)
        lpips_val=$(read_mapping lpips)
        depth_absrel=$(read_mapping depth_absrel)
        depth_rmse=$(read_mapping depth_rmse)
        depth_delta1=$(read_mapping depth_delta1)
        chamfer=$(read_mapping chamfer)
        accuracy=$(read_mapping accuracy)
        completeness=$(read_mapping completeness)
        fscore=$(read_mapping f_score)

        echo "$run,$dataset,$rmse,$wall_time,$wall_total,$model_load,$pipeline,$fps,$kf_sel_ms_per_frame,$submap_s_per_submap,$lc_s_per_submap,$kf_sel,$submap_build,$lc,$keyframes,$submaps,$loop_closures,$psnr_val,$ssim_val,$lpips_val,$depth_absrel,$depth_rmse,$depth_delta1,$chamfer,$accuracy,$completeness,$fscore" >> "$log_path"

        safeadd() { python3 -c "a='$1'; b='$2'; print(float(a)+float(b) if a!='' and b!='' else (float(a) if a!='' else float(b) if b!='' else 0))"; }
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
        total_psnr=$(safeadd "$total_psnr" "$psnr_val")
        total_ssim=$(safeadd "$total_ssim" "$ssim_val")
        total_lpips=$(safeadd "$total_lpips" "$lpips_val")
        total_depth_absrel=$(safeadd "$total_depth_absrel" "$depth_absrel")
        total_depth_rmse=$(safeadd "$total_depth_rmse" "$depth_rmse")
        total_depth_delta1=$(safeadd "$total_depth_delta1" "$depth_delta1")
        total_chamfer=$(safeadd "$total_chamfer" "$chamfer")
        total_accuracy=$(safeadd "$total_accuracy" "$accuracy")
        total_completeness=$(safeadd "$total_completeness" "$completeness")
        total_fscore=$(safeadd "$total_fscore" "$fscore")
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
    avg_psnr=$(python3 -c "print($total_psnr / $count)")
    avg_ssim=$(python3 -c "print($total_ssim / $count)")
    avg_lpips=$(python3 -c "print($total_lpips / $count)")
    avg_depth_absrel=$(python3 -c "print($total_depth_absrel / $count)")
    avg_depth_rmse=$(python3 -c "print($total_depth_rmse / $count)")
    avg_depth_delta1=$(python3 -c "print($total_depth_delta1 / $count)")
    avg_chamfer=$(python3 -c "print($total_chamfer / $count)")
    avg_accuracy=$(python3 -c "print($total_accuracy / $count)")
    avg_completeness=$(python3 -c "print($total_completeness / $count)")
    avg_fscore=$(python3 -c "print($total_fscore / $count)")
    echo "$run,Average,$avg_rmse,$avg_wall,$avg_wall_total,$avg_model_load,$avg_pipeline,$avg_fps,$avg_kf_ms,$avg_submap_s,$avg_lc_s,$avg_kf_sel,$avg_submap,$avg_lc,,,$avg_psnr,$avg_ssim,$avg_lpips,$avg_depth_absrel,$avg_depth_rmse,$avg_depth_delta1,$avg_chamfer,$avg_accuracy,$avg_completeness,$avg_fscore" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_rmse"
    echo "Average FPS for run $run: $avg_fps"
    echo "Average submap build: ${avg_submap_s}s/submap"
    echo "Average PSNR for run $run: $avg_psnr"
    echo "Average SSIM for run $run: $avg_ssim"
    echo "Average Chamfer for run $run: $avg_chamfer"
    echo "Average wall time for run $run: ${avg_wall}s"
    echo "Average pipeline time for run $run: ${avg_pipeline}s"
done
