#!/bin/bash
# Re-evaluate existing SLAM outputs without re-running SLAM.
# Finds output directories matching the given parameters, runs evo_ape and
# (for Replica) the mapping evaluation, then writes a fresh log.
#
# Usage:
#   ./evals/reeval.sh [dataset_type] [submap_size] [model] [use_ray_pose] [run] [force_mapping]
#
#   dataset_type  : tum | replica  (default: replica)
#   submap_size   : (default: 20)
#   model         : nested-giant | giant | large | base | small | full HF ID  (default: nested-giant)
#   use_ray_pose  : 0 | 1  (default: 0)
#   run           : run number, or 'all' to process every available run  (default: all)
#   force_mapping : 0 | 1 — recompute mapping_metrics.json even if cached  (default: 0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

dataset_type=${1:-replica}
submap_size=${2:-20}
model_arg=${3:-nested-giant}
use_ray_pose=${4:-0}
run_arg=${5:-all}
force_mapping=${6:-0}

# ── model alias resolution ────────────────────────────────────────────────────
case "$model_arg" in
    nested-giant) model_short="nested-giant" ;;
    giant)        model_short="giant" ;;
    large)        model_short="large" ;;
    base)         model_short="base" ;;
    small)        model_short="small" ;;
    *)            model_short=$(echo "$model_arg" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]') ;;
esac

if [ "$use_ray_pose" = "1" ]; then
    ray_suffix="_raydpose"
else
    ray_suffix=""
fi

dir_suffix="w${submap_size}_${model_short}${ray_suffix}"

# ── dataset configuration ─────────────────────────────────────────────────────
case "$dataset_type" in
    tum)
        dataset_path="${REPO_ROOT}/data/tum/"
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
        has_mapping=0
        gt_file_fn() { echo "${dataset_path}${1}/groundtruth.txt"; }
        log_path="$(pwd)/logs/tum_reeval_${dir_suffix}.txt"
        log_header="Run,Dataset,RMSE,WallTime,WallTotal,ModelLoad,Pipeline,FPS,KFSelMsPerFrame,SubmapSPerSubmap,LCSPerSubmap,KFSelection,SubmapBuilding,LoopClosure,Keyframes,Submaps,LoopClosures"
        ;;
    replica)
        dataset_path="${REPO_ROOT}/data/Replica/"
        datasets=(room0 room1 room2 office0 office1 office2 office3 office4)
        has_mapping=1
        gt_file_fn() { echo "${dataset_path}${1}/gt_tum.txt"; }
        log_path="$(pwd)/logs/replica_reeval_${dir_suffix}.txt"
        log_header="Run,Dataset,RMSE,WallTime,WallTotal,ModelLoad,Pipeline,FPS,KFSelMsPerFrame,SubmapSPerSubmap,LCSPerSubmap,KFSelection,SubmapBuilding,LoopClosure,Keyframes,Submaps,LoopClosures,PSNR,SSIM,LPIPS,DepthAbsRel,DepthRMSE,DepthDelta1,Chamfer,Accuracy,Completeness,FScore"
        ;;
    *)
        echo "ERROR: unknown dataset_type '$dataset_type'. Use 'tum' or 'replica'."
        exit 1
        ;;
esac

echo "Dataset type: $dataset_type"
echo "Config:       w${submap_size} ${model_short}${ray_suffix}"
echo "Runs:         ${run_arg}"
echo "Force remap:  ${force_mapping}"
echo "Log:          $log_path"

mkdir -p "$(pwd)/logs"

# ── discover runs ─────────────────────────────────────────────────────────────
if [ "$run_arg" = "all" ]; then
    # Infer available run numbers from existing directories
    runs=()
    for dataset in "${datasets[@]}"; do
        for d in "$(pwd)/logs/${dataset}_run"*"_${dir_suffix}"; do
            [ -d "$d" ] || continue
            r=$(echo "$d" | sed -E "s|.*_run([0-9]+)_.*|\1|")
            runs+=("$r")
        done
    done
    # Deduplicate and sort
    readarray -t runs < <(printf '%s\n' "${runs[@]}" | sort -nu)
    if [ ${#runs[@]} -eq 0 ]; then
        echo "No existing output directories found matching *_run*_${dir_suffix}"
        exit 1
    fi
    echo "Found runs: ${runs[*]}"
else
    runs=("$run_arg")
fi

# ── pre-generate Replica GT TUM files if needed ───────────────────────────────
if [ "$dataset_type" = "replica" ]; then
    for dataset in "${datasets[@]}"; do
        gt_tum="${dataset_path}${dataset}/gt_tum.txt"
        if [ ! -f "$gt_tum" ]; then
            echo "Converting GT for $dataset..."
            python3 "${SCRIPT_DIR}/convert_replica_gt.py" \
                --traj_file "${dataset_path}${dataset}/traj.txt" \
                --output_file "$gt_tum"
        fi
    done
fi

# ── write header ──────────────────────────────────────────────────────────────
echo "$log_header" > "$log_path"

# ── evaluate ──────────────────────────────────────────────────────────────────
for run in "${runs[@]}"; do
    echo "==== Evaluating Run $run ===="

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

    for dataset in "${datasets[@]}"; do
        out_dir="$(pwd)/logs/${dataset}_run${run}_${dir_suffix}"

        if [ ! -d "$out_dir" ]; then
            echo "  [skip] $dataset — output dir not found: $out_dir"
            continue
        fi

        est_path="${out_dir}/trajectory_tum.txt"
        if [ ! -f "$est_path" ]; then
            echo "  [skip] $dataset — trajectory_tum.txt missing"
            continue
        fi

        echo "  Evaluating $dataset (Run $run)"
        gt_file=$(gt_file_fn "$dataset")

        # ── pose evaluation ───────────────────────────────────────────────────
        ape_result=$(evo_ape tum "$gt_file" "$est_path" -as 2>/dev/null)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        rmse=${rmse:-0}

        # ── timing from saved JSON ────────────────────────────────────────────
        timings_json="${out_dir}/timings.json"
        read_timing() {
            if [ -f "$timings_json" ]; then
                python3 -c "import json; d=json.load(open('$timings_json')); print(d.get('$1', ''))"
            else
                echo ""
            fi
        }
        wall_time=$(read_timing wall_total)   # use internal timer as WallTime here
        wall_total=$(read_timing wall_total)
        model_load=$(read_timing model_load)
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

        # ── mapping evaluation (Replica only) ─────────────────────────────────
        psnr_val="" ssim_val="" lpips_val=""
        depth_absrel="" depth_rmse="" depth_delta1=""
        chamfer="" accuracy="" completeness="" fscore=""

        if [ "$has_mapping" = "1" ]; then
            mapping_json="${out_dir}/mapping_metrics.json"

            if [ ! -f "$mapping_json" ] || [ "$force_mapping" = "1" ]; then
                python3 "${SCRIPT_DIR}/eval_replica_mapping.py" \
                    --out_dir   "$out_dir" \
                    --scene_dir "${dataset_path}${dataset}" \
                    --cam_params "${dataset_path}cam_params.json" || \
                    echo "  [warn] Mapping eval failed for $dataset"
            else
                echo "  [cached] Using existing mapping_metrics.json"
            fi

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
        fi

        # ── log row ───────────────────────────────────────────────────────────
        if [ "$has_mapping" = "1" ]; then
            echo "$run,$dataset,$rmse,$wall_time,$wall_total,$model_load,$pipeline,$fps,$kf_sel_ms_per_frame,$submap_s_per_submap,$lc_s_per_submap,$kf_sel,$submap_build,$lc,$keyframes,$submaps,$loop_closures,$psnr_val,$ssim_val,$lpips_val,$depth_absrel,$depth_rmse,$depth_delta1,$chamfer,$accuracy,$completeness,$fscore" >> "$log_path"
        else
            echo "$run,$dataset,$rmse,$wall_time,$wall_total,$model_load,$pipeline,$fps,$kf_sel_ms_per_frame,$submap_s_per_submap,$lc_s_per_submap,$kf_sel,$submap_build,$lc,$keyframes,$submaps,$loop_closures" >> "$log_path"
        fi

        # ── accumulate ────────────────────────────────────────────────────────
        safeadd() { python3 -c "a='$1'; b='$2'; print(float(a)+float(b) if a!='' and b!='' else (float(a) if a!='' else float(b) if b!='' else 0))"; }
        total_rmse=$(safeadd "$total_rmse" "$rmse")
        total_wall=$(safeadd "$total_wall" "$wall_time")
        total_wall_total=$(safeadd "$total_wall_total" "$wall_total")
        total_model_load=$(safeadd "$total_model_load" "$model_load")
        total_pipeline=$(safeadd "$total_pipeline" "$pipeline")
        total_kf_sel=$(safeadd "$total_kf_sel" "$kf_sel")
        total_submap=$(safeadd "$total_submap" "$submap_build")
        total_lc=$(safeadd "$total_lc" "$lc")
        total_fps=$(safeadd "$total_fps" "$fps")
        total_kf_ms=$(safeadd "$total_kf_ms" "$kf_sel_ms_per_frame")
        total_submap_s=$(safeadd "$total_submap_s" "$submap_s_per_submap")
        total_lc_s=$(safeadd "$total_lc_s" "$lc_s_per_submap")
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

    [ "$count" -eq 0 ] && echo "  No datasets evaluated for run $run" && continue

    avg() { python3 -c "print($1 / $count)"; }
    avg_rmse=$(avg "$total_rmse")
    avg_wall=$(avg "$total_wall")
    avg_wall_total=$(avg "$total_wall_total")
    avg_model_load=$(avg "$total_model_load")
    avg_pipeline=$(avg "$total_pipeline")
    avg_kf_sel=$(avg "$total_kf_sel")
    avg_submap=$(avg "$total_submap")
    avg_lc=$(avg "$total_lc")
    avg_fps=$(avg "$total_fps")
    avg_kf_ms=$(avg "$total_kf_ms")
    avg_submap_s=$(avg "$total_submap_s")
    avg_lc_s=$(avg "$total_lc_s")
    avg_psnr=$(avg "$total_psnr")
    avg_ssim=$(avg "$total_ssim")
    avg_lpips=$(avg "$total_lpips")
    avg_depth_absrel=$(avg "$total_depth_absrel")
    avg_depth_rmse=$(avg "$total_depth_rmse")
    avg_depth_delta1=$(avg "$total_depth_delta1")
    avg_chamfer=$(avg "$total_chamfer")
    avg_accuracy=$(avg "$total_accuracy")
    avg_completeness=$(avg "$total_completeness")
    avg_fscore=$(avg "$total_fscore")

    if [ "$has_mapping" = "1" ]; then
        echo "$run,Average,$avg_rmse,$avg_wall,$avg_wall_total,$avg_model_load,$avg_pipeline,$avg_fps,$avg_kf_ms,$avg_submap_s,$avg_lc_s,$avg_kf_sel,$avg_submap,$avg_lc,,,$avg_psnr,$avg_ssim,$avg_lpips,$avg_depth_absrel,$avg_depth_rmse,$avg_depth_delta1,$avg_chamfer,$avg_accuracy,$avg_completeness,$avg_fscore" >> "$log_path"
    else
        echo "$run,Average,$avg_rmse,$avg_wall,$avg_wall_total,$avg_model_load,$avg_pipeline,$avg_fps,$avg_kf_ms,$avg_submap_s,$avg_lc_s,$avg_kf_sel,$avg_submap,$avg_lc,,," >> "$log_path"
    fi

    echo "==== Run $run complete (${count} datasets) ===="
    echo "  Average RMSE:          $avg_rmse"
    echo "  Average FPS:           $avg_fps"
    echo "  Average submap build:  ${avg_submap_s}s/submap"
    echo "  Average LC:            ${avg_lc_s}s/submap"
    [ "$has_mapping" = "1" ] && echo "  Average PSNR:     $avg_psnr"
    [ "$has_mapping" = "1" ] && echo "  Average Chamfer:  $avg_chamfer"
    echo "  Average pipeline:      ${avg_pipeline}s"
done

echo ""
echo "Log written to $log_path"
