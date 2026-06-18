# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DA3-SLAM is a monocular RGB SLAM system built on top of **Depth Anything 3 (DA3)**. It feeds batches of keyframes through DA3 to get per-frame metric depth, confidence, and camera poses, assembles them into local **submaps**, stitches submaps into a global **SL(4) pose graph** (GTSAM), and closes loops via DINOv2/SALAD descriptor retrieval + DA3 re-inference. The architecture deliberately mirrors **VGGT-SLAM** (frame overlap, SL(4) factor graph, loop-closure wiring) — comments throughout reference that lineage, and it is the right mental model when reasoning about design choices.

## Environment & setup

- Python **≥3.11**, CUDA 12.8, PyTorch 2.10. GPU required (DA3 + DINOv2 inference).
- Two heavy dependencies are **not on PyPI** and are cloned + installed from source by `setup.sh` into `/opt/third_party`: [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (provides `depth_anything_3.api`) and [SALAD](https://github.com/Dominic101/salad). Plain `pip install -r requirements.txt` is **not sufficient** — the pipeline will fail at `import depth_anything_3` without these.
- `uv` is the package manager. `setup.sh` assumes the repo is mounted at `/app` (Docker layout). The `Dockerfile` builds the full CUDA environment and runs `setup.sh`.
- GTSAM must be a build with **SL4 support** (`gtsam.SL4`, `PriorFactorSL4`, `BetweenFactorSL4`). Stock PyPI `gtsam` may not have these symbols — if `factor_graph.py` fails to import, that's why.

## Common commands

```bash
# Run the full pipeline on a directory of images
python scripts/run_slam.py --image_dir data/video1_30fps --out_dir outputs/run1

# Common overrides (any config/default.yaml scalar can be overridden on the CLI)
python scripts/run_slam.py --image_dir DIR --submap_size 12 \
    --confidence_percentile 75 --no_loop_closure --max_frames 200 --skip_ply

# Benchmark / ablation on TUM
python scripts/benchmark_tum.py
python scripts/ablation_tum.py
python scripts/grid_search_tum.py
python scripts/show_ablation.py        # render ablation results

# Benchmark on EuRoC MAV (ASL format; uses cam0, undistorts by default)
python scripts/benchmark_euroc.py --seq_dir data/EuRoC/vicon_room2/V2_01_easy --out_dir outputs/euroc/V2_01

# Component-level smoke tests (these are plain scripts, not pytest)
python scripts/test_keyframe_selector.py
python scripts/test_submap.py
python scripts/test_depth_estimator.py
python scripts/test_factor_graph.py
python scripts/test_loop_closure.py
python scripts/test_alignment.py
python scripts/test_pose_graph.py

# Visualization / debugging
python scripts/visualize_trajectory.py
python scripts/visualize_map.py
python scripts/debug_map.py

# Evaluation harnesses (bash drivers that run SLAM + score with evo_ape)
evals/eval_tum.sh [submap_size] [model] [use_ray_pose]   # e.g. ./evals/eval_tum.sh 20 nested-giant 0
evals/eval_euroc.sh [submap_size] [model] [use_ray_pose] # EuRoC; converts GT to camera-frame TUM first
evals/eval_replica.sh
evals/reeval.sh                                           # re-score existing output trajectories
```

There is **no pytest suite, no linter config, and no Makefile**. The `scripts/test_*.py` files are runnable smoke/integration scripts, not a unit-test framework (`test_pose_graph.py` and `test_keyframe_selector.py` run without a GPU; the rest need DA3). They share `scripts/smoke_test_utils.py` (header/check/image helpers). The TUM drivers (`benchmark_tum.py`, `ablation_tum.py`, `grid_search_tum.py`) share dataset parsing, ATE/RPE metrics, and the model-reusing `SharedSLAM` wrapper via `scripts/tum_eval_common.py`; `benchmark_euroc.py` reuses that same metric/`SharedSLAM` machinery with EuRoC-specific loaders. Model aliases accepted by the eval scripts: `nested-giant` (default/best), `giant`, `large`, `base`, `small`, or a full HuggingFace ID.

EuRoC support has two entry points sharing the dataset I/O in `scripts/euroc_common.py` (auto-discovers `mav0` incl. the doubly-nested `<seq>/<seq>/mav0` unpack; loads cam0 frames; converts GT from the IMU/body frame to the camera frame via `T_BS` from `cam0/sensor.yaml`, GT quaternions are `w,x,y,z`; optional radtan undistortion):
- **`scripts/benchmark_euroc.py`** — in-house ATE/RPE (mirrors `benchmark_tum.py`), undistorts by default (`--no_undistort` to skip), timestamps in seconds.
- **`evals/eval_euroc.sh`** — evo-based harness (mirrors `eval_tum.sh`): runs `run_slam.py` on the raw `mav0/cam0/data` frames, then `evals/convert_euroc_gt.py` writes a camera-frame TUM GT in **nanoseconds** to match run_slam's filename-derived timestamps, and `evo_ape ... -as --t_max_diff 20000000` (20 ms) scores it.

The evo wrappers `scripts/evo_compare.py` and `scripts/evo_batch.py` also take EuRoC: pass a EuRoC **sequence directory** (one containing `mav0/`) as the GT — `evo_compare.py compare --gt data/EuRoC/.../V2_01_easy --est <traj>` or `evo_batch.py --dataset euroc --gt_dir data/EuRoC --est_dir logs/` (also auto-detected). They convert the GT and rescale ns→s automatically (`scripts/evo_euroc.py` + `scripts/euroc_common.py`).

Monocular scale is arbitrary, so the **Sim3-aligned ATE** (evo `-as` / `--correct_scale`, or benchmark_euroc's Sim3 column) is the headline metric.

## Configuration model

`config/default.yaml` holds **all authoritative defaults**. All config dataclasses (`SLAMConfig`, `NoiseConfig`, `LoopClosureConfig`) live in `da3_slam/config.py` except `KeyframeSelectorConfig` (next to its component); the component modules re-export their config class. **`da3_slam.config` imports without the GPU stack** (no torch/gtsam/DA3) — `da3_slam/__init__.py` is lazy (PEP 562), so config loading and CLI `--help` work on machines without CUDA. The flow is:

1. `scripts/run_slam.py` loads the YAML to seed `argparse` defaults, then parses CLI overrides on top.
2. `da3_slam.config.load_slam_config(yaml_path, **overrides)` builds the typed `SLAMConfig`. Only **top-level scalar** keys can be passed as `overrides`; nested keys (keyframe/noise/loop_closure) are read straight from the YAML, so boolean/nested CLI flags are applied by mutating the returned config object afterward (see `build_config()` in `run_slam.py`).

When adding a tunable parameter: add it to `config/default.yaml`, to the relevant `@dataclass` in `da3_slam/config.py` (or `KeyframeSelectorConfig`), thread it through `load_slam_config()`, and optionally expose a CLI flag in `run_slam.py`.

Loop closure thresholding is an **L2 distance** on DINO-SALAD descriptors (`loop_closure.distance_threshold`, lower = stricter). The CLI flag is `--loop_distance_threshold` (`--loop_threshold` is kept as an alias).

## Pipeline architecture

The runner is `da3_slam/slam.py` → `DA3SLAM.run()`. It executes **four daemon threads** communicating over bounded `queue.Queue`s (a `_RunContext` dataclass holds all shared state, queues, timings, and a shared `backend_error` for cross-thread failure propagation):

```
_frontend  →(batch_queue)→  _inference  →(submap_queue)→  _processing  ⇄(lc_queue/lc_result_queue)⇄  _loop_closure_worker
```

- **`_frontend`** (`frontend/keyframe_selector.py`): `OnlineKeyframeSelector` uses Lucas-Kanade optical flow; a frame becomes a keyframe when mean tracked displacement exceeds `min_disparity_fraction * W`. Keyframes are batched into groups of `submap_size`. **Each submap shares its last keyframe as the first frame ("anchor") of the next submap** — this 1-frame overlap is structurally load-bearing (see below). (`KeyframeSelector` is a batch wrapper around the online selector for scripts.)
- **`_inference`** (`backend/inference/`): `SubmapBuilder` runs DA3 (`DepthEstimator` wraps `depth_anything_3.api`) on each keyframe batch, confidence-filters point clouds with **one threshold computed globally across the batch** (`confidence_percentile`), and produces a `Submap` of `Frame`s. Optionally attaches CLIP `semantic_vector`s via `SemanticEmbedder`.
- **`_processing`** (`backend/processing/factor_graph.py`): builds the GTSAM `PoseGraph` incrementally — one SL(4) node per keyframe keyed by `seq_idx`, between-factors for consecutive frames, a tight prior on frame 0. Runs incremental `optimize()` after each submap and a final optimization at the end.
- **`_loop_closure_worker`** (`backend/processing/loop_closure.py`): `LoopClosureDetector.process(submap)` matches per-frame DINO-SALAD descriptors against all previous frames (top-K priority queue, `min_submaps_apart` gap), re-runs DA3 on each matched image pair, gates on mean DA3 confidence, and returns `LoopClosure` objects carrying `relative_b_to_a` (the measured query→detected cam-to-world transform). The worker rescales its translation to global metric units and posts a between-factor (`loop=True`) back to `_processing`.

### Two concepts that are non-obvious and easy to break

1. **Anchor-frame submap bridging (no explicit inter-submap factor).** Consecutive submaps are joined *only* because submap N's last frame and submap N+1's first frame are the **same `seq_idx`**. `_processing` places submap N+1's frames in the global frame by composing the anchor's already-optimized global pose with DA3's local relative poses. `_build_keyframe_poses` dedupes the shared anchor ("first submap wins"). If you change how keyframes are batched in `_frontend`, preserve the 1-frame overlap or the graph disconnects.

2. **Per-submap metric scale accumulation.** Each DA3 batch has its own arbitrary metric scale. `_estimate_boundary_scale` takes the **median depth ratio at the shared anchor frame** between consecutive submaps; this `delta_scale` is multiplied into an `accumulated_scale` that scales the **translation component** of every between-factor (and loop-closure) measurement before it enters the graph (`_scaled_translation`). Loop closures apply an additional depth-scale correction (`_estimate_depth_scale` between the query frame's depth and the LC re-inference's depth) times the query submap's accumulated scale. Rotation is never scaled — only translation.

## Outputs

`run_slam.py` writes to `--out_dir`: `trajectory_kitti.txt` (12-value rows), `trajectory_tum.txt` (`timestamp tx ty tz qx qy qz qw`; real timestamps inferred from numeric filenames like TUM, else `seq_idx/fps`), `map.ply` (binary, merged colored cloud projected by optimized poses; skip with `--skip_ply`), and `timings.json`. `SLAMResult` (in `slam.py`) owns all the `save_*` methods and `retrieve_best_semantic_frame()` for open-vocab CLIP queries.

## Data & evaluation

`data/` holds datasets (TUM, Replica, custom `video*` dirs) and is gitignored. `evals/` contains bash drivers + Python scorers; `convert_*.py` adapt ground-truth/pose formats and `eval_replica_mapping.py` scores reconstruction. `outputs/` and `logs/` are working dirs (gitignored). Note: when an image directory mixes `frame*.jpg` and `depth*.png`, `run_slam.py` automatically keeps only the RGB frames.
