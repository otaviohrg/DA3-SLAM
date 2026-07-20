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
python scripts/show_ablation.py        # render ablation results
python scripts/kf_submap_grid.py       # keyframe-stride × submap-size study (also: make kf-grid)

# Benchmark on EuRoC MAV (ASL format; uses cam0, undistorts by default)
python scripts/benchmark_euroc.py --seq_dir data/EuRoC/vicon_room2/V2_01_easy --out_dir outputs/euroc/V2_01

# Component-level smoke tests (these are plain scripts, not pytest)
python scripts/test_keyframe_selector.py
python scripts/test_submap.py
python scripts/test_loop_closure.py
python scripts/test_pose_graph.py

# Visualization / debugging
python scripts/visualize_trajectory.py
python scripts/visualize_map.py
python scripts/plot_gt_vs_est.py --run_dir outputs/run1   # Sim3-aligned GT-vs-estimate chart

# Evaluation harnesses (bash drivers that run SLAM + score with evo_ape)
evals/eval_tum.sh [submap_size] [model] [use_ray_pose]   # e.g. ./evals/eval_tum.sh 20 nested-giant 0
evals/eval_euroc.sh [submap_size] [model] [use_ray_pose] # EuRoC; converts GT to camera-frame TUM first
evals/eval_replica.sh
evals/reeval.sh                                           # re-score existing output trajectories

# Docker (docker-compose.yml + Makefile wrap all of the above)
make build                                   # build the CUDA image (setup.sh)
make run IMAGE_DIR=data/foo OUT_DIR=out/bar  # offline pipeline in the container
make benchmark-tum SEQ_DIR=data/tum/<seq>    # (also -euroc, -replica); quote globs
make shell                                   # interactive shell in the container

# Live RealSense demo (real-time, with a Rerun viewer)
make viewer                                  # HOST: launch the Rerun viewer (:9876)
make realsense                               # Docker: stream the camera → live map
make run-viz IMAGE_DIR=data/foo OUT_DIR=out/bar  # offline replay streamed to the Rerun viewer
                                             # (viz compose service: host network, no USB;
                                             #  run_slam.py --viewer connect|serve|spawn|none)
```

The **live camera** entry point is `scripts/run_realsense.py` (monocular RGB). It reuses the streaming machinery: `RealSenseFrameSource` (`scripts/realsense_source.py`, lazy `pyrealsense2`) yields `FrameItem`s into `DA3SLAM.run_stream(source, on_update=viewer)`. `on_update` is an optional per-submap callback (added to `run_stream`/`_processing`) carrying a `SLAMUpdate` (current trajectory + the new submap's world points + per-frame camera-space points in `frame_points_cam`); `LiveViewer` (`scripts/live_viewer.py`, lazy `rerun-sdk`) logs it, caches each submap's camera-space points, and **re-projects already-drawn submaps whenever their poses shift** (loop closure / later optimisation) — without that, corrected geometry stays frozen and every closure looks like a duplicated map. `run_realsense.py` also overrides the benchmark-tuned defaults for the live domain (`set_defaults`: `submap_overlap` 2, `boundary_scale_deadband` 0.25, `between_huber_k` 1.345, loop distance 0.6, gap 2, top-3, confidence 0.2, translation gate 3 m, blur gate `sharpness_window` 2 / `min_sharpness_ratio` 0.5) — explicit flags still win. The camera source sets auto-exposure *priority* off (exposure capped at the frame budget → less motion blur at constant fps); `--exposure` switches to manual exposure for aggressive anti-blur. In Docker the Rerun **viewer runs on the host** (`make viewer`) and the container connects out to it at `127.0.0.1:9876` — the `realsense` service uses `network_mode: host`, so no X11 and no gateway/`host.docker.internal` reachability issues. Ctrl-C sets a stop-event so the stream ends cleanly and the final optimisation + save still run. Demo tip: `--selection_mode disparity` (lower latency than the segment default). The `realsense` compose service adds USB passthrough (`privileged` + `/dev/bus/usb`); demo deps live in `requirements-realsense.txt` (installed by `setup.sh`).

There is **no pytest suite and no linter config** (a `Makefile` and `docker-compose.yml` exist for the Docker workflow above; the pipeline itself is still driven by the `scripts/`). The `scripts/test_*.py` files are runnable smoke/integration scripts, not a unit-test framework (`test_pose_graph.py` and `test_keyframe_selector.py` run without a GPU; `test_submap.py` and `test_loop_closure.py` need DA3). They share `scripts/smoke_test_utils.py` (header/check/image helpers). Shared evaluation code is layered: `scripts/benchmark_common.py` is the dataset-agnostic benchmark standard (dataset I/O, ATE/RPE metrics, output schema — copied verbatim across the SLAM/ workspace repos, so edit it in lockstep or not at all); the `benchmark_*.py` drivers combine it with the `scripts/da3_runner.py` adapter (`SharedSLAM` model reuse, standardized run interface); `scripts/tum_eval_common.py` adds `SharedSLAM`/`evaluate_sequence`/`average_metrics` for the TUM study scripts (`ablation_tum.py`, `kf_submap_grid.py`) and re-exports the metrics from `benchmark_common`. Model aliases accepted by the eval scripts: `nested-giant` (default/best), `giant`, `large`, `base`, `small`, or a full HuggingFace ID.

EuRoC support has two entry points sharing the dataset I/O in `scripts/benchmark_common.py` (auto-discovers `mav0` incl. the doubly-nested `<seq>/<seq>/mav0` unpack; loads cam0 frames; converts GT from the IMU/body frame to the camera frame via `T_BS` from `cam0/sensor.yaml`, GT quaternions are `w,x,y,z`; optional radtan undistortion). `scripts/euroc_common.py` re-exports those loaders and adds the TUM-format GT conversion glue (`ensure_euroc_gt_tum`) used by `evals/convert_euroc_gt.py`:
- **`scripts/benchmark_euroc.py`** — in-house ATE/RPE (mirrors `benchmark_tum.py`), undistorts by default (`--no_undistort` to skip), timestamps in seconds.
- **`evals/eval_euroc.sh`** — evo-based harness (mirrors `eval_tum.sh`): runs `run_slam.py` on the raw `mav0/cam0/data` frames, then `evals/convert_euroc_gt.py` writes a camera-frame TUM GT in **nanoseconds** to match run_slam's filename-derived timestamps, and `evo_ape ... -as --t_max_diff 20000000` (20 ms) scores it.


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
_frontend  →(batch_queue)→  _inference  →(submap_queue)→  _processing  ⇄(loop_closure_queue/loop_closure_result_queue)⇄  _loop_closure_worker
```

- **`_frontend`** (`frontend/keyframe_selector.py`): `OnlineKeyframeSelector` uses Lucas-Kanade optical flow; a frame becomes a keyframe when mean tracked displacement exceeds `min_disparity_fraction * W`. Optional blur gating (off by default; live demo on): in segment mode `sharpness_window` swaps each stride-picked keyframe for the sharpest frame (Laplacian variance) within ±window; in disparity mode `min_sharpness_ratio` defers keyframes much blurrier than the recent median (the max_submap_size force still fires, so sustained blur cannot stall the pipeline) — motion blur degrades DA3 poses, retrieval descriptors and repair re-inference all at once. Keyframes are batched into groups of `submap_size`. **Each submap shares its last keyframe as the first frame ("anchor") of the next submap** — this 1-frame overlap is structurally load-bearing (see below). (`KeyframeSelector` is a batch wrapper around the online selector for scripts.)
- **`_inference`** (`backend/inference/`): `SubmapBuilder` runs DA3 (`DepthEstimator` wraps `depth_anything_3.api`) on each keyframe batch, confidence-filters point clouds with **one threshold computed globally across the batch** (`confidence_percentile`), and produces a `Submap` of `Frame`s. Optionally attaches CLIP `semantic_vector`s via `SemanticEmbedder`.
- **`_processing`** (`backend/processing/factor_graph.py`): builds the GTSAM `PoseGraph` incrementally — one SL(4) node per keyframe keyed by `seq_idx`, between-factors for consecutive frames, a tight prior on frame 0. Runs incremental `optimize()` after each submap and a final optimization at the end.
- **`_loop_closure_worker`** (`backend/processing/loop_closure.py`): `LoopClosureDetector.process(submap)` matches per-frame DINO-SALAD descriptors against all previous frames (top-K priority queue, `min_submaps_apart` gap), re-runs DA3 on each matched pair **plus `context_frames` temporal neighbours per side** (multi-view support — bare 2-frame wide-baseline inference is poorly conditioned), gates on mean DA3 confidence over the two matched frames, and returns `LoopClosure` objects carrying `relative_b_to_a` (the measured query→detected cam-to-world transform; the matched frames sit at `query_frame_pos`/`detected_frame_pos` in the re-inference submap). The worker rescales its translation to global metric units and posts a between-factor (`loop=True`) back to `_processing`, which applies a final **geometric sanity gate** (`loop_closure.max_rotation_error_deg` / `max_translation_error`: measurement vs. current graph prediction, generous thresholds) before inserting the factor. Gate failures are **held in a corroboration pool, not dropped** (`_drain_loop_closure_results` + `_HeldLoop`): after an odometry break the graph itself is wrong by exactly the disagreement the gate measures, so when two held closures imply the same correction (translation/rotation consistency, see `_corrections_consistent`) the group is inserted — this is what merges a duplicated map back together. Held closures are re-evaluated against the updated graph on every drain, and one that stays held for `_HELD_ACCEPT_AFTER_DRAINS` drains (≈ submaps) is accepted on its own (uncontradicted; the Huber noise cushions a residual aliased match). Detection-side, `_dedupe_candidates` keeps only the best candidate per detected submap — near-duplicate matches each cost a full re-inference while adding little independent evidence. Separately from retrieval, a detected **boundary break** (scale dead-band fired, or the overlap-2 pose check failed) triggers a **boundary repair**: the worker re-infers a frame pair spanning the broken boundary (`LoopClosureDetector.verify_boundary`) and posts the resulting factor as `trusted` — it bypasses the geometric gate (the frames are known-adjacent; no aliasing risk) and arbitrates the two contradictory boundary factors, rotation included, so segments cannot settle at different angles. Because trusted factors skip the gate, each repair must instead pass `_repair_rotation_deviation`: its rotation is compared against the same relative pose composed through both batches' own chains (via the shared anchor), and a deviation beyond `_REPAIR_MAX_ROTATION_DEV_DEG` (45°) rejects it — low-confidence wide-baseline re-inference can emit ~120° garbage rotations that would corrupt the downstream map orientation. Repairs never fire on benchmarks (they require deadband > 0 or overlap ≥ 2). Loop factors use a **Huber-robustified noise model** (`noise.loop_huber_k`, null = plain Gaussian) so an aliased closure that survives the gates is down-weighted instead of warping the map; `noise.between_huber_k` (null by default, live demo 1.345) does the same for odometry factors — it only acts where the graph has redundancy (overlap≥2 boundary pairs, loop-closure cycles) and concentrates a broken boundary's error at that factor instead of spreading it over the cycle, which is what smears revisited geometry into side-by-side ghost copies.

### Two concepts that are non-obvious and easy to break

1. **Anchor-frame submap bridging (no explicit inter-submap factor).** Consecutive submaps are joined *only* because submap N's last `submap_overlap` frames and submap N+1's first frames are the **same `seq_idx`s** (default overlap 1, VGGT-SLAM style). `_processing` places submap N+1's frames in the global frame by composing the anchor's already-optimized global pose with DA3's local relative poses. `_build_keyframe_poses` dedupes the shared anchors ("first submap wins"). If you change how keyframes are batched in `_frontend`, preserve the overlap or the graph disconnects. With `submap_overlap >= 2` (the live-demo default) the shared frame *pair* is measured by both DA3 batches: `_add_consecutive_frame_factors` deliberately adds the duplicate between-factor (boundary redundancy) and `_check_boundary_consistency` warns when the two measurements disagree — the signature of an odometry break, where a single bad anchor pose used to displace every subsequent submap ("the trajectory jumps and the map rebuilds elsewhere").

2. **Per-submap metric scale accumulation.** Each DA3 batch has its own arbitrary metric scale. `_estimate_boundary_scale` takes the **median depth ratio at the shared anchor frame** between consecutive submaps; this `delta_scale` is multiplied into an `accumulated_scale` that scales the **translation component** of every between-factor (and loop-closure) measurement before it enters the graph (`_scaled_translation`). Loop closures apply an additional depth-scale correction (`_estimate_depth_scale` between the query frame's depth and the loop-closure re-inference's depth) times the query submap's accumulated scale. Rotation is never scaled — only translation. **Note:** `boundary_scale_damping: 1.0` (the shipped default since the UAS/TUM scale sweeps) forces every applied boundary ratio to 1.0 — DA3's metric consistency beat the chained ratios on every dataset tested — so the chain is effectively off and `_processing` skips the median-ratio estimate; set damping < 1 to re-enable chaining. **`boundary_scale_deadband`** (default 0 = off; live demo 0.25) handles the regime the sweeps never saw: live D455 runs showed genuine per-batch metric scale *breaks* (boundary translation-unit ratios of 2–6× with near-zero rotation disagreement — the trajectory suddenly stretches and the map rebuilds elsewhere). With deadband d > 0, ratios within `max(r,1/r)-1 <= d` are forced to 1.0 (the sweep-validated behaviour) but a break-sized ratio is applied in full, damping ignored (see `_boundary_delta_scale`).

## Outputs

`run_slam.py` writes to `--out_dir`: `trajectory_kitti.txt` (12-value rows), `trajectory_tum.txt` (`timestamp tx ty tz qx qy qz qw`; real timestamps inferred from numeric filenames like TUM, else `seq_idx/fps`), `map.ply` (binary, merged colored cloud projected by optimized poses; skip with `--skip_ply`), and `timings.json`. `SLAMResult` (in `slam.py`) owns all the `save_*` methods and `retrieve_best_semantic_frame()` for open-vocab CLIP queries.

## Data & evaluation

`data/` holds datasets (TUM, Replica, custom `video*` dirs) and is gitignored. `evals/` contains bash drivers + Python scorers; `convert_*.py` adapt ground-truth/pose formats and `eval_replica_mapping.py` scores reconstruction. `outputs/` and `logs/` are working dirs (gitignored). Note: when an image directory mixes `frame*.jpg` and `depth*.png`, `run_slam.py` automatically keeps only the RGB frames.
