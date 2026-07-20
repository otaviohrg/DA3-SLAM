# DA3-SLAM

Monocular RGB SLAM built on top of **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3)**.

Batches of keyframes are fed through DA3 to get per-frame metric depth, confidence, and camera poses; the batches are assembled into local **submaps**, stitched into a global **SL(4) pose graph** (GTSAM), and loops are closed via DINO-SALAD descriptor retrieval followed by DA3 re-inference on the matched image pair. The architecture mirrors **VGGT-SLAM** (anchor-frame submap overlap — default 1, ≥ 2 for boundary redundancy on live runs — SL(4) factor graph, loop-closure wiring). Monocular scale is arbitrary, so the **Sim3-aligned ATE** is the headline accuracy metric throughout.

## Requirements

- Python ≥ 3.11, CUDA 12.8, PyTorch 2.10 — a GPU is required (DA3 + DINO-SALAD inference)
- GTSAM built **with SL4 support** (`gtsam.SL4`, `BetweenFactorSL4`); the stock PyPI wheel may lack these symbols
- Two dependencies are not on PyPI and are installed from source by `setup.sh`:
  [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) and [SALAD](https://github.com/Dominic101/salad)
- The live camera demo additionally needs `requirements-realsense.txt` (pyrealsense2 + rerun-sdk; baked into the image by `setup.sh`)

The `Dockerfile` builds the full CUDA environment and runs `setup.sh` (which assumes the repo is mounted at `/app`). `uv` is the package manager.

## Quick start

```bash
# Run the full pipeline on a directory of images
python scripts/run_slam.py --image_dir data/video1_30fps --out_dir outputs/run1

# Common overrides (defaults come from config/default.yaml)
python scripts/run_slam.py --image_dir DIR --submap_size 12 \
    --confidence_percentile 75 --no_loop_closure --max_frames 200 --skip_ply

# Or via Docker (docker-compose.yml + Makefile wrap everything)
make build                                   # build the CUDA image
make run IMAGE_DIR=data/foo OUT_DIR=out/bar  # offline pipeline in the container
make shell                                   # interactive shell in the container
```

Outputs in `--out_dir`: `trajectory_kitti.txt`, `trajectory_tum.txt`, `map.ply` (merged colored point cloud), and `timings.json`.

## Live RealSense demo & real-time replay (Rerun viewer)

```bash
make viewer                                      # HOST: launch the Rerun viewer (:9876)
make realsense                                   # Docker: stream the camera → live map
make run-viz IMAGE_DIR=data/foo OUT_DIR=out/bar  # offline replay streamed to the viewer
```

The live entry point is `scripts/run_realsense.py` (monocular RGB; the sensor's own
depth is ignored — DA3 predicts it). It ships live-tuned defaults (submap overlap 2,
scale-break dead-band, Huber odometry noise, looser loop detection, blur gating);
explicit flags still override them. Ctrl-C ends the stream cleanly and the final
optimisation + trajectory/map save still run.

## Repository layout

```
da3_slam/
  config.py                      all config dataclasses + YAML loading (imports without GPU stack)
  slam.py                        DA3SLAM pipeline runner (4 threads) + SLAMResult export
  frontend/keyframe_selector.py  optical-flow keyframe selection (disparity + segment modes, blur gating)
  backend/inference/             DA3 wrapper, Submap/Frame construction, CLIP embeddings
  backend/processing/            SL(4) GTSAM pose graph, loop closure detection/verification
config/default.yaml              authoritative parameter defaults
scripts/
  run_slam.py, run_realsense.py  offline / live entry points
  benchmark_common.py            shared benchmark standard: dataset I/O + ATE/RPE metrics
                                 (copied verbatim across the SLAM workspace — edit in lockstep or not at all)
  da3_runner.py                  DA3-SLAM adapter: CLI knobs, model reuse, shared per-sequence driver
  benchmark_{tum,euroc,replica,uas}.py   thin per-dataset drivers on top of the two above
  ablation_tum.py, kf_submap_grid.py, show_ablation.py   parameter studies
  live_viewer.py, trajectory_snapshots.py, visualize_*.py, plot_gt_vs_est.py   viewers & plots
  test_*.py                      runnable smoke tests (no pytest suite)
evals/                           bash eval drivers (ATE/RPE via evo) + Replica mapping scorer
```

## Evaluation

```bash
# Fetch the TUM freiburg1 sequences used in the benchmark
python scripts/download_tum.py

# Per-dataset benchmarks (shared scoring + output schema; Sim3 ATE is the headline)
python scripts/benchmark_tum.py     --seq_dir data/tum/rgbd_dataset_freiburg1_xyz --out_dir outputs/benchmark
python scripts/benchmark_euroc.py   --seq_dir data/EuRoC/vicon_room2/V2_01_easy   --out_dir outputs/euroc      # cam0, undistorts by default
python scripts/benchmark_replica.py --scene_dir data/Replica/office0              --out_dir outputs/replica
python scripts/benchmark_uas.py     --seq_dir data/UAS/fyllingsdalen_tunnel       --out_dir outputs/uas        # ROS 1 bags; pip install rosbags

# Parameter studies (model loaded once, results checkpointed, --resume supported)
python scripts/ablation_tum.py   --seq_dir ... --out_dir outputs/ablation
python scripts/kf_submap_grid.py --seq_dir ... --out_dir outputs/kf_grid   # keyframe-stride × submap-size study

# Bash harnesses (run SLAM + score with evo_ape)
evals/eval_tum.sh   [submap_size] [model] [use_ray_pose]
evals/eval_euroc.sh [submap_size] [model] [use_ray_pose]   # converts EuRoC GT to camera-frame TUM first
evals/eval_replica.sh
```

Model aliases accepted by the eval scripts: `nested-giant` (default/best), `giant`, `large`, `base`, `small`, or a full HuggingFace ID.

## Smoke tests

There is no pytest suite; `scripts/test_*.py` are runnable check scripts. `test_pose_graph.py` (synthetic, needs only GTSAM) and `test_keyframe_selector.py` (needs only OpenCV) run without a GPU; the rest exercise DA3 on real images.

See `CLAUDE.md` for a deeper architecture walkthrough, including the two structural invariants (anchor-frame submap bridging and per-submap metric scale accumulation) that are easy to break.
