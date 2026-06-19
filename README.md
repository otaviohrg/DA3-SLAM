# DA3-SLAM

Monocular RGB SLAM built on top of **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3)**.

Batches of keyframes are fed through DA3 to get per-frame metric depth, confidence, and camera poses; the batches are assembled into local **submaps**, stitched into a global **SL(4) pose graph** (GTSAM), and loops are closed via DINO-SALAD descriptor retrieval followed by DA3 re-inference on the matched image pair. The architecture mirrors **VGGT-SLAM** (1-frame submap overlap, SL(4) factor graph, loop-closure wiring).

## Requirements

- Python ≥ 3.11, CUDA 12.8, PyTorch 2.10 — a GPU is required (DA3 + DINO-SALAD inference)
- GTSAM built **with SL4 support** (`gtsam.SL4`, `BetweenFactorSL4`); the stock PyPI wheel may lack these symbols
- Two dependencies are not on PyPI and are installed from source by `setup.sh`:
  [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) and [SALAD](https://github.com/Dominic101/salad)

The `Dockerfile` builds the full CUDA environment and runs `setup.sh` (which assumes the repo is mounted at `/app`). `uv` is the package manager.

## Quick start

```bash
# Run the full pipeline on a directory of images
python scripts/run_slam.py --image_dir data/video1_30fps --out_dir outputs/run1

# Common overrides (defaults come from config/default.yaml)
python scripts/run_slam.py --image_dir DIR --submap_size 12 \
    --confidence_percentile 75 --no_loop_closure --max_frames 200 --skip_ply
```

Outputs in `--out_dir`: `trajectory_kitti.txt`, `trajectory_tum.txt`, `map.ply` (merged colored point cloud), and `timings.json`.

## Real-time ROS 2 (rosbag / live topics)

DA3-SLAM runs in a container without ROS, while ROS 2 + the data live on the host.
They are bridged over a **bind-mounted spool directory** — no ROS inside the container,
no networking. The host bridge subscribes to the camera + ground-truth topics, applies a
**drop-oldest** real-time policy, and writes frames + a TUM ground-truth log into the spool;
the container runner tails the spool and feeds frames into `DA3SLAM.run_stream()`.

```bash
# ── host (system python3.12, ROS 2 jazzy) ──────────────────────────────────
source /opt/ros/jazzy/setup.bash
# (one-time, if the bag has no metadata.yaml) ros2 bag reindex <bag_dir> -s sqlite3
python3 scripts/ros_spool_bridge.py --spool_dir /tmp/da3_spool --clean \
    --image_topic /camera_frames_0 --odom_topic /ground_truth/odom
# in a second host terminal — replay the bag (or just run live sensors):
ros2 bag play <bag_dir>

# ── container (the SLAM env) — mount the same spool dir ─────────────────────
#   docker run ... -v /tmp/da3_spool:/spool ...
python scripts/run_slam_ros.py --spool_dir /spool --out_dir outputs/ros_run
```

The runner stops on the bridge's `DONE` sentinel (bag finished / idle / Ctrl-C) and writes
the usual trajectory/map outputs plus a copy of `ground_truth_tum.txt` for `evo_ape ... -as`.
Use `--process_all` to disable drop-oldest (process every frame; not real-time but reproducible).

## Repository layout

```
da3_slam/
  config.py                      all config dataclasses + YAML loading (imports without GPU stack)
  slam.py                        DA3SLAM pipeline runner (4 threads) + SLAMResult export
  frontend/keyframe_selector.py  optical-flow keyframe selection
  backend/inference/             DA3 wrapper, Submap/Frame construction, CLIP embeddings
  backend/processing/            SL(4) GTSAM pose graph, loop closure, alignment diagnostics
config/default.yaml              authoritative parameter defaults
scripts/                         runner, smoke tests, TUM benchmark/ablation/grid-search, visualisation
evals/                           bash eval drivers (ATE/RPE via evo) + Replica mapping scorer
```

## Evaluation

```bash
# TUM RGB-D benchmark / parameter studies
python scripts/benchmark_tum.py --seq_dir data/tum/rgbd_dataset_freiburg1_xyz --out_dir outputs/benchmark
python scripts/ablation_tum.py  --seq_dir ... --out_dir outputs/ablation
python scripts/grid_search_tum.py --seq_dir ... --out_dir outputs/grid_search

# EuRoC MAV benchmark (ASL format; uses cam0, undistorts by default)
python scripts/benchmark_euroc.py --seq_dir data/EuRoC/vicon_room2/V2_01_easy --out_dir outputs/euroc/V2_01

# Bash harnesses (run SLAM + score with evo_ape)
evals/eval_tum.sh   [submap_size] [model] [use_ray_pose]
evals/eval_euroc.sh [submap_size] [model] [use_ray_pose]   # converts EuRoC GT to camera-frame TUM first
evals/eval_replica.sh
```

## Smoke tests

There is no pytest suite; `scripts/test_*.py` are runnable check scripts. `test_pose_graph.py` (synthetic, needs only GTSAM) and `test_keyframe_selector.py` (needs only OpenCV) run without a GPU; the rest exercise DA3 on real images.

See `CLAUDE.md` for a deeper architecture walkthrough, including the two structural invariants (anchor-frame submap bridging and per-submap metric scale accumulation) that are easy to break.
