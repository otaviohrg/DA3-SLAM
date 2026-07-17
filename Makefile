# DA3-SLAM — Docker & pipeline helpers.
# Run `make help` for the list of targets.

# Compose invocation (`docker compose` v2; override with `make COMPOSE="docker-compose" ...`).
COMPOSE ?= docker compose
SERVICE ?= da3-slam
RUN      = $(COMPOSE) run --rm $(SERVICE)

# Common run_slam.py args (override on the CLI, e.g. `make run IMAGE_DIR=data/foo OUT_DIR=outputs/bar`).
# Paths are relative to the repo root (bind-mounted to /app in the container).
IMAGE_DIR ?= data/video1_30fps
OUT_DIR   ?= outputs/run1
# Dataset sequence dir(s) for the benchmark targets, e.g.
#   make benchmark-tum SEQ_DIR=data/tum/rgbd_dataset_freiburg1_xyz
SEQ_DIR   ?=
ARGS      ?=

# Collapse any embedded newlines/extra whitespace in the user-supplied values.
# A newline pasted inside SEQ_DIR="..." would otherwise split the recipe into
# two shell commands (make executes each recipe line separately), producing
# baffling "Permission denied" errors on the second half.
override SEQ_DIR := $(strip $(SEQ_DIR))
override ARGS    := $(strip $(ARGS))

# Expand SEQ_DIR to a --seq_dir flag only when it is set.
seq_flag = $(if $(SEQ_DIR),--seq_dir $(SEQ_DIR),)

.DEFAULT_GOAL := help

## -- Images ---------------------------------------------------------------

.PHONY: build
build: ## Build the CUDA image (runs setup.sh: torch cu128, DA3, SALAD, gtsam)
	$(COMPOSE) build

.PHONY: rebuild
rebuild: ## Build the image from scratch, ignoring the cache
	$(COMPOSE) build --no-cache

## -- Running --------------------------------------------------------------

.PHONY: shell
shell: ## Open an interactive bash shell in the container
	$(RUN) bash

.PHONY: run
run: ## Run the full pipeline (IMAGE_DIR, OUT_DIR, ARGS overridable)
	$(RUN) python3 scripts/run_slam.py --image_dir $(IMAGE_DIR) --out_dir $(OUT_DIR) $(ARGS)

.PHONY: run-viz
run-viz: ## Run the pipeline with a live Rerun view (start `make viewer` on the host first)
	$(COMPOSE) run --rm viz \
		python3 scripts/run_slam.py --image_dir $(IMAGE_DIR) --out_dir $(OUT_DIR) \
			--viewer connect $(ARGS)

.PHONY: py
py: ## Drop into a python3 REPL in the container
	$(RUN) python3

## -- Live RealSense demo --------------------------------------------------

# Memory budget for the Rerun viewer; it drops the oldest data past this
# instead of OOM-crashing on a long live stream. Override: make viewer VIEWER_MEM=8GB
VIEWER_MEM ?= 4GB

.PHONY: viewer
viewer: ## Launch the Rerun viewer on the HOST (pip install rerun-sdk first)
	rerun --memory-limit $(VIEWER_MEM)

.PHONY: realsense
realsense: ## Run the live RealSense demo in Docker (start `make viewer` on the host first)
	$(COMPOSE) run --rm realsense \
		python3 scripts/run_realsense.py --selection_mode disparity $(ARGS)

## -- Benchmarks & ablations ----------------------------------------------

.PHONY: benchmark-tum
benchmark-tum: ## Run the TUM benchmark (set SEQ_DIR=data/tum/<seq>)
	$(RUN) python3 scripts/benchmark_tum.py $(seq_flag) $(ARGS)

.PHONY: benchmark-uas
benchmark-uas: ## Run the UAS benchmark (set SEQ_DIR=data/UAS/<seq>)
	$(RUN) python3 scripts/benchmark_uas.py $(seq_flag) $(ARGS)

.PHONY: download-tum
download-tum: ## Download the TUM freiburg1 benchmark sequences into data/tum/
	$(RUN) python3 scripts/download_tum.py $(ARGS)

.PHONY: benchmark-euroc
benchmark-euroc: ## Run the EuRoC benchmark (set SEQ_DIR=data/EuRoC/<seq>)
	$(RUN) python3 scripts/benchmark_euroc.py $(seq_flag) $(ARGS)

.PHONY: benchmark-replica
benchmark-replica: ## Run the Replica benchmark (set SEQ_DIR=data/Replica/<scene>)
	$(RUN) python3 scripts/benchmark_replica.py $(if $(SEQ_DIR),--scene_dir $(SEQ_DIR),) $(ARGS)

.PHONY: ablation-tum
ablation-tum: ## Run the TUM ablation study
	$(RUN) python3 scripts/ablation_tum.py $(ARGS)

.PHONY: kf-grid
kf-grid: ## Keyframe-stride × submap-size grid study (set SEQ_DIR=...)
	$(RUN) python3 scripts/kf_submap_grid.py $(seq_flag) $(ARGS)

## -- Charts & tables --------------------------------------------------------

.PHONY: show-ablation
show-ablation: ## Print ablation tables from a saved ablation_results.json (ARGS=--results ...)
	$(RUN) python3 scripts/show_ablation.py $(ARGS)

.PHONY: plot-gt
plot-gt: ## GT-vs-estimate chart for a run dir (set OUT_DIR=...; Sim3-aligned)
	$(RUN) python3 scripts/plot_gt_vs_est.py --run_dir $(OUT_DIR) $(ARGS)

.PHONY: plot-trajectory
plot-trajectory: ## Trajectory PNGs from a run's trajectory_tum.txt (set OUT_DIR=...)
	$(RUN) python3 scripts/visualize_trajectory.py --tum $(OUT_DIR)/trajectory_tum.txt $(ARGS)

.PHONY: plot-map
plot-map: ## Multi-view PNG renders of a run's map.ply (set OUT_DIR=...)
	$(RUN) python3 scripts/visualize_map.py $(OUT_DIR)/map.ply $(ARGS)

## -- Smoke tests ----------------------------------------------------------

.PHONY: smoke
smoke: ## Run the GPU-free smoke tests (keyframe selector + pose graph)
	$(RUN) python3 scripts/test_pose_graph.py
	$(RUN) python3 scripts/test_keyframe_selector.py

.PHONY: smoke-all
smoke-all: ## Run all component smoke tests (needs GPU/DA3)
	$(RUN) python3 scripts/test_keyframe_selector.py
	$(RUN) python3 scripts/test_submap.py
	$(RUN) python3 scripts/test_loop_closure.py
	$(RUN) python3 scripts/test_pose_graph.py

## -- Housekeeping ---------------------------------------------------------

.PHONY: down
down: ## Stop and remove containers
	$(COMPOSE) down

.PHONY: clean
clean: ## Remove containers (keeps the model-cache volume)
	$(COMPOSE) down --remove-orphans

.PHONY: clean-cache
clean-cache: ## Remove containers AND the downloaded-model cache volume
	$(COMPOSE) down --remove-orphans --volumes

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
