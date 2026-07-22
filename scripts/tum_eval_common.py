"""
Shared TUM RGB-D evaluation utilities.

Used by ablation_tum.py, kf_submap_grid.py and da3_runner.py:
  - SharedSLAM: reuses one loaded DA3 model across many configurations
  - evaluate_sequence(): run SLAM on one TUM sequence and score it against GT
  - average_metrics(): cross-sequence aggregation of evaluate_sequence dicts

Dataset parsing, timestamp association, and the SE(3)/Sim(3) ATE / RPE
metrics live in benchmark_common.py (the dataset-agnostic benchmark standard
shared across the SLAM/ workspace) and are re-exported here for convenience.

These scripts are run from the repo root as `python scripts/<name>.py`, so
Python puts the scripts/ directory on sys.path and `import tum_eval_common`
resolves without any path manipulation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmark_common import (
    associate,
    compute_ate,
    compute_rpe,
    load_groundtruth,
    load_rgb_list,
    parse_tum_file,
    se3_align,
    sim3_align,
)

__all__ = [
    "parse_tum_file", "load_rgb_list", "load_groundtruth", "associate",
    "se3_align", "sim3_align", "compute_ate", "compute_rpe",
    "SharedSLAM", "evaluate_sequence", "average_metrics", "TIMING_MODULES",
]


# ── shared SLAM wrapper ───────────────────────────────────────────────────────

class SharedSLAM:
    """
    Loads the DA3 depth model once and reuses it across many configurations.
    Only the cheap components (SubmapBuilder, LoopClosureDetector) are rebuilt
    when the configuration changes.
    """

    def __init__(self, base_config) -> None:
        from da3_slam.slam import DA3SLAM
        self._slam = DA3SLAM(base_config)

    @property
    def config(self):
        """The live SLAMConfig (resolved backbone/resolution, mutable)."""
        return self._slam.config

    def set_keyframe_io(self, keyframes_from=None, dump_keyframes=None) -> None:
        """Point the next run's frozen-keyframe replay/dump at these paths.

        Set per sequence by the benchmark loop so each sequence gets its own
        keyframe list (the model, and hence this config, is reused across
        sequences).  None disables the corresponding side.
        """
        self._slam.config.keyframes_from = keyframes_from
        self._slam.config.dump_keyframes = dump_keyframes

    def set_resolution(self, resolution: int) -> None:
        """Change the DA3 processing resolution without reloading the model.

        Resolution is just the ``process_res`` argument to DA3's forward, held
        on the DepthEstimator — which ``reconfigure`` does NOT rebuild — so a
        resolution sweep reuses one loaded model across all resolutions of a
        given size.  (Model *size* is a different checkpoint and still needs a
        fresh SharedSLAM.)
        """
        self._slam.estimator.process_resolution = int(resolution)
        self._slam.config.depth_model_resolution = int(resolution)

    def set_build_pointclouds(self, flag: bool) -> None:
        """Enable/disable per-frame point clouds (needed for the map-detail
        proxy).  Rebuilds the SubmapBuilder, which fixes the flag at
        construction; the loaded model is untouched."""
        from da3_slam.backend.inference.submap import SubmapBuilder
        self._slam.config.build_pointclouds = bool(flag)
        self._slam.builder = SubmapBuilder(
            self._slam.estimator,
            confidence_percentile=self._slam.config.confidence_percentile,
            build_pointclouds=bool(flag))
        # The detector shares the builder; point it at the new one rather than
        # rebuilding (a rebuild would reload DINO-SALAD via torch.hub).
        if self._slam.detector is not None:
            self._slam.detector.builder = self._slam.builder
        elif self._slam.config.enable_loop_closure:
            self._slam.detector = self._make_detector()

    def reconfigure(self, config) -> None:
        """Swap in a new config without reloading the depth model."""
        from da3_slam.backend.inference.submap import SubmapBuilder

        self._slam.config = config
        self._slam.builder = SubmapBuilder(
            self._slam.estimator,
            confidence_percentile=config.confidence_percentile,
            build_pointclouds=config.build_pointclouds,
        )
        self._slam.detector = self._make_detector()

    def run(self, image_paths: list[str], on_update=None, on_loop_closure=None):
        """Run SLAM, clearing the loop-closure detector's per-sequence state
        between sequences so descriptors from a previous sequence don't produce
        cross-sequence loop closures with stale indices.  `on_update` /
        `on_loop_closure` are forwarded to DA3SLAM.run (per-submap live-viewer
        hook and pre-optimisation loop-closure hook; see SLAMUpdate and
        _RunContext.on_loop_closure).

        The detector is *reset*, not rebuilt: rebuilding reloads DINO-SALAD,
        which re-validates DINOv2 against GitHub via torch.hub and can kill a
        long sweep on a transient 504.  It is built once, on first use.
        """
        if self._slam.detector is not None:
            self._slam.detector.reset()
        elif self._slam.config.enable_loop_closure:
            self._slam.detector = self._make_detector()
        return self._slam.run(image_paths, on_update=on_update,
                              on_loop_closure=on_loop_closure)

    def _make_detector(self):
        from da3_slam.backend.processing.loop_closure import LoopClosureDetector
        config = self._slam.config
        if not config.enable_loop_closure:
            return None
        return LoopClosureDetector(config.loop_closure, builder=self._slam.builder)


# ── per-sequence evaluation ───────────────────────────────────────────────────

def evaluate_sequence(
    run_slam: Callable[[list[str]], Any],
    seq_dir: Path,
    out_dir: Path,
    max_frames: int | None,
) -> dict | None:
    """
    Run SLAM on one TUM sequence and score it against ground truth.

    Args:
        run_slam:   callable mapping image paths → SLAMResult
                    (e.g. SharedSLAM.run or DA3SLAM.run)
        seq_dir:    TUM sequence directory (must contain rgb.txt + groundtruth.txt)
        out_dir:    where to write trajectory_est.txt
        max_frames: optional frame cap

    Returns:
        compact metrics dict, or None if the sequence could not be evaluated.
    """
    seq_name = seq_dir.name
    rgb_txt = seq_dir / "rgb.txt"
    gt_txt = seq_dir / "groundtruth.txt"
    if not rgb_txt.exists() or not gt_txt.exists():
        print(f"  [SKIP] {seq_name}: missing rgb.txt or groundtruth.txt")
        return None

    image_paths, timestamps = load_rgb_list(seq_dir, max_frames)

    start_time = time.time()
    result = run_slam(image_paths)
    wall_seconds = time.time() - start_time

    ts_map = {i: ts for i, ts in enumerate(timestamps)}
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_tum(str(out_dir / "trajectory_est.txt"), timestamps=ts_map)

    est_ts_to_pose = {
        ts_map[seq_idx]: pose
        for seq_idx, pose in result.keyframe_poses.items()
        if seq_idx in ts_map
    }

    gt_all = load_groundtruth(gt_txt)
    gt_stamps = [entry[0] for entry in gt_all]
    est_stamps = sorted(est_ts_to_pose.keys())
    pairs = associate(est_stamps, gt_stamps, max_diff=0.02)

    if len(pairs) < 3:
        print(f"  [SKIP] {seq_name}: too few matched poses ({len(pairs)})")
        return None

    gt_matched = [gt_all[ib][1] for _, ib in pairs]
    est_matched = [est_ts_to_pose[est_stamps[ia]] for ia, _ in pairs]

    ate_se3 = compute_ate(gt_matched, est_matched, align="se3")
    ate_sim3 = compute_ate(gt_matched, est_matched, align="sim3")
    rpe_1 = compute_rpe(gt_matched, est_matched, delta=1)

    n_submaps = len([s for s in result.submaps if not s.is_loop_closure_submap])

    return {
        "sequence": seq_name,
        "ate_se3_rmse": ate_se3["rmse"],
        "ate_sim3_rmse": ate_sim3["rmse"],
        "rpe_trans_rmse": rpe_1["trans_rmse"],
        "rpe_rot_rmse_deg": rpe_1["rot_rmse_deg"],
        "n_frames": len(image_paths),
        "n_keyframes": result.n_keyframes,
        "n_submaps": n_submaps,
        "n_loop_closures": len(result.loop_closures),
        "wall_seconds": round(wall_seconds, 1),
        "timings": result.timings,
    }


# ── cross-sequence aggregation ────────────────────────────────────────────────

TIMING_MODULES = ("keyframe_selection", "submap_building",
                  "graph_building", "loop_closure", "optimization")


def average_metrics(seq_metrics: list[dict]) -> dict:
    """Cross-sequence averages of the dicts produced by evaluate_sequence().

    Timing ratios (s/frame, s/keyframe, s/submap) are computed per sequence
    and then averaged, so longer sequences don't dominate the mean.
    """
    if not seq_metrics:
        return {}

    def mean(key: str) -> float:
        return float(np.mean([m[key] for m in seq_metrics]))

    module_ms_per_frame = {
        module: float(np.mean([
            m["timings"].get(module, 0.0) / max(m["n_frames"], 1) * 1000
            for m in seq_metrics
        ]))
        for module in TIMING_MODULES
    }

    return {
        "ate_se3_rmse": mean("ate_se3_rmse"),
        "ate_sim3_rmse": mean("ate_sim3_rmse"),
        "rpe_trans_rmse": mean("rpe_trans_rmse"),
        "rpe_rot_rmse_deg": mean("rpe_rot_rmse_deg"),
        "n_keyframes": mean("n_keyframes"),
        "n_submaps": mean("n_submaps"),
        "n_loop_closures": mean("n_loop_closures"),
        "wall_seconds": mean("wall_seconds"),
        "s_per_frame": float(np.mean(
            [m["wall_seconds"] / m["n_frames"] for m in seq_metrics]
        )),
        "s_per_kf": float(np.mean(
            [m["wall_seconds"] / m["n_keyframes"]
             for m in seq_metrics if m["n_keyframes"] > 0]
        )),
        "s_per_submap": float(np.mean(
            [m["wall_seconds"] / max(m["n_submaps"], 1) for m in seq_metrics]
        )),
        "module_ms_per_frame": module_ms_per_frame,
    }
