"""
Append-only JSONL results log for the compute-reduction sweep (plan Step 0c).

One line = one run = one (backbone_size, resolution, repeat, sequence) config
with its ATE, per-stage latency, backbone-only latency and peak GPU memory.
JSONL (not a single JSON array) so runs can be appended incrementally across
many invocations and read back with a one-liner:

    import json; rows = [json.loads(l) for l in open(path)]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# DA3's DINOv2 encoder patch size.  The encoder runs per frame and its cost
# grows ~quadratically with the token-grid side (resolution / patch), so
# (resolution / patch)^2 is the hardware-independent compute proxy the plan
# asks for (a cheaper stand-in for a one-shot fvcore FLOP count).
DA3_PATCH_SIZE = 14


def token_proxy(resolution: int, patch_size: int = DA3_PATCH_SIZE) -> int:
    """Analytic per-frame encoder token count: (resolution / patch)^2."""
    side = max(1, int(resolution) // int(patch_size))
    return side * side


def append_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON object as a line to `path` (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def build_row(config, args, metrics: dict) -> dict[str, Any]:
    """Flatten a benchmark metrics dict + run config into one sweep row.

    `config` is the resolved SLAMConfig (so the backbone/resolution are the
    values actually used, not the possibly-None CLI args); `metrics` is the
    dict returned by benchmark_common.evaluate_trajectory (its `timings` has
    already been enriched with the backbone-only figures by run_da3).
    """
    timings = metrics.get("timings", {})
    ate_sim3 = metrics.get("ate_sim3", {})
    ate_se3 = metrics.get("ate_se3", {})
    rpe = metrics.get("rpe_delta1", {})

    # Cross-view token merging (off unless the merging sweep turned it on).
    merging = getattr(config, "token_merging", None)
    merge_on = bool(merging is not None and merging.enable)

    # At km scale a metre of ATE means something very different on a 200 m
    # sequence than on a 2 km one, so UAS is reported as % of the ground-truth
    # path length actually evaluated (the matched GT poses).  Kept as a field
    # rather than a separate metric so every row carries both.
    path_len = metrics.get("gt_path_length_m")

    def _pct(value):
        if value is None or not path_len:
            return None
        return 100.0 * value / path_len

    return {
        "system": metrics.get("system"),
        "dataset": metrics.get("dataset"),
        "sequence": metrics.get("sequence"),
        "status": metrics.get("status", "ok"),
        # config identity
        "backbone_size": config.depth_model,
        "resolution": config.depth_model_resolution,
        "backbone_dtype": getattr(config, "backbone_dtype", "fp32"),
        "repeat": getattr(args, "repeat", 0),
        "submap_size": config.submap_size,
        "loop_closure": config.enable_loop_closure,
        # cross-view token merging
        "merging": merge_on,
        "merge_start": merging.start if merge_on else None,
        "merge_ratio": merging.merge_ratio if merge_on else None,
        "merge_min_frames": merging.min_frames if merge_on else None,
        # realised (measured, not requested) merge behaviour
        "merge_token_ratio": metrics.get("merge_token_ratio"),
        "merge_calls": metrics.get("merge_calls"),
        "merge_calls_passthrough": metrics.get("merge_calls_passthrough"),
        # one-shot compute proxy (config-level; see token_proxy)
        "tokens_per_frame": token_proxy(config.depth_model_resolution),
        # accuracy (Sim3 is the monocular headline)
        "ate_sim3_rmse": ate_sim3.get("rmse"),
        "ate_se3_rmse": ate_se3.get("rmse"),
        "rpe_trans_rmse": rpe.get("trans_rmse"),
        "rpe_rot_rmse_deg": rpe.get("rot_rmse_deg"),
        # scale-normalised accuracy (the honest metric at km scale — UAS)
        "gt_path_length_m": path_len,
        "ate_sim3_pct": _pct(ate_sim3.get("rmse")),
        "ate_se3_pct": _pct(ate_se3.get("rmse")),
        # counts
        "n_frames": metrics.get("n_frames"),
        "n_keyframes": metrics.get("n_keyframes"),
        "n_submaps": metrics.get("n_submaps"),
        "n_loop_closures": metrics.get("n_loop_closures"),
        # latency (s) + backbone-only figures + peak memory
        "total_s": timings.get("total_s"),
        "fps": timings.get("fps"),
        "backbone_forward_s": timings.get("backbone_forward_s"),
        "backbone_calls": timings.get("backbone_calls"),
        "peak_gpu_mem_mb": timings.get("peak_gpu_mem_mb"),
        "keyframe_selection_s": timings.get("keyframe_selection"),
        "submap_building_s": timings.get("submap_building"),
        "loop_closure_s": timings.get("loop_closure"),
        "graph_building_s": timings.get("graph_building"),
        "optimization_s": timings.get("optimization"),
    }
