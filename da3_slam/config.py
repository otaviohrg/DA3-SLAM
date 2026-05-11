"""
Config loading for DA3-SLAM.

The canonical parameter values live in config/default.yaml.
Use load_slam_config() to build a SLAMConfig from a YAML file with
optional keyword overrides for the top-level scalar keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from da3_slam.slam import SLAMConfig
from da3_slam.frontend.keyframe_selector import KeyframeSelectorConfig
from da3_slam.backend.processing.factor_graph import NoiseConfig
from da3_slam.backend.processing.loop_closure import LoopClosureConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = _REPO_ROOT / "config" / "default.yaml"


def load_slam_config(
    yaml_path: str | Path = DEFAULT_YAML,
    **overrides: Any,
) -> "SLAMConfig":
    """
    Build a SLAMConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML config file (defaults to config/default.yaml).
        **overrides: Scalar top-level keys to override, e.g.
                     submap_size=12, confidence_percentile=75.

    Returns:
        A fully populated SLAMConfig.
    """

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Apply scalar top-level overrides
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    keyframe = cfg.get("keyframe", {})
    noise    = cfg.get("noise", {})
    lc       = cfg.get("loop_closure", {})

    return SLAMConfig(
        submap_size=cfg["submap_size"],
        confidence_percentile=cfg["confidence_percentile"],
        depth_model=cfg["depth_model"],
        depth_model_resolution=cfg["depth_model_resolution"],
        enable_loop_closure=lc.get("enable", True),
        keyframe=KeyframeSelectorConfig(
            min_disparity_fraction=keyframe["min_disparity_fraction"],
            max_submap_size=cfg["submap_size"],
            max_corners=keyframe["max_corners"],
            quality_level=keyframe["quality_level"],
            min_distance=keyframe["min_distance"],
            block_size=keyframe["block_size"],
            flow_window_size=tuple(keyframe["flow_window_size"]),
            flow_pyramid_levels=keyframe["flow_pyramid_levels"],
            flow_stop_criteria=tuple(keyframe["flow_stop_criteria"]),
        ),
        noise=NoiseConfig(
            prior_rotation_sigma=noise["prior_rotation_sigma"],
            prior_translation_sigma=noise["prior_translation_sigma"],
            between_rotation_sigma=noise["between_rotation_sigma"],
            between_translation_sigma=noise["between_translation_sigma"],
            between_scale_sigma=noise["between_scale_sigma"],
            loop_rotation_sigma=noise["loop_rotation_sigma"],
            loop_translation_sigma=noise["loop_translation_sigma"],
            loop_scale_sigma=noise["loop_scale_sigma"],
        ),
        loop_closure=LoopClosureConfig(
            similarity_threshold=lc["similarity_threshold"],
            min_submaps_apart=lc["min_submaps_apart"],
            dinov2_model=lc["dinov2_model"],
            icp_max_iterations=lc["icp_max_iterations"],
            icp_tolerance=lc["icp_tolerance"],
            icp_max_distance=lc["icp_max_distance"],
            icp_num_points=lc["icp_num_points"],
            icp_max_rmse=lc["icp_max_rmse"],
            icp_max_scale_deviation=lc["icp_max_scale_deviation"],
            n_descriptor_frames=lc["n_descriptor_frames"],
        ),
    )
