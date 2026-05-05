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

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = _REPO_ROOT / "config" / "default.yaml"


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_slam_config(
    yaml_path: str | Path = DEFAULT_YAML,
    **overrides: Any,
) -> "SLAMConfig":
    """
    Build a SLAMConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML config file (defaults to config/default.yaml).
        **overrides: Scalar top-level keys to override, e.g.
                     submap_size=12, conf_percentile=75.

    Returns:
        A fully populated SLAMConfig.
    """
    from da3_slam.slam import SLAMConfig
    from da3_slam.keyframe_selector import KeyframeSelectorConfig
    from da3_slam.factor_graph import NoiseConfig
    from da3_slam.loop_closure import LoopClosureConfig

    cfg = _load_yaml(yaml_path)

    # Apply scalar top-level overrides
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    kf = cfg.get("keyframe", {})
    noise = cfg.get("noise", {})
    lc = cfg.get("loop_closure", {})

    return SLAMConfig(
        submap_size=cfg["submap_size"],
        conf_percentile=cfg["conf_percentile"],
        da3_model=cfg["da3_model"],
        da3_process_res=cfg["da3_process_res"],
        enable_loop_closure=lc.get("enable", True),
        keyframe=KeyframeSelectorConfig(
            min_disparity_frac=kf["min_disparity_frac"],
            max_submap_size=cfg["submap_size"],
            max_corners=kf["max_corners"],
            quality_level=kf["quality_level"],
            min_distance=kf["min_distance"],
            lk_win_size=tuple(kf["lk_win_size"]),
            lk_max_level=kf["lk_max_level"],
        ),
        noise=NoiseConfig(
            prior_rot_sigma=noise["prior_rot_sigma"],
            prior_trans_sigma=noise["prior_trans_sigma"],
            between_rot_sigma=noise["between_rot_sigma"],
            between_trans_sigma=noise["between_trans_sigma"],
            loop_rot_sigma=noise["loop_rot_sigma"],
            loop_trans_sigma=noise["loop_trans_sigma"],
        ),
        loop_closure=LoopClosureConfig(
            similarity_threshold=lc["similarity_threshold"],
            min_submaps_apart=lc["min_submaps_apart"],
            dinov2_model=lc["dinov2_model"],
            icp_max_iter=lc["icp_max_iter"],
            icp_tol=lc["icp_tol"],
            icp_max_dist=lc["icp_max_dist"],
            icp_n_points=lc["icp_n_points"],
        ),
    )
