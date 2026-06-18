"""
Configuration types and YAML loading for DA3-SLAM.

The canonical parameter values live in config/default.yaml.  Use
load_slam_config() to build a SLAMConfig from a YAML file, with optional
keyword overrides for the top-level scalar keys.

All configuration dataclasses are defined here (except
KeyframeSelectorConfig, which lives next to the keyframe selector) so that
loading and inspecting configuration never requires the heavy GPU stack
(torch, gtsam, Depth Anything 3).  The component modules re-export their
own config class for convenience, e.g.
``from da3_slam.backend.processing.factor_graph import NoiseConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from da3_slam.frontend.keyframe_selector import KeyframeSelectorConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = _REPO_ROOT / "config" / "default.yaml"


@dataclass
class NoiseConfig:
    """Isotropic noise sigmas for the SL(4) pose graph (see factor_graph.py).

    SL(4) noise is 15-dimensional (dim of the sl(4) Lie algebra); each sigma
    below is applied to all 15 dimensions.
    Canonical values: config/default.yaml → noise.*
    """

    # Prior on the first frame — very tight so the map is anchored at the origin
    prior_sigma: float

    # Between-factor noise for consecutive frames within a submap
    between_sigma: float

    # Between-factor noise for loop closure constraints
    loop_sigma: float


@dataclass
class LoopClosureConfig:
    """Tuning knobs for loop closure detection (see loop_closure.py).

    Canonical values: config/default.yaml → loop_closure.*
    """

    # L2 distance threshold between DINO-SALAD descriptors for a frame pair
    # to become a loop candidate.  Lower distance = more similar, so
    # *lowering* this threshold makes detection stricter.
    distance_threshold: float

    # Minimum submap index gap between the query and any candidate
    min_submaps_apart: int

    # Maximum loop closures accepted per submap (priority queue capacity)
    max_loop_closures: int

    # Minimum mean DA3 depth confidence [0, 1] required to accept a closure.
    # Analogous to VGGT-SLAM's image_match_ratio >= 0.85 gate.
    min_confidence_ratio: float


@dataclass
class SLAMConfig:
    """Top-level configuration for the full DA3-SLAM pipeline.

    Canonical values: config/default.yaml.
    Use load_slam_config() to construct from YAML.
    """

    keyframe: KeyframeSelectorConfig
    noise: NoiseConfig
    loop_closure: LoopClosureConfig

    # Frames per submap (including the 1-frame anchor overlap)
    submap_size: int

    # Global confidence percentile threshold for point cloud filtering
    confidence_percentile: float

    # DA3 model
    depth_model: str
    depth_model_resolution: int

    # Enable loop closure (can disable for speed during debugging)
    enable_loop_closure: bool

    # Use DA3's ray-based pose estimation instead of the camera decoder
    use_ray_pose: bool = False

    # Inter-submap boundary scale chaining (see DA3SLAM._processing).
    # Damping g applies ratio^(1-g) to each boundary depth-ratio:
    # 0 = full chaining (default), 1 = ignore ratios and trust DA3's metric depth.
    boundary_scale_damping: float = 0.0

    # Clamp each applied boundary ratio to [1/c, c] (None = off).
    boundary_scale_clamp: float | None = None

    # HuggingFace CLIP model ID for semantic embeddings (None = disabled)
    semantic_model: str | None = None


def load_slam_config(
    yaml_path: str | Path = DEFAULT_YAML,
    **overrides: Any,
) -> SLAMConfig:
    """
    Build a SLAMConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML config file (defaults to config/default.yaml).
        **overrides: Scalar top-level keys to override, e.g.
                     submap_size=12, confidence_percentile=75.
                     Values of None are ignored, so CLI args can be passed
                     through unconditionally.

    Returns:
        A fully populated SLAMConfig.

    Note:
        Only top-level scalar keys can be overridden here.  Nested keys
        (keyframe.*, noise.*, loop_closure.*) are read straight from the
        YAML; callers that need to override them mutate the returned config
        object (see scripts/run_slam.py).
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    keyframe = cfg.get("keyframe", {})
    noise = cfg.get("noise", {})
    loop_closure = cfg.get("loop_closure", {})

    return SLAMConfig(
        submap_size=cfg["submap_size"],
        confidence_percentile=cfg["confidence_percentile"],
        depth_model=cfg["depth_model"],
        depth_model_resolution=cfg["depth_model_resolution"],
        use_ray_pose=bool(cfg.get("use_ray_pose", False)),
        boundary_scale_damping=float(cfg.get("boundary_scale_damping", 0.0)),
        boundary_scale_clamp=(
            float(cfg["boundary_scale_clamp"])
            if cfg.get("boundary_scale_clamp") is not None else None
        ),
        enable_loop_closure=loop_closure.get("enable", True),
        semantic_model=cfg.get("semantic_model", None),
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
            flow_downsample_factor=int(keyframe.get("flow_downsample_factor", 4)),
        ),
        noise=NoiseConfig(
            prior_sigma=noise["prior_sigma"],
            between_sigma=noise["between_sigma"],
            loop_sigma=noise["loop_sigma"],
        ),
        loop_closure=LoopClosureConfig(
            distance_threshold=loop_closure["distance_threshold"],
            min_submaps_apart=loop_closure["min_submaps_apart"],
            max_loop_closures=loop_closure["max_loop_closures"],
            min_confidence_ratio=loop_closure["min_confidence_ratio"],
        ),
    )
