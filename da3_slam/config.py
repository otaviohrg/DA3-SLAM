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


def _optional_float(mapping: dict, key: str) -> float | None:
    """Float value of a YAML key, or None when the key is absent/null."""
    value = mapping.get(key)
    return float(value) if value is not None else None


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

    # Huber kernel parameter k for loop-closure factors (robust M-estimator).
    # A single wrong closure (perceptual aliasing) is then down-weighted
    # instead of warping the whole map.  None = plain Gaussian noise.
    loop_huber_k: float | None = None

    # Huber kernel k for odometry between-factors.  Only matters where the
    # graph has redundancy (duplicate boundary factors from submap_overlap
    # >= 2, cycles created by loop closures): the error of a broken boundary
    # measurement then concentrates at that one factor instead of being
    # spread over the whole cycle — which is what smears revisited geometry
    # into side-by-side copies.  None = plain Gaussian noise.
    between_huber_k: float | None = None


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
    # Analogous to VGGT-SLAM's image_match_ratio >= 0.85 gate.  With
    # context_frames > 0 the mean is taken over the two *matched* frames only.
    min_confidence_ratio: float

    # Temporal neighbours of each matched frame included in the verification
    # re-inference (per side, clipped to the submap).  1 → up to 3 frames per
    # matched frame, 6 total.  Wide-baseline 2-frame DA3 inference is poorly
    # conditioned; neighbours give it multi-view support, making both the
    # relative pose and the confidence gate meaningful.  0 = pair only.
    context_frames: int = 1

    # Geometric sanity gate: reject a closure whose measured relative pose
    # disagrees with the pose graph's current prediction by more than this
    # rotation angle (degrees).  Generous by design — drift is exactly what
    # closures must correct.  None = disabled.
    max_rotation_error_deg: float | None = 90.0

    # Same gate on the translation discrepancy, in global metric units
    # (~metres).  Must be well above the worst plausible drift for the
    # scenario — room-scale demos can use a few metres; large-scale runs
    # should leave it off.  None = disabled.
    max_translation_error: float | None = None


@dataclass
class SLAMConfig:
    """Top-level configuration for the full DA3-SLAM pipeline.

    Canonical values: config/default.yaml.
    Use load_slam_config() to construct from YAML.
    """

    keyframe: KeyframeSelectorConfig
    noise: NoiseConfig
    loop_closure: LoopClosureConfig

    # Frames per submap (including the anchor overlap)
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

    # Anchor keyframes shared between consecutive submaps.  1 = single shared
    # node (VGGT-SLAM style).  >= 2 measures the shared frame pair in *both*
    # DA3 batches: the duplicate between-factor adds redundancy across the
    # boundary and enables the boundary-consistency check, so one bad DA3
    # anchor pose can no longer displace a whole submap silently.
    submap_overlap: int = 1

    # Inter-submap boundary scale chaining (see DA3SLAM._processing).
    # Damping g applies ratio^(1-g) to each boundary depth-ratio:
    # 0 = full chaining, 1 = ignore ratios and trust DA3's metric depth
    # (default.yaml ships 1.0 — validated by the UAS/TUM scale sweeps).
    boundary_scale_damping: float = 0.0

    # Clamp each applied boundary ratio to [1/c, c] (None = off).
    boundary_scale_clamp: float | None = None

    # Scale-break dead-band: when > 0, a boundary depth-ratio within
    # max(r, 1/r) - 1 <= deadband is treated as 1.0 (trust DA3's metric
    # consistency — the sweep-validated regime), but a ratio *outside* the
    # band is applied in full, damping ignored (a genuine per-batch metric
    # scale break, seen on live D455 runs as 2-6x translation-unit jumps —
    # the trajectory suddenly stretches and the map rebuilds elsewhere).
    # 0 = disabled (legacy damping behaviour only).
    boundary_scale_deadband: float = 0.0

    # HuggingFace CLIP model ID for semantic embeddings (None = disabled)
    semantic_model: str | None = None

    # Runtime flag (not read from YAML): set False when nothing consumes point
    # clouds (no map.ply export, no live viewer) — benchmark/sweep drivers do
    # this.  SubmapBuilder then stores empty points/colors/confidence/mask on
    # every Frame, cutting resident memory per keyframe ~3x.
    build_pointclouds: bool = True


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
        submap_overlap=int(cfg.get("submap_overlap", 1)),
        confidence_percentile=cfg["confidence_percentile"],
        depth_model=cfg["depth_model"],
        depth_model_resolution=cfg["depth_model_resolution"],
        use_ray_pose=bool(cfg.get("use_ray_pose", False)),
        boundary_scale_damping=float(cfg.get("boundary_scale_damping", 0.0)),
        boundary_scale_clamp=_optional_float(cfg, "boundary_scale_clamp"),
        boundary_scale_deadband=float(cfg.get("boundary_scale_deadband", 0.0)),
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
            selection_mode=str(keyframe.get("selection_mode", "disparity")),
            segment_length=int(keyframe.get("segment_length", 128)),
            segment_disparity_threshold=float(
                keyframe.get("segment_disparity_threshold", 650.0)),
            segment_strides=tuple(keyframe.get("segment_strides", (8, 16))),
            sharpness_window=int(keyframe.get("sharpness_window", 0)),
            min_sharpness_ratio=float(keyframe.get("min_sharpness_ratio", 0.0)),
        ),
        noise=NoiseConfig(
            prior_sigma=noise["prior_sigma"],
            between_sigma=noise["between_sigma"],
            loop_sigma=noise["loop_sigma"],
            loop_huber_k=_optional_float(noise, "loop_huber_k"),
            between_huber_k=_optional_float(noise, "between_huber_k"),
        ),
        loop_closure=LoopClosureConfig(
            distance_threshold=loop_closure["distance_threshold"],
            min_submaps_apart=loop_closure["min_submaps_apart"],
            max_loop_closures=loop_closure["max_loop_closures"],
            min_confidence_ratio=loop_closure["min_confidence_ratio"],
            context_frames=int(loop_closure.get("context_frames", 1)),
            max_rotation_error_deg=_optional_float(
                loop_closure, "max_rotation_error_deg"),
            max_translation_error=_optional_float(
                loop_closure, "max_translation_error"),
        ),
    )
