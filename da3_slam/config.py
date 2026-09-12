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

from argparse import BooleanOptionalAction
from dataclasses import dataclass, field
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

    # Dense-matching verification gate (RoMa v2).  Retrieval with a single
    # global DINO-SALAD descriptor cannot do spatial verification and gives no
    # credit for partial overlap, so `distance_threshold` ends up doing two
    # jobs at once and does not transfer between domains (TUM wants 0.80, UAS
    # 0.60).  With this gate on, retrieval becomes a pure RECALL knob — set
    # distance_threshold loose — and precision comes from dense correspondence:
    # a candidate is rejected unless RoMa's predicted overlap reaches
    # `roma_min_overlap`.  None/0 = gate disabled (the shipped behaviour).
    roma_gate: bool = False
    roma_min_overlap: float = 0.30

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
class TokenMergingConfig:
    """Cross-view token merging in the DA3 backbone (the FastVGGT port).

    Training-free acceleration of the backbone's *global* (cross-view)
    attention: most tokens are merged into a smaller destination set before
    attention and scattered back afterwards.  Implementation and the full
    rationale: ``da3_slam.backend.inference.token_merge``.

    **Off by default.**  This is an experimental compute lever, not a tuned
    one — the baseline it must be compared against is ``enable: false``, and
    that comparison is the whole point of the flag.
    """

    # Master switch.  False = the backbone runs exactly as upstream DA3 does.
    enable: bool = False

    # First merging block, as a POSITION AMONG THE GLOBAL BLOCKS (0 = merge in
    # all of them).  Not a raw block index: giant has 14 global blocks and large
    # has 8, so only the relative position transfers across model sizes.
    start: int = 0

    # Fraction of tokens to absorb into the destination set.
    merge_ratio: float = 0.9

    # Destination stride over the patch grid (one kept per sx-by-sy cell).
    sx: int = 2
    sy: int = 2

    # Hold a uniform stride of tokens out of the merge entirely.
    protect: bool = True
    protect_ratio: float = 0.1

    # Seed for the destination choice — fixed so a config is reproducible.
    seed: int = 33

    # Never merge batches smaller than this.  Keeps the loop-closure worker's
    # small re-inference batches on the exact path, where merging would save
    # nothing and could degrade the inference that accepts or rejects a closure.
    min_frames: int = 8


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

    # Weight precision for the two ViT backbones: "fp32" (as shipped) or
    # "bf16".  The backbones already run under autocast bf16, so fp32 masters
    # are storage the forward never uses; bf16 cuts peak GPU memory ~40% and
    # roughly doubles the reachable submap size.  EXPERIMENTAL — it perturbs
    # poses well beyond the bf16 noise floor, so validate ATE before adopting.
    # See da3_slam.backend.inference.precision.
    backbone_dtype: str = "fp32"

    # Anchor keyframes shared between consecutive submaps.  1 = single shared
    # node (VGGT-SLAM style).  >= 2 measures the shared frame pair in *both*
    # DA3 batches: the duplicate between-factor adds redundancy across the
    # boundary and enables the boundary-consistency check, so one bad DA3
    # anchor pose can no longer displace a whole submap silently.
    submap_overlap: int = 1

    # Pose-graph parameterisation: "sl4" (15 DOF projective, inherited from
    # VGGT-SLAM) or "sim3" (7 DOF rigid+scale).  SL(4)'s extra DOF are
    # justified by projective ambiguity in UNCALIBRATED monocular
    # reconstruction; DA3 predicts metric depth and intrinsics, so the measured
    # inter-submap disagreement is rotation + scale, i.e. Sim(3).  Note the
    # choice is inert in a chain-only graph (exactly determined) — it can only
    # matter where loop closures or overlap>=2 create redundancy.
    pose_parameterisation: str = "sl4"

    # Extra within-submap between-factors linking frames k apart (k = 2, 4, 8).
    # DA3 estimates a batch jointly, so a stride-k relative pose is one direct
    # measurement rather than k composed ones; adding them costs no inference
    # and makes the pose graph over-determined (a consecutive-only chain is
    # exactly determined, so the noise model has no effect on it at all).
    # Empty = consecutive links only (previous behaviour).
    submap_skip_strides: tuple[int, ...] = ()

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

    # Cross-view token merging in the DA3 backbone (off by default — it is an
    # experimental compute lever whose baseline is `enable: false`).
    token_merging: TokenMergingConfig = field(default_factory=TokenMergingConfig)

    # Runtime flag (not read from YAML): set False when nothing consumes point
    # clouds (no map.ply export, no live viewer) — benchmark/sweep drivers do
    # this.  SubmapBuilder then stores empty points/colors/confidence/mask on
    # every Frame, cutting resident memory per keyframe ~3x.
    build_pointclouds: bool = True

    # Frozen-keyframe harness (runtime flags, not read from YAML; set by the
    # CLI in run_slam.py / da3_runner.py).  keyframes_from replays exactly the
    # recorded keyframe seq_idxs, bypassing optical-flow selection so two
    # configs are compared on byte-identical frames; dump_keyframes writes the
    # selected list after the run.  See da3_slam.frontend.keyframe_selector.
    keyframes_from: str | None = None
    dump_keyframes: str | None = None


class _BoolFlag(BooleanOptionalAction):
    """BooleanOptionalAction that also accepts the underscore negation.

    Stock argparse only generates ``--no-token_merging``; every other negated
    flag in this repo is underscored (``--no_loop_closure``, ``--no_undistort``),
    so both spellings are registered and mean the same thing.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.option_strings = list(self.option_strings) + [
            f"--no_{opt[2:]}" for opt in option_strings if opt.startswith("--")
        ]

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest,
                not option_string.startswith(("--no-", "--no_")))


def add_token_merging_cli(parser, yaml_defaults: dict | None = None) -> None:
    """Add the token-merging flags to an argparse parser.

    Shared by run_slam.py and the benchmark drivers so the two never drift.
    All defaults are None ("leave the YAML value alone") except the master
    switch, which reads its default from the YAML when one is supplied.

        --token_merging / --no_token_merging     the A/B switch
        --merge_start / --merge_ratio / --merge_min_frames / --no_merge_protect
    """
    merging = (yaml_defaults or {}).get("token_merging", {}) or {}
    group = parser.add_argument_group("cross-view token merging (experimental)")
    group.add_argument(
        "--token_merging", action=_BoolFlag,
        default=bool(merging.get("enable", False)) if yaml_defaults else None,
        help="Merge cross-view attention tokens in the DA3 backbone "
             "(--no_token_merging for the unmerged baseline)")
    group.add_argument(
        "--merge_start", type=int, default=None,
        help="First merging block as a position among the global blocks "
             "(0 = all of them)")
    group.add_argument(
        "--merge_ratio", type=float, default=None,
        help="Fraction of tokens absorbed into the destination set")
    group.add_argument(
        "--merge_min_frames", type=int, default=None,
        help="Never merge batches smaller than this")
    group.add_argument(
        "--merge_protect", action=_BoolFlag, default=None,
        help="Hold a uniform stride of tokens out of the merge")


def apply_token_merging_cli(config: "SLAMConfig", args) -> None:
    """Apply the flags added by add_token_merging_cli() to a built config."""
    merging = config.token_merging
    if getattr(args, "token_merging", None) is not None:
        merging.enable = bool(args.token_merging)
    for flag, field_name in (
        ("merge_start", "start"),
        ("merge_ratio", "merge_ratio"),
        ("merge_min_frames", "min_frames"),
        ("merge_protect", "protect"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(merging, field_name, value)


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
    token_merging = cfg.get("token_merging", {}) or {}

    return SLAMConfig(
        submap_size=cfg["submap_size"],
        submap_overlap=int(cfg.get("submap_overlap", 1)),
        confidence_percentile=cfg["confidence_percentile"],
        depth_model=cfg["depth_model"],
        depth_model_resolution=cfg["depth_model_resolution"],
        use_ray_pose=bool(cfg.get("use_ray_pose", False)),
        backbone_dtype=str(cfg.get("backbone_dtype", "fp32")),
        pose_parameterisation=str(cfg.get("pose_parameterisation", "sl4")),
        submap_skip_strides=tuple(cfg.get("submap_skip_strides", ()) or ()),
        boundary_scale_damping=float(cfg.get("boundary_scale_damping", 0.0)),
        boundary_scale_clamp=_optional_float(cfg, "boundary_scale_clamp"),
        boundary_scale_deadband=float(cfg.get("boundary_scale_deadband", 0.0)),
        enable_loop_closure=loop_closure.get("enable", True),
        semantic_model=cfg.get("semantic_model", None),
        token_merging=TokenMergingConfig(
            enable=bool(token_merging.get("enable", False)),
            start=int(token_merging.get("start", 0)),
            merge_ratio=float(token_merging.get("merge_ratio", 0.9)),
            sx=int(token_merging.get("sx", 2)),
            sy=int(token_merging.get("sy", 2)),
            protect=bool(token_merging.get("protect", True)),
            protect_ratio=float(token_merging.get("protect_ratio", 0.1)),
            seed=int(token_merging.get("seed", 33)),
            min_frames=int(token_merging.get("min_frames", 8)),
        ),
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
            roma_gate=bool(loop_closure.get("roma_gate", False)),
            roma_min_overlap=float(loop_closure.get("roma_min_overlap", 0.30)),
            max_rotation_error_deg=_optional_float(
                loop_closure, "max_rotation_error_deg"),
            max_translation_error=_optional_float(
                loop_closure, "max_translation_error"),
        ),
    )
