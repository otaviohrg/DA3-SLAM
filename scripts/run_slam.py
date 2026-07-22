"""
DA3-SLAM entry point.

Defaults are read from config/default.yaml.
Any value can be overridden on the command line.

Usage:
    python scripts/run_slam.py --image_dir data/video1_30fps
    python scripts/run_slam.py --image_dir data/video1_30fps --config config/default.yaml
    python scripts/run_slam.py --image_dir data/video1_30fps \\
        --out_dir outputs/run1 --submap_size 12 --confidence_percentile 75 --no_loop_closure
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from da3_slam.config import load_slam_config, DEFAULT_YAML, SLAMConfig


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(yaml_config: dict) -> argparse.Namespace:
    """Build argument parser with defaults drawn from the loaded YAML config."""
    loop_closure_cfg = yaml_config.get("loop_closure", {})
    keyframe_cfg = yaml_config.get("keyframe", {})

    parser = argparse.ArgumentParser(
        description="DA3-SLAM runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── meta ──────────────────────────────────────────────────────────────────
    parser.add_argument("--image_dir", required=True,
                        help="Directory of input images")
    parser.add_argument("--out_dir", default="/app/outputs/slam",
                        help="Output directory for trajectory and map files")
    parser.add_argument("--config", default=str(DEFAULT_YAML),
                        help="Path to YAML config file")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap the number of input frames (for quick tests)")

    # ── frozen-keyframe harness (Step 0a) ─────────────────────────────────────
    parser.add_argument("--dump_keyframes", default=None,
                        help="Write the selected keyframe seq_idx list here "
                             "after the run (frozen-keyframe capture)")
    parser.add_argument("--keyframes_from", default=None,
                        help="Replay exactly this keyframe list, bypassing "
                             "optical-flow selection (byte-identical frames "
                             "across configs)")

    # ── DA3 model ─────────────────────────────────────────────────────────────
    parser.add_argument("--depth_model", default=yaml_config.get("depth_model"),
                        help="DA3 model ID")
    parser.add_argument("--depth_model_resolution", type=int,
                        default=yaml_config.get("depth_model_resolution"),
                        help="DA3 processing resolution")
    parser.add_argument("--use_ray_pose", action=argparse.BooleanOptionalAction,
                        default=bool(yaml_config.get("use_ray_pose", False)),
                        help="Use ray-based pose estimation instead of the camera decoder")

    # ── submap ────────────────────────────────────────────────────────────────
    parser.add_argument("--submap_size", type=int,
                        default=yaml_config.get("submap_size"),
                        help="Max keyframes per submap (including anchor overlap)")
    parser.add_argument("--boundary_scale_damping", type=float,
                        default=yaml_config.get("boundary_scale_damping"),
                        help="Damping g for inter-submap scale chaining: each "
                             "boundary depth-ratio is raised to (1-g). "
                             "0 = full chaining, 1 = trust DA3 metric depth")
    parser.add_argument("--boundary_scale_clamp", type=float,
                        default=yaml_config.get("boundary_scale_clamp"),
                        help="Clamp each boundary scale ratio to [1/c, c] "
                             "(unset = no clamping)")

    # ── keyframe selection ────────────────────────────────────────────────────
    parser.add_argument("--min_disparity_fraction", type=float,
                        default=keyframe_cfg.get("min_disparity_fraction"),
                        help="Min optical flow as fraction of image width [0,1]")
    parser.add_argument("--selection_mode", choices=["disparity", "segment"],
                        default=keyframe_cfg.get("selection_mode", "disparity"),
                        help="Keyframe policy: 'disparity' (frame-level threshold) "
                             "or 'segment' (segment-level density control)")
    parser.add_argument("--segment_length", type=int,
                        default=keyframe_cfg.get("segment_length"),
                        help="Segment mode: frames per segment (N_S)")
    parser.add_argument("--segment_threshold", type=float,
                        default=keyframe_cfg.get("segment_disparity_threshold"),
                        help="Segment mode: accumulated flow (px) above which the "
                             "dense stride is used (tau_seg)")

    # ── point cloud ───────────────────────────────────────────────────────────
    parser.add_argument("--confidence_percentile", type=float,
                        default=yaml_config.get("confidence_percentile"),
                        help="Global confidence percentile threshold (0-100). "
                             "Higher = fewer but cleaner points")

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument("--skip_ply", action="store_true",
                        help="Skip saving the dense point cloud (map.ply). "
                             "Use during benchmarking to avoid I/O overhead.")

    # ── live viewer (optional) ────────────────────────────────────────────────
    parser.add_argument("--viewer", choices=["connect", "serve", "spawn", "none"],
                        default="none",
                        help="Live Rerun viewer for the replay: connect to a "
                             "host viewer / serve a web viewer / spawn a native "
                             "window / disabled. Requires rerun-sdk; in Docker "
                             "the container needs host networking to reach the "
                             "host viewer (use `make run-viz`)")
    parser.add_argument("--viewer_addr",
                        default="rerun+http://127.0.0.1:9876/proxy",
                        help="Address of the host Rerun viewer (mode=connect)")
    parser.add_argument("--viewer_max_points", type=int, default=60_000,
                        help="Max points logged per submap (subsampled for speed)")

    # ── loop closure ──────────────────────────────────────────────────────────
    parser.add_argument("--no_loop_closure", action="store_true",
                        default=not loop_closure_cfg.get("enable", True),
                        help="Disable loop closure detection")
    # --loop_threshold kept as an alias for backwards compatibility
    parser.add_argument("--loop_distance_threshold", "--loop_threshold", type=float,
                        default=loop_closure_cfg.get("distance_threshold"),
                        help="DINO-SALAD descriptor L2 distance threshold for loop "
                             "detection (lower = stricter)")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SLAMConfig:
    """Build the SLAMConfig from YAML, applying CLI overrides on top."""
    # load_slam_config handles top-level scalars; nested keys are set below.
    config = load_slam_config(
        args.config,
        submap_size=args.submap_size,
        confidence_percentile=args.confidence_percentile,
        depth_model=args.depth_model,
        depth_model_resolution=args.depth_model_resolution,
        use_ray_pose=args.use_ray_pose,
        boundary_scale_damping=args.boundary_scale_damping,
        boundary_scale_clamp=args.boundary_scale_clamp,
    )
    if args.no_loop_closure:
        config.enable_loop_closure = False
    if args.loop_distance_threshold is not None:
        config.loop_closure.distance_threshold = args.loop_distance_threshold
    if args.min_disparity_fraction is not None:
        config.keyframe.min_disparity_fraction = args.min_disparity_fraction
    config.keyframe.selection_mode = args.selection_mode
    if args.segment_length is not None:
        config.keyframe.segment_length = args.segment_length
    if args.segment_threshold is not None:
        config.keyframe.segment_disparity_threshold = args.segment_threshold
    config.keyframes_from = args.keyframes_from
    config.dump_keyframes = args.dump_keyframes
    return config


# ── input collection ───────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_paths(image_dir: str, max_frames: int | None) -> list[str]:
    """Sorted RGB image paths from a directory.

    If the directory mixes RGB frames (frame*.jpg) and depth maps
    (depth*.png), only the RGB frames are kept.
    """
    all_images = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if any(p.stem.startswith("depth") for p in all_images):
        all_images = [p for p in all_images if not p.stem.startswith("depth")]
    image_paths = [str(p) for p in all_images]
    if max_frames:
        image_paths = image_paths[:max_frames]
    return image_paths


def timestamps_from_filenames(image_paths: list[str]) -> dict[int, float] | None:
    """Map seq_idx → timestamp when filenames are numeric (e.g. TUM), else None."""
    try:
        return {i: float(Path(p).stem) for i, p in enumerate(image_paths)}
    except ValueError:
        return None  # non-numeric filenames — caller falls back to seq_idx / fps


# ── reporting ──────────────────────────────────────────────────────────────────

def print_summary(result, n_frames: int, model_load_seconds: float, pipeline_seconds: float) -> None:
    """Print the end-of-run report: counts, trajectory length, and the
    per-module timing breakdown normalised per frame / per submap."""
    positions = result.trajectory[:, :3, 3]
    path_length = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    n_submaps = max(len(result.submaps), 1)
    timings = result.timings

    print("\n" + "─" * 60)
    print("  SLAM Summary")
    print("─" * 60)
    print(f"  Frames processed:    {n_frames}")
    print(f"  Keyframes:           {result.n_keyframes}")
    print(f"  Submaps:             {len(result.submaps)}")
    print(f"  Loop closures:       {len(result.loop_closures)}")
    print(f"  Opt. final error:    {result.optimization.final_error:.6f}")
    print(f"  Trajectory length:   {path_length:.3f} m")
    print(f"  Model load time:     {model_load_seconds:.1f}s")
    print(f"  Pipeline time:       {pipeline_seconds:.1f}s")
    print(f"  Total wall time:     {model_load_seconds + pipeline_seconds:.1f}s")
    print(f"  FPS (pipeline):      {n_frames / pipeline_seconds:.1f}")
    print()
    print("  Timing breakdown (total | per unit):")
    print(f"    {'keyframe_selection':<25} {timings['keyframe_selection']:6.2f}s  "
          f"| {timings['keyframe_selection'] / n_frames * 1000:.2f} ms/frame")
    print(f"    {'submap_building':<25} {timings['submap_building']:6.2f}s  "
          f"| {timings['submap_building'] / n_submaps:.2f} s/submap")
    print(f"    {'loop_closure':<25} {timings['loop_closure']:6.2f}s  "
          f"| {timings['loop_closure'] / n_submaps:.2f} s/submap")
    print(f"    {'graph_building':<25} {timings['graph_building']:6.2f}s  "
          f"| {timings['graph_building'] / n_submaps * 1000:.1f} ms/submap")
    print(f"    {'optimization':<25} {timings['optimization']:6.2f}s  "
          f"| {timings['optimization'] / n_submaps * 1000:.1f} ms/submap")


def save_timings_json(path: Path, result, n_frames: int,
                      model_load_seconds: float, pipeline_seconds: float) -> None:
    """Write timings.json.

    The key names are read by the eval harness (evals/eval_tum.sh and
    friends) — do not rename them.
    """
    n_submaps = len(result.submaps)
    timings = result.timings
    data = {
        "model_load": round(model_load_seconds, 3),
        "pipeline": round(pipeline_seconds, 3),
        "wall_total": round(model_load_seconds + pipeline_seconds, 3),
        "frames": n_frames,
        "keyframes": result.n_keyframes,
        "submaps": n_submaps,
        "loop_closures": len(result.loop_closures),
        **{k: round(v, 3) for k, v in timings.items()},
        # Derived per-step metrics
        "fps": round(n_frames / pipeline_seconds, 2) if pipeline_seconds > 0 else 0,
        "kf_sel_ms_per_frame": round(timings["keyframe_selection"] / n_frames * 1000, 3) if n_frames > 0 else 0,
        "submap_s_per_submap": round(timings["submap_building"] / n_submaps, 3) if n_submaps > 0 else 0,
        "lc_s_per_submap": round(timings["loop_closure"] / n_submaps, 3) if n_submaps > 0 else 0,
        "graph_ms_per_submap": round(timings["graph_building"] / n_submaps * 1000, 3) if n_submaps > 0 else 0,
        "opt_ms_per_submap": round(timings["optimization"] / n_submaps * 1000, 3) if n_submaps > 0 else 0,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    """Run the full pipeline over an image directory and save all outputs
    (trajectories, optional map.ply, timings.json) to --out_dir."""
    # Parse --config first so the YAML can seed the remaining CLI defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_YAML))
    known, _ = pre_parser.parse_known_args()
    with open(known.config) as f:
        yaml_config = yaml.safe_load(f)
    args = parse_args(yaml_config)

    image_paths = collect_image_paths(args.image_dir, args.max_frames)
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        sys.exit(1)
    print(f"[run_slam] {len(image_paths)} images from {args.image_dir}")

    config = build_config(args)

    # Build the viewer before loading the heavy model so a bad --viewer_addr
    # fails fast (rerun-sdk imported lazily inside LiveViewer).
    viewer = None
    if args.viewer != "none":
        from live_viewer import LiveViewer
        viewer = LiveViewer(
            mode=args.viewer,
            addr=args.viewer_addr,
            max_points_per_submap=args.viewer_max_points,
        )

    # Lean frames (no per-frame point clouds) when nothing consumes them —
    # cuts resident memory per keyframe ~3x on long runs.
    config.build_pointclouds = (not args.skip_ply) or (viewer is not None)

    # Imported here so `--help` works without the GPU stack installed.
    from da3_slam.slam import DA3SLAM

    model_load_seconds = time.time()
    slam = DA3SLAM(config)
    model_load_seconds = time.time() - model_load_seconds

    pipeline_seconds = time.time()
    result = slam.run(image_paths, on_update=viewer)
    pipeline_seconds = time.time() - pipeline_seconds

    print_summary(result, len(image_paths), model_load_seconds, pipeline_seconds)

    # ── save outputs ───────────────────────────────────────────────────────────
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.save_kitti(str(out / "trajectory_kitti.txt"))
    result.save_tum(str(out / "trajectory_tum.txt"),
                    timestamps=timestamps_from_filenames(image_paths))
    if not args.skip_ply:
        result.save_ply(str(out / "map.ply"))
    save_timings_json(out / "timings.json", result, len(image_paths), model_load_seconds, pipeline_seconds)

    print(f"\n  Outputs saved to {out}/")
    print(f"    trajectory_kitti.txt  ({result.n_keyframes} poses)")
    print(f"    trajectory_tum.txt    ({result.n_keyframes} poses)")
    if not args.skip_ply:
        n_points = sum(len(sm.points_world) for sm in result.submaps)
        print(f"    map.ply               ({n_points:,} points)")


if __name__ == "__main__":
    main()
