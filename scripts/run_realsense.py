"""
Real-time DA3-SLAM from an Intel RealSense camera, with a live Rerun viewer.

Runs **inside the SLAM container** (see docker-compose.yml `realsense` service).
Streams the camera's colour frames into DA3SLAM.run_stream(); a LiveViewer
callback logs the growing trajectory + map to a Rerun viewer as submaps are
optimised.  Ctrl-C ends the stream cleanly — the pipeline runs its final graph
optimisation and saves the trajectory + map exactly like scripts/run_slam.py.

Live viewer (no X11 needed): run the Rerun viewer on the host and let the
container connect out to it —

    host$   pip install rerun-sdk && rerun          # listens on :9876
    host$   docker compose run --rm realsense \\
                python3 scripts/run_realsense.py --selection_mode disparity

`--selection_mode disparity` is recommended for the demo: it emits keyframes as
you move (lower latency than the segment-density default).

Requires pyrealsense2 + rerun-sdk (requirements-realsense.txt, baked into the
demo image by setup.sh).
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

# scripts/ on sys.path so the sibling reused modules import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from da3_runner import add_da3_cli, build_config  # noqa: E402
from run_slam import print_summary, save_timings_json  # noqa: E402
from realsense_source import RealSenseFrameSource  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Real-time DA3-SLAM from a RealSense camera with a live viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # DA3-SLAM knobs (--config, --submap_size, --loop_* , keyframe, model, …).
    add_da3_cli(ap)

    # Live-demo defaults (any explicit flag still overrides).
    # The YAML defaults are benchmark-tuned; on a D455 640x480 stream genuine
    # revisits score ~0.45-0.55 SALAD distance and the 2-frame confidence
    # gate rejects true matches, so closures starve and drift goes
    # uncorrected (= the map gets rebuilt offset on revisits).  Looser
    # detection here is safe because the Huber loop noise + geometric gate
    # (with its corroboration pool) absorb the occasional false match; 3.0 m
    # translation disagreement is generous for room-scale drift.
    # submap_overlap=2 measures each boundary in both DA3 batches — handheld
    # runs showed single-anchor bridging letting one bad DA3 pose displace
    # every subsequent submap (the trajectory "jumps" and the map rebuilds
    # elsewhere); the redundant boundary factor + consistency warning make
    # that failure visible and much less damaging.
    # boundary_scale_deadband=0.25: live D455 runs showed genuine per-batch
    # metric scale breaks (boundary translation-unit ratios of 2-6x with
    # near-zero rotation disagreement) that damping=1.0 leaves uncorrected —
    # the dead-band keeps the sweep-validated "trust DA3" behaviour for
    # ratios near 1 but applies a break-sized ratio in full.
    # between_huber_k=1.345: pose breaks contained by the overlap-2 redundancy
    # still deform the graph when both contradictory boundary factors carry
    # full weight (the disagreement gets spread over the cycle → revisited
    # geometry smears into side-by-side copies); Huber concentrates that
    # error at the broken factor instead.
    # loop threshold 0.6: run #4 had zero candidates over its whole second
    # half (revisits scoring 0.56-0.8), so each re-pass re-drew geometry at
    # its drifted offset; dedup + gates + robust noise absorb the extra
    # false-candidate load.
    # Blur gating (sharpness_window / min_sharpness_ratio): fast handheld
    # motion smears frames, which degrades every measurement at once — DA3
    # poses (boundary breaks), descriptors (no retrieval; run #7 saw
    # distance-0.19 matches rejected at confidence 0.036) and repair
    # re-inference.  Prefer the sharp frames that survive at direction
    # reversals and micro-pauses.
    ap.set_defaults(
        submap_overlap=2,
        boundary_scale_deadband=0.25,
        between_huber_k=1.345,
        loop_distance_threshold=0.6,
        min_submaps_apart=2,
        max_loop_closures=3,
        min_confidence_ratio=0.2,
        loop_max_translation_error=3.0,
        sharpness_window=2,
        min_sharpness_ratio=0.5,
    )

    # ── camera ──────────────────────────────────────────────────────────────
    ap.add_argument("--width", type=int, default=640, help="Colour width (px)")
    ap.add_argument("--height", type=int, default=480, help="Colour height (px)")
    ap.add_argument("--fps", type=int, default=30, help="Colour stream FPS")
    ap.add_argument("--serial", default=None,
                    help="RealSense device serial (default: first camera)")
    ap.add_argument("--exposure", type=float, default=None,
                    help="Manual colour exposure in device units (D4xx: tenths "
                         "of a ms; lower = less motion blur, darker image; "
                         "disables auto-exposure).  Default: auto-exposure with "
                         "priority off, capping exposure at the frame budget")

    # ── output ──────────────────────────────────────────────────────────────
    ap.add_argument("--out_dir", default="/app/outputs/realsense",
                    help="Output directory for trajectory and map files")
    ap.add_argument("--skip_ply", action="store_true",
                    help="Skip saving the dense point cloud (map.ply)")

    # ── live viewer ─────────────────────────────────────────────────────────
    ap.add_argument("--viewer", choices=["connect", "serve", "spawn", "none"],
                    default="connect",
                    help="Rerun viewer mode: connect to a host viewer / serve a "
                         "web viewer / spawn a native window / disable")
    ap.add_argument("--viewer_addr",
                    default="rerun+http://127.0.0.1:9876/proxy",
                    help="Address of the host Rerun viewer (mode=connect). "
                         "127.0.0.1 works because the realsense service uses "
                         "host networking (see docker-compose.yml)")
    ap.add_argument("--viewer_max_points", type=int, default=60_000,
                    help="Max points logged per submap (subsampled for speed)")
    return ap.parse_args()


# ── stop handling ───────────────────────────────────────────────────────────────

class _StopHandler:
    """Signal handler: first Ctrl-C ends the stream and saves, second force-quits."""

    def __init__(self, stop_event: threading.Event):
        self._stop_event = stop_event
        self._count = 0

    def __call__(self, signum, frame) -> None:
        self._count += 1
        if self._count == 1:
            print("\n[run_realsense] stop requested — finishing pipeline and "
                  "saving (press Ctrl-C again to force quit) ...")
            self._stop_event.set()
        else:
            print("\n[run_realsense] force quit.")
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt


def install_stop_handler(stop_event: threading.Event) -> None:
    handler = _StopHandler(stop_event)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Stream the RealSense camera through DA3-SLAM with a live viewer.

    Ctrl-C stops the stream cleanly (final optimisation still runs and
    outputs are saved); a second Ctrl-C force-quits without saving.
    """
    args = parse_args()
    config = build_config(args)
    lc = config.loop_closure
    print(f"[run_realsense] submap size={config.submap_size} "
          f"overlap={config.submap_overlap}  "
          f"scale deadband={config.boundary_scale_deadband:g}  "
          f"huber between/loop={config.noise.between_huber_k}/"
          f"{config.noise.loop_huber_k}  "
          f"blur gate window={config.keyframe.sharpness_window}/"
          f"ratio={config.keyframe.min_sharpness_ratio:g}")
    print(f"[run_realsense] loop closure: distance<{lc.distance_threshold:g}  "
          f"gap>{lc.min_submaps_apart}  top-{lc.max_loop_closures}  "
          f"conf>{lc.min_confidence_ratio:g}  context={lc.context_frames}  "
          f"geo-gate rot<{lc.max_rotation_error_deg}° "
          f"trans<{lc.max_translation_error} (+corroboration pool)")

    stop_event = threading.Event()
    install_stop_handler(stop_event)

    # Build the viewer before loading the heavy model so a bad --viewer_addr
    # fails fast (rerun/pyrealsense imported lazily inside these helpers).
    viewer = None
    if args.viewer != "none":
        from live_viewer import LiveViewer
        viewer = LiveViewer(
            mode=args.viewer,
            addr=args.viewer_addr,
            max_points_per_submap=args.viewer_max_points,
        )

    source = RealSenseFrameSource(
        width=args.width, height=args.height, fps=args.fps,
        serial=args.serial, stop_event=stop_event, max_frames=args.max_frames,
        exposure=args.exposure,
    )

    from da3_slam.slam import DA3SLAM  # imported here so --help needs no GPU stack

    t_load = time.time()
    slam = DA3SLAM(config)
    t_load = time.time() - t_load

    print("[run_realsense] streaming — move the camera to build the map. "
          "Ctrl-C to stop and save.")
    t_run = time.time()
    try:
        result = slam.run_stream(source, on_update=viewer)
    except KeyboardInterrupt:
        print("\n[run_realsense] force-quit before finalising — nothing saved.")
        sys.exit(130)
    t_run = time.time() - t_run

    n_frames = max(source.n_yielded, 1)
    print_summary(result, n_frames, t_load, t_run)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save_kitti(str(out / "trajectory_kitti.txt"))
    result.save_tum(str(out / "trajectory_tum.txt"), timestamps=source.timestamps)
    if not args.skip_ply:
        result.save_ply(str(out / "map.ply"))
    save_timings_json(out / "timings.json", result, n_frames, t_load, t_run)

    print(f"\n  Outputs saved to {out}/")
    print(f"    trajectory_tum.txt   ({result.n_keyframes} poses)")


if __name__ == "__main__":
    main()
