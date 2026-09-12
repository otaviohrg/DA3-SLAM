"""
Real-time DA3-SLAM from a robot-mounted RGB camera streamed over the network.

Runs **inside the SLAM container on the GPU host**.  The robot runs
`scripts/spot_frame_sender.py`, which ships JPEG frames over TCP; this script
feeds them into `DA3SLAM.run_stream()` exactly as `run_realsense.py` feeds a
local camera.  Ctrl-C ends the stream cleanly — the final graph optimisation
still runs and the trajectory is still saved.

WHY OFFBOARD
------------
DA3-SLAM needs `nested-giant` to reach usable accuracy (the whole small->giant
ladder measured ~3x worse on TUM) and that peaks near 7 GB of GPU memory even
with bf16 backbones — more than a Jetson-class payload has.  Running inference
on the workstation keeps the system **byte-identical to the benchmarked one**,
which is the entire point of a robot test: it should tell you how the system
you tuned behaves on real hardware, not how a shrunken variant behaves.

The system stays purely MONOCULAR.  No odometry, no IMU, no depth — the robot
contributes pixels and nothing else.  If you want a reference trajectory to
score against, log it separately with `scripts/spot_log_odometry.py`; that data
is for evaluation only and never enters the pipeline.

Usage:
    # on the robot
    robot$  python scripts/spot_frame_sender.py --port 5555 --exposure 80

    # on the GPU host (viewer optional, run `rerun` on the host first)
    host$   docker compose run --rm --no-deps da3-slam \\
                python scripts/run_spot.py --robot_host 192.168.80.3 \\
                    --selection_mode disparity --backbone_dtype bf16

`--selection_mode disparity` is recommended: it emits keyframes as the robot
moves, rather than on a frame-count stride that assumes 30 fps.  A network
link typically delivers less, and the segment strides (8/16 frames) were tuned
at 30 fps — at 10 fps they triple the baseline per keyframe and push you into
the over-sparse regime.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

# scripts/ on sys.path so the sibling reused modules import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from da3_runner import add_da3_cli, build_config  # noqa: E402
from run_slam import print_summary, save_timings_json  # noqa: E402
from network_source import NetworkFrameSource  # noqa: E402


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Real-time DA3-SLAM from a networked robot camera",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_da3_cli(ap)

    # Live-domain defaults, same as run_realsense.py.  These were tuned for
    # handheld real-time rather than for the offline benchmarks, and a walking
    # robot is closer to that regime — especially the blur gate, since gait
    # oscillation blurs frames in a way neither TUM (handheld) nor UAS (drone)
    # exercises.  Explicit flags still win.
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
        backbone_dtype="bf16",
    )

    # ── robot link ───────────────────────────────────────────────────────────
    ap.add_argument("--robot_host", required=True,
                    help="Address of the robot running spot_frame_sender.py")
    ap.add_argument("--robot_port", type=int, default=5555)
    ap.add_argument("--connect_timeout", type=float, default=60.0,
                    help="Seconds to keep retrying the initial connection")
    ap.add_argument("--recv_timeout", type=float, default=10.0,
                    help="Seconds without data before the stream is considered "
                         "ended (WiFi on a walking robot does drop out; ending "
                         "cleanly still saves the trajectory)")

    # ── output ───────────────────────────────────────────────────────────────
    ap.add_argument("--out_dir", default="/app/outputs/spot",
                    help="Output directory for trajectory and map files")
    ap.add_argument("--skip_ply", action="store_true",
                    help="Skip saving the dense point cloud")

    # ── live viewer ──────────────────────────────────────────────────────────
    ap.add_argument("--viewer", choices=["connect", "serve", "spawn", "none"],
                    default="none",
                    help="Live Rerun viewer; 'connect' reaches a viewer running "
                         "on the host at --viewer_addr")
    ap.add_argument("--viewer_addr",
                    default="rerun+http://127.0.0.1:9876/proxy")
    ap.add_argument("--viewer_max_points", type=int, default=60_000)
    return ap.parse_args()


class _StopHandler:
    """First Ctrl-C ends the stream cleanly; a second force-quits."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.fired = False

    def __call__(self, signum, frame):
        if self.fired:
            print("\n[run_spot] second interrupt — force quit, nothing saved.")
            sys.exit(130)
        self.fired = True
        self.stop_event.set()
        print("\n[run_spot] stopping stream — finalising graph and saving...")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    lc = config.loop_closure
    print(f"[run_spot] MONOCULAR only — no odometry, no IMU, no depth")
    print(f"[run_spot] model={config.depth_model} @ "
          f"{config.depth_model_resolution} backbone={config.backbone_dtype}")
    print(f"[run_spot] submap size={config.submap_size} "
          f"overlap={config.submap_overlap}  "
          f"scale deadband={config.boundary_scale_deadband:g}  "
          f"blur gate window={config.keyframe.sharpness_window}/"
          f"ratio={config.keyframe.min_sharpness_ratio:g}")
    print(f"[run_spot] loop closure: distance<{lc.distance_threshold:g}  "
          f"gap>{lc.min_submaps_apart}  top-{lc.max_loop_closures}  "
          f"conf>{lc.min_confidence_ratio:g}  context={lc.context_frames}")

    stop_event = threading.Event()
    handler = _StopHandler(stop_event)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Build the viewer before the heavy model so a bad --viewer_addr fails fast.
    viewer = None
    if args.viewer != "none":
        from live_viewer import LiveViewer
        viewer = LiveViewer(mode=args.viewer, addr=args.viewer_addr,
                            max_points_per_submap=args.viewer_max_points)

    source = NetworkFrameSource(
        host=args.robot_host, port=args.robot_port, stop_event=stop_event,
        max_frames=args.max_frames, connect_timeout=args.connect_timeout,
        recv_timeout=args.recv_timeout,
    )

    from da3_slam.slam import DA3SLAM   # here so --help needs no GPU stack

    t_load = time.time()
    slam = DA3SLAM(config)
    t_load = time.time() - t_load

    print("[run_spot] streaming — drive the robot to build the map. "
          "Ctrl-C to stop and save.")
    t_run = time.time()
    try:
        result = slam.run_stream(source, on_update=viewer)
    except KeyboardInterrupt:
        print("\n[run_spot] force-quit before finalising — nothing saved.")
        sys.exit(130)
    except ConnectionError as exc:
        print(f"\n[run_spot] {exc}")
        print("           is spot_frame_sender.py running on the robot, and is "
              "the port reachable?")
        sys.exit(1)
    t_run = time.time() - t_run

    n_frames = max(source.n_yielded, 1)
    print_summary(result, n_frames, t_load, t_run)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save_kitti(str(out / "trajectory_kitti.txt"))
    # Capture timestamps (taken on the robot) key the TUM export, so the
    # trajectory is directly comparable to anything else logged there.
    result.save_tum(str(out / "trajectory_tum.txt"), timestamps=source.timestamps)
    if not args.skip_ply:
        try:
            result.save_ply(str(out / "map.ply"))
        except ValueError as exc:
            print(f"  [warn] no point cloud saved: {exc}")
    save_timings_json(out / "timings.json", result, n_frames, t_load, t_run)
    (out / "frame_timestamps.json").write_text(
        json.dumps({str(k): v for k, v in source.timestamps.items()}, indent=2))

    print(f"\n  Outputs saved to {out}/")
    print(f"    trajectory_tum.txt      ({result.n_keyframes} poses)")
    print(f"    frame_timestamps.json   (seq_idx -> robot capture time)")
    print(f"\n  To score against a reference trajectory logged with "
          f"spot_log_odometry.py:")
    print(f"    evo_ape tum <odometry.tum> {out}/trajectory_tum.txt -as")


if __name__ == "__main__":
    main()
