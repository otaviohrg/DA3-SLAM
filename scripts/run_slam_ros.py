"""
Real-time DA3-SLAM runner — consumes the spool written by ros_spool_bridge.py.

Runs **inside the SLAM container** (no ROS needed).  It tails the bind-mounted
spool directory the host bridge writes into, streams frames into
`DA3SLAM.run_stream()`, and on end-of-stream (the bridge's DONE sentinel, or
Ctrl-C) saves the trajectory and map exactly like scripts/run_slam.py.

Real-time drop-oldest: when SLAM falls behind, only the newest queued frame is
fed in and the backlog is discarded (pass --process_all to disable). Consumed
frame files are deleted as they are read, so the spool stays bounded.

Typical use (host runs ros_spool_bridge.py + `ros2 bag play` against the same
--spool_dir, bind-mounted here as /spool):

    python scripts/run_slam_ros.py --spool_dir /spool --out_dir outputs/ros_run
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import yaml

from da3_slam.config import DEFAULT_YAML, SLAMConfig, load_slam_config

# Reuse the offline runner's reporting helpers (no Namespace dependency).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_slam import print_summary, save_timings_json  # noqa: E402

# A streamed frame: (RGB image, seq_idx, label) — see da3_slam.slam.FrameItem.
# Spelled out here so this module's --help works without the GPU/torch stack.
FrameItem = tuple[np.ndarray, int, str]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(cfg: dict) -> argparse.Namespace:
    lc = cfg.get("loop_closure", {})
    kf = cfg.get("keyframe", {})
    ap = argparse.ArgumentParser(
        description="Real-time DA3-SLAM over a spool directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--spool_dir", default="/spool",
                    help="Spool dir shared with the host ros_spool_bridge.py")
    ap.add_argument("--out_dir", default="/app/outputs/slam_ros")
    ap.add_argument("--config", default=str(DEFAULT_YAML))
    ap.add_argument("--poll_interval", type=float, default=0.01,
                    help="Seconds between spool dir scans")
    ap.add_argument("--startup_timeout", type=float, default=60.0,
                    help="Max seconds to wait for the first frame before giving up")
    ap.add_argument("--process_all", action="store_true",
                    help="Process every spooled frame in order (disable real-time "
                         "drop-oldest). Use for reproducible offline-style runs.")
    ap.add_argument("--keep_frames", action="store_true",
                    help="Do not delete consumed frame files from the spool dir")

    # SLAM overrides (mirror run_slam.py).
    ap.add_argument("--depth_model", default=cfg.get("depth_model"))
    ap.add_argument("--depth_model_resolution", type=int,
                    default=cfg.get("depth_model_resolution"))
    ap.add_argument("--use_ray_pose", action=argparse.BooleanOptionalAction,
                    default=bool(cfg.get("use_ray_pose", False)))
    ap.add_argument("--submap_size", type=int, default=cfg.get("submap_size"))
    ap.add_argument("--boundary_scale_damping", type=float,
                    default=cfg.get("boundary_scale_damping"))
    ap.add_argument("--boundary_scale_clamp", type=float,
                    default=cfg.get("boundary_scale_clamp"))
    ap.add_argument("--min_disparity_fraction", type=float,
                    default=kf.get("min_disparity_fraction"))
    ap.add_argument("--confidence_percentile", type=float,
                    default=cfg.get("confidence_percentile"))
    ap.add_argument("--skip_ply", action="store_true")
    ap.add_argument("--no_loop_closure", action="store_true",
                    default=not lc.get("enable", True))
    ap.add_argument("--loop_distance_threshold", "--loop_threshold", type=float,
                    default=lc.get("distance_threshold"))
    return ap.parse_args()


def build_config(args: argparse.Namespace) -> SLAMConfig:
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
    return config


# ── spool frame source ────────────────────────────────────────────────────────

def _parse_frame_name(path: Path) -> tuple[int, float]:
    """frame_<seq:08d>_<ts_ns>.jpg → (seq, timestamp_seconds)."""
    _, seq, ts_ns = path.stem.split("_")
    return int(seq), int(ts_ns) / 1e9


class SpoolFrameSource:
    """Iterable frame source that tails the bridge's spool directory.

    Yields (RGB image, seq, label) and records seq→timestamp in `.timestamps`
    for trajectory export.  Iteration ends once the DONE sentinel exists and no
    unread frames remain.
    """

    def __init__(self, spool_dir: str, *, poll_interval: float, startup_timeout: float,
                 drop_oldest: bool, delete_consumed: bool):
        self.spool = Path(spool_dir)
        self.poll_interval = poll_interval
        self.startup_timeout = startup_timeout
        self.drop_oldest = drop_oldest
        self.delete_consumed = delete_consumed

        self.timestamps: dict[int, float] = {}
        self._seen: set[str] = set()
        self.n_yielded = 0
        self.n_dropped = 0

    def _done(self) -> bool:
        return (self.spool / "DONE").exists()

    def _pending(self) -> list[Path]:
        files = sorted(p for p in self.spool.glob("frame_*.jpg")
                       if p.name not in self._seen)
        return files

    def _consume(self, path: Path) -> None:
        self._seen.add(path.name)
        if self.delete_consumed:
            path.unlink(missing_ok=True)

    def _read(self, path: Path) -> tuple[np.ndarray, int, str] | None:
        seq, ts = _parse_frame_name(path)
        bgr = cv2.imread(str(path))
        self._consume(path)
        if bgr is None:
            return None  # rare: file vanished / unreadable — skip
        self.timestamps[seq] = ts
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), seq, str(path)

    def __iter__(self) -> Iterator[FrameItem]:
        # Wait for the bridge to produce the first frame.
        waited = 0.0
        while not self._pending() and not self._done():
            time.sleep(self.poll_interval)
            waited += self.poll_interval
            if waited > self.startup_timeout:
                raise TimeoutError(
                    f"No frames in {self.spool} after {self.startup_timeout}s — "
                    "is the host bridge running and bag playing?")

        while True:
            pending = self._pending()
            if not pending:
                if self._done():
                    return
                time.sleep(self.poll_interval)
                continue

            if self.drop_oldest and len(pending) > 1:
                # Real-time: keep only the newest; discard the backlog.
                for stale in pending[:-1]:
                    self._consume(stale)
                    self.n_dropped += 1
                pending = pending[-1:]

            for path in pending:
                frame = self._read(path)
                if frame is None:
                    continue
                self.n_yielded += 1
                yield frame


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_YAML))
    known, _ = pre.parse_known_args()
    with open(known.config) as f:
        cfg = yaml.safe_load(f)
    args = parse_args(cfg)

    config = build_config(args)
    source = SpoolFrameSource(
        args.spool_dir,
        poll_interval=args.poll_interval,
        startup_timeout=args.startup_timeout,
        drop_oldest=not args.process_all,
        delete_consumed=not args.keep_frames,
    )

    from da3_slam.slam import DA3SLAM  # imported here so --help needs no GPU stack

    t_load = time.time()
    slam = DA3SLAM(config)
    t_load = time.time() - t_load

    print(f"[run_slam_ros] tailing spool {args.spool_dir} "
          f"(drop_oldest={not args.process_all}) ...")
    t_run = time.time()
    try:
        result = slam.run_stream(source)
    except KeyboardInterrupt:
        print("\n[run_slam_ros] interrupted — no result to save.")
        sys.exit(130)
    t_run = time.time() - t_run

    n_frames = source.n_yielded
    print_summary(result, n_frames, t_load, t_run)
    print(f"  Frames dropped (realtime): {source.n_dropped}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save_kitti(str(out / "trajectory_kitti.txt"))
    result.save_tum(str(out / "trajectory_tum.txt"), timestamps=source.timestamps)
    if not args.skip_ply:
        result.save_ply(str(out / "map.ply"))
    save_timings_json(out / "timings.json", result, max(n_frames, 1), t_load, t_run)

    # Copy the GT log next to the trajectory for convenient evo comparison.
    gt_src = Path(args.spool_dir) / "ground_truth_tum.txt"
    if gt_src.exists():
        (out / "ground_truth_tum.txt").write_bytes(gt_src.read_bytes())

    print(f"\n  Outputs saved to {out}/")
    print(f"    trajectory_tum.txt   ({result.n_keyframes} poses)")
    if gt_src.exists():
        print("    ground_truth_tum.txt (copied from spool — evo_ape ... -as to compare)")


if __name__ == "__main__":
    main()
