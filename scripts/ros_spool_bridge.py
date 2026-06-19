#!/usr/bin/env python3
"""
Host-side ROS → spool bridge for real-time DA3-SLAM.

DA3-SLAM runs inside a CUDA container that has *no* ROS, while ROS jazzy and
the rosbag live on the host.  This node lives on the **host** (system
python3.12, rclpy + cv_bridge) and bridges the two over a plain bind-mounted
**spool directory** — no networking, no ROS inside the container.

    ┌─ host ─────────────────────────┐        ┌─ container ──────────────────┐
    │ ros2 bag play  →  this node    │        │ scripts/run_slam_ros.py      │
    │   /camera_frames_0  ─┐         │        │   tails spool, streams into  │
    │   /ground_truth/odom │  spool/ │ <════> │   DA3SLAM.run_stream()       │
    └──────────────────────┴─────────┘  bind  └──────────────────────────────┘
                                         mount

What it writes into --spool_dir:
  frame_<seq:08d>_<ts_ns>.jpg   one accepted camera frame (atomic rename)
  ground_truth_tum.txt          GT odometry as TUM (ts tx ty tz qx qy qz qw)
  DONE                          sentinel: stream ended (bag finished / Ctrl-C)

Real-time policy: incoming frames go onto a bounded drop-oldest queue, so if
the consumer (DA3 inference) falls behind the camera rate, the *oldest* unwritten
frames are discarded rather than the bridge lagging behind wall-clock.

Run on the host (after `source /opt/ros/jazzy/setup.bash`):
    python3 scripts/ros_spool_bridge.py --spool_dir /tmp/da3_spool --clean
    # then, in another terminal:
    ros2 bag play drone2_20260618_195702
"""

from __future__ import annotations

import argparse
import collections
import os
import threading
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def stamp_to_ns(stamp) -> int:
    """builtin_interfaces/Time → integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class SpoolBridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("da3_spool_bridge")
        self.spool = Path(args.spool_dir)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = int(args.jpeg_quality)
        self.idle_timeout = float(args.idle_timeout)
        self.bridge = CvBridge()

        # Drop-oldest queue: (ts_ns, bgr image). maxlen discards the oldest
        # pending frame when the writer can't keep up — see module docstring.
        self._lock = threading.Lock()
        self._dq: collections.deque = collections.deque(maxlen=int(args.queue_size))
        self._stop = threading.Event()

        self._n_received = 0
        self._n_written = 0
        self._n_dropped = 0
        self._last_image_t = time.monotonic()  # wall time of last received frame

        # GT log (TUM format). Header comment matches trajectory_tum.txt.
        self._gt_file = open(self.spool / "ground_truth_tum.txt", "w", buffering=1)
        self._gt_file.write("# timestamp tx ty tz qx qy qz qw\n")
        self._n_gt = 0

        # Subscribe BEST_EFFORT (sensor_data) so we are QoS-compatible with both
        # reliable and best-effort publishers / bag playback.
        self.create_subscription(
            Image, args.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, args.odom_topic, self._on_odom, qos_profile_sensor_data
        )

        self._writer = threading.Thread(target=self._writer_loop, name="spool-writer",
                                        daemon=True)
        self._writer.start()

        self.get_logger().info(
            f"spool={self.spool}  image={args.image_topic}  odom={args.odom_topic}  "
            f"queue_size={args.queue_size}  idle_timeout={self.idle_timeout}s"
        )

    # ── subscriptions ───────────────────────────────────────────────────────
    def _on_image(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - log and skip a bad frame
            self.get_logger().warning(f"imgmsg_to_cv2 failed: {exc}")
            return
        ts_ns = stamp_to_ns(msg.header.stamp) or self.get_clock().now().nanoseconds
        with self._lock:
            if len(self._dq) == self._dq.maxlen:
                self._n_dropped += 1  # this append will evict the oldest
            self._dq.append((ts_ns, bgr))
            self._n_received += 1
        self._last_image_t = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        ts = stamp_to_ns(msg.header.stamp) / 1e9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # ROS quaternion is (x, y, z, w) — same component order as TUM.
        self._gt_file.write(
            f"{ts:.9f} {p.x:.9f} {p.y:.9f} {p.z:.9f} "
            f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"
        )
        self._n_gt += 1

    # ── writer thread ───────────────────────────────────────────────────────
    def _writer_loop(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        while True:
            item = None
            with self._lock:
                if self._dq:
                    item = self._dq.popleft()  # oldest pending first → preserves order
            if item is None:
                if self._stop.is_set():
                    break
                time.sleep(0.002)
                continue
            ts_ns, bgr = item
            name = f"frame_{self._n_written:08d}_{ts_ns}.jpg"
            # Encode to JPEG bytes (codec from the ".jpg" hint, not the temp
            # filename), write to a temp file, then atomically rename so the
            # consumer only ever sees a complete *.jpg.
            ok, buf = cv2.imencode(".jpg", bgr, params)
            if not ok:
                self.get_logger().warning(f"jpeg encode failed for {name}")
                continue
            tmp = self.spool / (name + ".tmp")
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, self.spool / name)  # atomic: file appears complete
            self._n_written += 1

        # Drained and stopping: signal end-of-stream to the consumer.
        (self.spool / "DONE").write_text(
            f"received={self._n_received} written={self._n_written} "
            f"dropped={self._n_dropped} gt={self._n_gt}\n"
        )
        self.get_logger().info(
            f"DONE  received={self._n_received} written={self._n_written} "
            f"dropped={self._n_dropped} gt={self._n_gt}"
        )

    # ── lifecycle ───────────────────────────────────────────────────────────
    def idle_elapsed(self) -> float:
        return time.monotonic() - self._last_image_t

    def shutdown(self) -> None:
        self._stop.set()
        self._writer.join(timeout=10.0)
        self._gt_file.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ROS → spool bridge for DA3-SLAM")
    ap.add_argument("--spool_dir", default="/tmp/da3_spool",
                    help="Directory shared (bind-mounted) with the SLAM container")
    ap.add_argument("--image_topic", default="/camera_frames_0")
    ap.add_argument("--odom_topic", default="/ground_truth/odom")
    ap.add_argument("--queue_size", type=int, default=4,
                    help="Drop-oldest buffer depth between ROS and disk writer")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--idle_timeout", type=float, default=5.0,
                    help="Seconds with no new frame after which the stream is "
                         "declared finished (writes DONE). <=0 disables.")
    ap.add_argument("--clean", action="store_true",
                    help="Remove stale frames / DONE / GT log from the spool dir first")
    return ap.parse_args()


def clean_spool(spool: Path) -> None:
    for p in spool.glob("frame_*.jpg"):
        p.unlink()
    for p in spool.glob("frame_*.jpg.tmp"):
        p.unlink()
    for name in ("DONE", "ground_truth_tum.txt"):
        (spool / name).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.clean:
        spool = Path(args.spool_dir)
        spool.mkdir(parents=True, exist_ok=True)
        clean_spool(spool)

    rclpy.init()
    node = SpoolBridge(args)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if (args.idle_timeout > 0 and node._n_received > 0
                    and node.idle_elapsed() > args.idle_timeout):
                node.get_logger().info(
                    f"No frames for {args.idle_timeout}s — finishing.")
                break
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted — finishing.")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
