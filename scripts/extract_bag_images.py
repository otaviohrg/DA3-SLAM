"""
Extract a camera topic from a ROS 2 bag into a directory of JPEGs.

Reads the bag directly (rosbag2 SequentialReader — as fast as disk, no
realtime replay) and writes one file per message named like the spool the
bridge produces, so timestamps are embedded:

    frame_<seq:08d>_<ts_ns>.jpg     (ts_ns = message header stamp)

Run on the host (system python3.12) after `source /opt/ros/jazzy/setup.bash`.
If the bag has no metadata.yaml:  ros2 bag reindex <bag_dir> -s sqlite3

    python3 scripts/extract_bag_images.py \\
        --bag drone1_20260618_204741 --out_dir /tmp/drone1_frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract a bag camera topic to JPEGs")
    ap.add_argument("--bag", required=True, help="Bag directory (contains *.db3)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topic", default="/camera_frames_0")
    ap.add_argument("--storage", default="sqlite3", choices=["sqlite3", "mcap"])
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--max_frames", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id=args.storage),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    bridge = CvBridge()
    params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    n = 0
    while reader.has_next():
        if args.max_frames is not None and n >= args.max_frames:
            break
        _topic, data, _ts = reader.read_next()
        msg = deserialize_message(data, Image)
        bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ts_ns = stamp_to_ns(msg.header.stamp)
        cv2.imwrite(str(out / f"frame_{n:08d}_{ts_ns}.jpg"), bgr, params)
        n += 1

    print(f"extracted {n} frames from {args.topic} → {out}")


if __name__ == "__main__":
    main()
