"""
Frame sender — runs ON the robot (Spot payload PC or CORE I/O), not on the GPU host.

WHY THIS EXISTS
---------------
DA3-SLAM needs `nested-giant` to reach usable accuracy (the whole small->giant
ladder is ~3x worse on TUM), and that model peaks at ~7 GB of GPU memory even
with bf16 backbones.  That does not fit a Jetson-class payload, so inference
runs on a workstation and the robot only ships pixels.  A USB3 tether to a
walking robot is not an option, so frames go over the network.

The SLAM system itself is unchanged: the receiver hands
`(RGB, seq_idx, label)` tuples to `DA3SLAM.run_stream()` exactly as
`RealSenseFrameSource` does locally.

WIRE FORMAT (deliberately dependency-free — plain TCP, no ZMQ/ROS)
------------------------------------------------------------------
Each frame is a 20-byte little-endian header followed by JPEG bytes:

    magic   4s   b"DA3F"
    seq     I    capture-order index, strictly increasing, gaps allowed
    stamp   d    capture timestamp (seconds, time.time() on the robot)
    nbytes  I    length of the JPEG payload that follows

TIMESTAMPS ARE TAKEN AT CAPTURE, NOT AT RECEIPT.  Network jitter would
otherwise corrupt the keyframe timing and make the trajectory unscoreable
against any external reference.

BACKPRESSURE
------------
If the link or the GPU cannot keep up, the sender DROPS the oldest queued
frames rather than buffering without bound.  A SLAM front end wants recent
frames, not a growing backlog of stale ones; keyframe selection copes with gaps
because `seq_idx` only has to be unique and increasing, not contiguous.

Usage (on the robot):
    python scripts/spot_frame_sender.py --host 0.0.0.0 --port 5555
    python scripts/spot_frame_sender.py --host 0.0.0.0 --port 5555 --exposure 80
    python scripts/spot_frame_sender.py --source v4l2 --device 0     # any UVC cam
"""

from __future__ import annotations

import argparse
import queue
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

HEADER = struct.Struct("<4sIdI")
MAGIC = b"DA3F"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream RGB frames from the robot to the SLAM host",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address; the GPU host connects to this")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--source", choices=["realsense", "v4l2"], default="realsense",
                   help="realsense = pyrealsense2 colour stream; v4l2 = any UVC "
                        "camera through OpenCV")
    p.add_argument("--device", type=int, default=0, help="v4l2 device index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", default=None, help="RealSense serial number")
    p.add_argument("--exposure", type=float, default=None,
                   help="Manual colour exposure (D4xx: tenths of a ms).  A "
                        "walking robot has gait-induced motion blur, and short "
                        "exposure is the only real defence — the sharpness gate "
                        "downstream can only discard blurred frames, not "
                        "un-blur them.  None = auto-exposure with priority off")
    p.add_argument("--jpeg_quality", type=int, default=90,
                   help="90 is visually lossless enough for DA3; drop to ~75 "
                        "only if the link is the bottleneck")
    p.add_argument("--queue", type=int, default=4,
                   help="Frames buffered before the oldest are dropped")
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


# ── capture backends ──────────────────────────────────────────────────────────

def realsense_frames(args):
    """Yield (bgr, capture_timestamp) from a RealSense colour stream."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    sensor = profile.get_device().first_color_sensor()
    if args.exposure is not None:
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(args.exposure))
        print(f"[sender] manual exposure {args.exposure}")
    else:
        # Auto-exposure PRIORITY off: the driver may not drop below the
        # configured fps to gather light, which caps exposure at the frame
        # budget and keeps blur bounded at constant frame rate.
        try:
            sensor.set_option(rs.option.auto_exposure_priority, 0)
            print("[sender] auto-exposure, priority off")
        except Exception as exc:
            print(f"[sender] could not set auto-exposure priority: {exc}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            colour = frames.get_color_frame()
            if not colour:
                continue
            # Device timestamp is milliseconds since an arbitrary epoch; anchor
            # it to wall clock once so the stream is comparable to anything
            # else logged on the robot.
            yield np.asanyarray(colour.get_data()), time.time()
    finally:
        pipeline.stop()


def v4l2_frames(args):
    """Yield (bgr, capture_timestamp) from any OpenCV-readable camera."""
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)     # manual on most UVC drivers
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    if not cap.isOpened():
        raise SystemExit(f"could not open v4l2 device {args.device}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            yield bgr, time.time()
    finally:
        cap.release()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    frames = (realsense_frames if args.source == "realsense" else v4l2_frames)(args)

    outbox: queue.Queue = queue.Queue(maxsize=args.queue)
    stop = threading.Event()
    dropped = 0

    def capture() -> None:
        nonlocal dropped
        seq = 0
        for bgr, stamp in frames:
            if stop.is_set() or (args.max_frames and seq >= args.max_frames):
                break
            ok, buf = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ok:
                continue
            item = (seq, stamp, buf.tobytes())
            seq += 1
            try:
                outbox.put_nowait(item)
            except queue.Full:
                # Drop the OLDEST: a SLAM front end wants recent frames, and an
                # unbounded backlog would make the trajectory lag reality.
                try:
                    outbox.get_nowait()
                    dropped += 1
                except queue.Empty:
                    pass
                try:
                    outbox.put_nowait(item)
                except queue.Full:
                    dropped += 1
        outbox.put(None)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[sender] listening on {args.host}:{args.port} "
          f"({args.source} {args.width}x{args.height}@{args.fps})")
    print("[sender] waiting for the SLAM host to connect...")
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[sender] connected: {addr}")

    threading.Thread(target=capture, daemon=True, name="capture").start()

    sent = 0
    t0 = time.time()
    try:
        while True:
            item = outbox.get()
            if item is None:
                break
            seq, stamp, payload = item
            conn.sendall(HEADER.pack(MAGIC, seq, stamp, len(payload)) + payload)
            sent += 1
            if sent % 100 == 0:
                dt = time.time() - t0
                print(f"[sender] sent {sent}  dropped {dropped}  "
                      f"{sent/dt:.1f} fps out")
    except (BrokenPipeError, ConnectionResetError):
        print("[sender] SLAM host disconnected")
    except KeyboardInterrupt:
        print("\n[sender] interrupted")
    finally:
        stop.set()
        try:
            conn.close()
        except OSError:
            pass
        server.close()
        print(f"[sender] done: {sent} frames sent, {dropped} dropped")


if __name__ == "__main__":
    main()
