"""
Log Spot's odometry to TUM format — FOR EVALUATION ONLY.

Runs ON the robot, alongside `spot_frame_sender.py`.

READ THIS BEFORE WIRING IT ANYWHERE ELSE
----------------------------------------
This data is a REFERENCE TRAJECTORY for scoring, not an input.  It must never
reach the SLAM pipeline.  The point of the robot test is to find out how the
*monocular* system behaves on real hardware; feeding it odometry would make it
a different system and the result would not be comparable to the TUM/UAS
benchmarks it is meant to extend.

(Fusing odometry is a defensible thing to build — this session measured that
per-submap metric SCALE is DA3's weakest axis: SE3 ATE is 3x worse than Sim3
even with zero submap boundaries, and boundary scale disagreement runs 4.8% on
TUM and 12.2% on UAS.  Spot's kinematic odometry would address exactly that.
But it is a separate system, and it should be evaluated as one.)

WHAT IT WRITES
--------------
TUM format, one line per sample:

    timestamp tx ty tz qx qy qz qw

Timestamps use the robot's wall clock — the same clock `spot_frame_sender.py`
stamps frames with — so the two logs align without any extra bookkeeping.

Poses are taken from Spot's `odom` frame, which is the drift-prone but locally
smooth kinematic estimate.  It is not survey-grade truth: over a long walk it
accumulates its own drift, so treat it as a strong reference over minutes
rather than absolute truth over an hour.  For a scale-free comparison use
Sim3 alignment (`evo_ape ... -as`), which is what every other DA3-SLAM
benchmark in this repo reports as the headline anyway.

Usage (on the robot):
    python scripts/spot_log_odometry.py --robot 192.168.80.3 \\
        --out odometry.tum --rate 20

    export BOSDYN_CLIENT_USERNAME=user BOSDYN_CLIENT_PASSWORD=...

Scoring afterwards:
    evo_ape tum odometry.tum outputs/spot/trajectory_tum.txt -as
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log Spot odometry to TUM format (evaluation reference only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--robot", required=True, help="Robot hostname or IP")
    p.add_argument("--out", default="odometry.tum", help="Output TUM file")
    p.add_argument("--rate", type=float, default=20.0,
                   help="Sampling rate (Hz).  Spot's state service updates "
                        "faster than this; 20 Hz is ample to interpolate "
                        "against keyframe timestamps")
    p.add_argument("--frame", choices=["odom", "vision"], default="odom",
                   help="Spot frame to log.  'odom' is the kinematic estimate "
                        "(smooth, drifts); 'vision' is Spot's own visual "
                        "odometry (less drift, but it is itself a vision "
                        "system — prefer 'odom' as an independent reference)")
    p.add_argument("--duration", type=float, default=None,
                   help="Stop after this many seconds (default: until Ctrl-C)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import bosdyn.client
        import bosdyn.client.util
        from bosdyn.client.robot_state import RobotStateClient
        from bosdyn.client.frame_helpers import (
            get_a_tform_b, BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME)
    except ImportError:
        raise SystemExit(
            "the Boston Dynamics SDK is required on the robot side:\n"
            "    pip install bosdyn-client bosdyn-api")

    sdk = bosdyn.client.create_standard_sdk("DA3SLAM-OdometryLogger")
    robot = sdk.create_robot(args.robot)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    parent = ODOM_FRAME_NAME if args.frame == "odom" else VISION_FRAME_NAME
    print(f"[odom] logging {parent} -> {BODY_FRAME_NAME} at {args.rate:g} Hz")
    print(f"[odom] EVALUATION REFERENCE ONLY — never feed this to the SLAM "
          f"pipeline")

    running = {"go": True}

    def stop(signum, frame):
        running["go"] = False
        print("\n[odom] stopping...")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / max(args.rate, 1e-3)
    deadline = time.time() + args.duration if args.duration else None
    n = 0
    with open(args.out, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write(f"# Spot {parent}->{BODY_FRAME_NAME}, evaluation reference only\n")
        while running["go"]:
            loop_start = time.time()
            if deadline and loop_start > deadline:
                break
            try:
                state = state_client.get_robot_state()
                tform = get_a_tform_b(
                    state.kinematic_state.transforms_snapshot,
                    parent, BODY_FRAME_NAME)
                if tform is None:
                    continue
                # Stamp with the robot's wall clock, matching the frame sender,
                # so the two logs share a time base with no extra alignment.
                ts = loop_start
                pos, rot = tform.position, tform.rotation
                f.write(f"{ts:.6f} {pos.x:.9f} {pos.y:.9f} {pos.z:.9f} "
                        f"{rot.x:.9f} {rot.y:.9f} {rot.z:.9f} {rot.w:.9f}\n")
                n += 1
                if n % 200 == 0:
                    f.flush()
                    print(f"[odom] {n} samples")
            except Exception as exc:
                print(f"[odom] sample failed ({exc}); continuing")
            sleep = period - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    print(f"[odom] wrote {n} samples to {args.out}")
    print(f"[odom] score with:  evo_ape tum {args.out} "
          f"outputs/spot/trajectory_tum.txt -as")


if __name__ == "__main__":
    main()
