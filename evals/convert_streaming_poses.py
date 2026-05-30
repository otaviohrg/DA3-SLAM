#!/usr/bin/env python3
"""Convert DA3-Streaming camera_poses.txt to TUM RGB-D trajectory format.

DA3-Streaming writes one c2w 4x4 matrix (16 floats, row-major) per line,
one line per input image in the same sorted order the streamer processes them.
This script maps each pose back to the corresponding image timestamp (extracted
from the filename stem, as TUM images are named by their timestamps) and writes
the standard TUM format expected by evo_ape:

    timestamp tx ty tz qx qy qz qw
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description="Convert DA3-Streaming poses to TUM format")
    parser.add_argument("--poses_file", required=True, help="camera_poses.txt from DA3-Streaming")
    parser.add_argument("--image_dir", required=True, help="Directory of input RGB images")
    parser.add_argument("--output_file", required=True, help="Output trajectory_tum.txt path")
    args = parser.parse_args()

    # DA3-Streaming processes sorted .jpg then .png (same glob order)
    img_list = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg"))
        + glob.glob(os.path.join(args.image_dir, "*.png"))
    )
    if not img_list:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    poses = []
    with open(args.poses_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            if len(vals) != 16:
                raise ValueError(f"Expected 16 floats per line, got {len(vals)}: {line[:80]}")
            poses.append(np.array(vals).reshape(4, 4))

    if len(poses) != len(img_list):
        raise AssertionError(
            f"Pose count ({len(poses)}) does not match image count ({len(img_list)}). "
            "The streaming run may have been incomplete."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for img_path, c2w in zip(img_list, poses):
            try:
                ts = float(Path(img_path).stem)
            except ValueError:
                raise ValueError(
                    f"Cannot extract timestamp from filename: {img_path}. "
                    "TUM images must be named by their timestamp."
                )
            t = c2w[:3, 3]
            q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # (qx, qy, qz, qw)
            f.write(
                f"{ts:.6f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )

    print(f"Saved {len(poses)} poses to {args.output_file}")


if __name__ == "__main__":
    main()
