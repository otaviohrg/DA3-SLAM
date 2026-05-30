#!/usr/bin/env python3
"""Convert Replica traj.txt (c2w 4x4 per line) to TUM format.

Replica GT has one pose per frame, indexed 0..N-1. Since frame filenames
are frame000000.jpg (non-numeric), run_slam.py falls back to seq_idx / fps
for estimated trajectory timestamps. This script uses the same convention
so evo_ape can associate estimated keyframe poses to the correct GT frames.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_file", required=True, help="Replica traj.txt path")
    parser.add_argument("--output_file", required=True, help="Output TUM trajectory path")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Must match save_tum fps fallback (default: 30.0)")
    args = parser.parse_args()

    poses = []
    with open(args.traj_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            if len(vals) != 16:
                raise ValueError(f"Expected 16 floats per line, got {len(vals)}")
            poses.append(np.array(vals).reshape(4, 4))

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, c2w in enumerate(poses):
            t = c2w[:3, 3]
            q = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # (qx, qy, qz, qw)
            ts = i / args.fps
            f.write(
                f"{ts:.6f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )

    print(f"Saved {len(poses)} poses to {args.output_file}")


if __name__ == "__main__":
    main()
