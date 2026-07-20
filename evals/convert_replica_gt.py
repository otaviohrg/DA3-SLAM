#!/usr/bin/env python3
"""Convert Replica traj.txt (c2w 4x4 per line) to TUM format.

Replica GT has one pose per frame, indexed 0..N-1. Since frame filenames
are frame000000.jpg (non-numeric), run_slam.py falls back to seq_idx / fps
for estimated trajectory timestamps. This script uses the same convention
so evo_ape can associate estimated keyframe poses to the correct GT frames.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# The TUM trajectory writer lives in scripts/euroc_common.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from euroc_common import write_tum_trajectory  # noqa: E402


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

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tum_trajectory([(i / args.fps, c2w) for i, c2w in enumerate(poses)],
                         out_path)
    print(f"Saved {len(poses)} poses to {out_path}")


if __name__ == "__main__":
    main()
