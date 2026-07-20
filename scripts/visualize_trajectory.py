"""
Visualize SLAM trajectory from KITTI or TUM output files.
Saves PNG plots to the same directory as the input file.

Usage:
    python scripts/visualize_trajectory.py --tum outputs/slam/trajectory_tum.txt
    python scripts/visualize_trajectory.py --kitti outputs/slam/trajectory_kitti.txt
"""

import argparse
import numpy as np
from pathlib import Path


def load_tum(path: str) -> np.ndarray:
    """Return (N, 3) positions from TUM trajectory file."""
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(poses)


def load_kitti(path: str) -> np.ndarray:
    """Return (N, 3) positions from KITTI trajectory file."""
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            # 3x4 matrix, translation is last column: vals[3], vals[7], vals[11]
            poses.append([vals[3], vals[7], vals[11]])
    return np.array(poses)


def main():
    """Load a trajectory file and save 3D, top-down (XZ) and per-axis PNGs
    next to it."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tum",   help="TUM format trajectory file")
    group.add_argument("--kitti", help="KITTI format trajectory file")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.tum:
        path = args.tum
        positions = load_tum(path)
    else:
        path = args.kitti
        positions = load_kitti(path)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    out_dir = Path(path).parent

    # ── 3D trajectory ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, "-o", markersize=3, linewidth=1.5, color="royalblue")
    ax.scatter(x[0],  y[0],  z[0],  color="green", s=80, zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], z[-1], color="red",   s=80, zorder=5, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"DA3-SLAM Trajectory  ({len(positions)} keyframes)")
    ax.legend()
    p3d = str(out_dir / "trajectory_3d.png")
    fig.savefig(p3d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p3d}")

    # ── top-down XZ view ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, z, "-o", markersize=3, linewidth=1.5, color="royalblue")
    ax.scatter(x[0],  z[0],  color="green", s=80, zorder=5, label="Start")
    ax.scatter(x[-1], z[-1], color="red",   s=80, zorder=5, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Top-down view (XZ plane)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    p2d = str(out_dir / "trajectory_topdown.png")
    fig.savefig(p2d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p2d}")

    # ── per-axis vs frame ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (label, vals) in enumerate(zip("XYZ", [x, y, z])):
        axes[i].plot(vals, linewidth=1.5, color="royalblue")
        axes[i].set_ylabel(f"{label} (m)")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Keyframe index")
    axes[0].set_title("Camera position per axis")
    p_axes = str(out_dir / "trajectory_axes.png")
    fig.savefig(p_axes, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p_axes}")

    total = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    print(f"\nTrajectory: {len(positions)} keyframes, {total:.3f} m total length")


if __name__ == "__main__":
    main()
