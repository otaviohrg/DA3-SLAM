"""
Visualize the DA3-SLAM point cloud map from map.ply.
Saves PNG renders from multiple viewpoints to the same directory.

Usage:
    python scripts/visualize_map.py outputs/slam/map.ply
    python scripts/visualize_map.py outputs/slam/map.ply --max_points 200000
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def load_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a PLY file (ASCII or binary little-endian).
    Returns:
        points : (N, 3) float32
        colors : (N, 3) uint8
    """
    with open(path, "rb") as f:
        # --- header ---
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        is_binary = False
        props = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("format binary_little_endian"):
                is_binary = True
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)

        prop_names = [p[1] for p in props]
        prop_types = [p[0] for p in props]

        if is_binary:
            # Build struct format
            fmt_map = {
                "float": "f", "float32": "f",
                "double": "d", "float64": "d",
                "uchar": "B", "uint8": "B",
                "int": "i", "uint": "I",
                "short": "h", "ushort": "H",
            }
            fmt = "<" + "".join(fmt_map[t] for t in prop_types)
            row_size = struct.calcsize(fmt)
            raw = f.read(n_vertices * row_size)
            rows = [struct.unpack_from(fmt, raw, i * row_size)
                    for i in range(n_vertices)]
            data = np.array(rows, dtype=np.float64)
        else:
            rows = []
            for _ in range(n_vertices):
                rows.append(list(map(float, f.readline().split())))
            data = np.array(rows, dtype=np.float64)

    xi = prop_names.index("x")
    yi = prop_names.index("y")
    zi = prop_names.index("z")
    ri = prop_names.index("red")
    gi = prop_names.index("green")
    bi = prop_names.index("blue")

    points = data[:, [xi, yi, zi]].astype(np.float32)
    colors = data[:, [ri, gi, bi]].astype(np.uint8)
    return points, colors


def subsample(points: np.ndarray, colors: np.ndarray, n: int):
    """Randomly keep at most n points (colors stay aligned)."""
    if len(points) <= n:
        return points, colors
    idx = np.random.choice(len(points), n, replace=False)
    return points[idx], colors[idx]


def render_views(
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
    n_points: int,
) -> None:
    """Save four 3D scatter views (perspective/top/front/side) of a subsampled
    cloud, plus a full-resolution 2-D top-down log-density histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, col = subsample(points, colors, n_points)
    rgb = col.astype(np.float32) / 255.0  # matplotlib wants [0,1]

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    dot_size = max(0.1, 2.0 * (5000 / len(pts)) ** 0.5)

    views = [
        ("perspective",  25,  45),
        ("top_down",     90,   0),
        ("front",         0,   0),
        ("side",          0,  90),
    ]

    for name, elev, azim in views:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=rgb, s=dot_size, linewidths=0, depthshade=True)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"DA3-SLAM Map  ({len(points):,} pts, showing {len(pts):,})  — {name}")
        ax.view_init(elev=elev, azim=azim)
        out = str(out_dir / f"map_{name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # ── top-down XZ heat density (2-D) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    h = ax.hist2d(
        points[:, 0], points[:, 2],
        bins=512,
        cmap="inferno",
        norm=matplotlib.colors.LogNorm(),
    )
    plt.colorbar(h[3], ax=ax, label="point density (log)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"DA3-SLAM Map — top-down density  ({len(points):,} pts)")
    ax.set_aspect("equal")
    out_density = str(out_dir / "map_density_topdown.png")
    fig.savefig(out_density, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_density}")


def main():
    """Load map.ply, print its spatial extent, and render views next to it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Path to map.ply")
    parser.add_argument(
        "--max_points", type=int, default=150_000,
        help="Max points to render in 3D scatter plots (default 150k)",
    )
    args = parser.parse_args()

    print(f"Loading {args.ply} ...")
    points, colors = load_ply(args.ply)
    print(f"Loaded {len(points):,} points")

    # Basic stats
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"  X: [{mins[0]:.2f}, {maxs[0]:.2f}] m")
    print(f"  Y: [{mins[1]:.2f}, {maxs[1]:.2f}] m")
    print(f"  Z: [{mins[2]:.2f}, {maxs[2]:.2f}] m")

    out_dir = Path(args.ply).parent
    render_views(points, colors, out_dir, args.max_points)


if __name__ == "__main__":
    main()
