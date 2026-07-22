"""
Map-detail proxy for the compute-reduction sweep (plan Step 0g).

Trajectory ATE can stay flat while the *dense map* degrades as resolution or
model size drops — that dissociation is the study's publishable nuance, so we
need a cheap map-quality number alongside ATE:

  * point count            — free; how many confident points the config keeps.
  * Chamfer distance to a   — symmetric mean nearest-neighbour distance between
    reference cloud           a config's cloud and the 504/Giant cloud used as
                              pseudo-ground-truth.  Rises as the map coarsens.

Reads the binary PLY that da3_slam.slam.SLAMResult.save_ply writes
(little-endian: float32 x,y,z + uchar r,g,b per vertex).  CLI:

    python scripts/map_detail.py --ref outputs/sweep/ref.ply --cloud a.ply b.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_ply_points(path: str | Path) -> np.ndarray:
    """Read (N, 3) float32 XYZ from our binary PLY (ignores the RGB bytes)."""
    path = Path(path)
    with open(path, "rb") as f:
        # Header is ASCII, terminated by a line == "end_header".
        n_vertices = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header found")
            text = line.decode("ascii", "replace").strip()
            if text.startswith("element vertex"):
                n_vertices = int(text.split()[-1])
            elif text == "end_header":
                break
        # Each vertex: 12 bytes xyz (float32) + 3 bytes rgb (uint8) = 15 bytes.
        raw = np.frombuffer(f.read(n_vertices * 15), dtype=np.uint8)
    if raw.size != n_vertices * 15:
        raise ValueError(f"{path}: truncated body "
                         f"({raw.size} bytes, expected {n_vertices * 15})")
    xyz_bytes = raw.reshape(n_vertices, 15)[:, :12].copy()
    return xyz_bytes.view(np.float32).reshape(n_vertices, 3)


def _nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance from each src point to the dst set."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(dst).query(src, k=1)[0]
    except ImportError:  # brute-force fallback (chunked to bound memory)
        out = np.empty(len(src), dtype=np.float64)
        for i in range(0, len(src), 4096):
            block = src[i:i + 4096]
            d = np.linalg.norm(block[:, None, :] - dst[None, :, :], axis=-1)
            out[i:i + 4096] = d.min(axis=1)
        return out


def chamfer_distance(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    max_points: int = 50_000,
    seed: int = 0,
) -> float:
    """Symmetric mean nearest-neighbour distance between two point clouds.

    Both clouds are randomly subsampled to at most `max_points` (Chamfer is
    O(N log N) with a KD-tree, and the clouds are large); the RNG is seeded so
    the proxy is reproducible across configs.  Returns NaN if either cloud is
    empty.
    """
    if len(cloud_a) == 0 or len(cloud_b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)

    def subsample(cloud: np.ndarray) -> np.ndarray:
        if len(cloud) <= max_points:
            return cloud
        idx = rng.choice(len(cloud), size=max_points, replace=False)
        return cloud[idx]

    a, b = subsample(cloud_a), subsample(cloud_b)
    return float(0.5 * (_nn_distances(a, b).mean() + _nn_distances(b, a).mean()))


def chamfer_vs_reference(cloud_path: str | Path, ref_path: str | Path,
                         **kwargs) -> float:
    """Chamfer distance between a config's PLY and a reference (pseudo-GT) PLY."""
    return chamfer_distance(load_ply_points(cloud_path),
                            load_ply_points(ref_path), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map-detail proxy: point count + Chamfer vs a reference cloud")
    parser.add_argument("--ref", required=True,
                        help="Reference PLY (pseudo-GT, e.g. the 504/Giant cloud)")
    parser.add_argument("--cloud", nargs="+", required=True,
                        help="One or more PLYs to score against the reference")
    parser.add_argument("--max_points", type=int, default=50_000)
    args = parser.parse_args()

    ref = load_ply_points(args.ref)
    print(f"reference {args.ref}: {len(ref):,} points")
    for cloud_path in args.cloud:
        pts = load_ply_points(cloud_path)
        cd = chamfer_distance(pts, ref, max_points=args.max_points)
        print(f"  {cloud_path}: {len(pts):,} points  chamfer={cd:.5f}")


if __name__ == "__main__":
    main()
