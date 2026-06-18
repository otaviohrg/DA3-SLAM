"""
Shared helpers for the scripts/test_*.py smoke tests.

These are runnable scripts (not a pytest suite): each prints PASS/FAIL lines
and raises on the first failure so shell exit codes reflect the outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"FAIL: {label}")


def list_images(image_dir: str, limit: int | None = None) -> list[str]:
    """Sorted image paths in a directory, optionally capped to `limit`."""
    paths = sorted(
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths[:limit] if limit else paths


def load_rgb_images(paths: list[str]) -> list[np.ndarray]:
    """Load images as HxWx3 uint8 RGB arrays."""
    import cv2
    images = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return images


def save_ascii_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    """Write a coloured point cloud to an ASCII PLY file."""
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(col[0])} {int(col[1])} {int(col[2])}\n")
