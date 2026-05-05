"""
Keyframe selection via Lucas-Kanade optical flow.

A frame is promoted to a keyframe when the mean displacement of tracked
feature points from the last keyframe exceeds `min_disparity` pixels.
Mirrors the strategy used in VGGT-SLAM's frame_overlap.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class KeyframeSelectorConfig:
    # Minimum mean optical flow as a fraction of image width [0, 1].
    # A frame is a keyframe when mean displacement >= min_disparity_frac * W.
    # 0.15 means 15% of image width — works across resolutions and framerates.
    min_disparity_frac: float = 0.15

    # Maximum number of frames in a submap before forcing a new keyframe
    max_submap_size: int = 8

    # cv2.goodFeaturesToTrack parameters
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: float = 8.0

    # cv2.calcOpticalFlowPyrLK parameters
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class KeyframeResult:
    # Indices into the input frame list that were selected as keyframes
    indices: list[int] = field(default_factory=list)

    # Mean optical flow displacement for each frame (0.0 for the first frame)
    disparities: list[float] = field(default_factory=list)

    @property
    def n_keyframes(self) -> int:
        return len(self.indices)


# ── selector ──────────────────────────────────────────────────────────────────

class KeyframeSelector:
    """
    Selects keyframes from an ordered sequence of RGB images.

    The first frame is always a keyframe. Subsequent frames are promoted
    when their mean optical flow from the current keyframe exceeds
    `config.min_disparity`, or when `config.max_submap_size` is reached.
    """

    def __init__(self, config: KeyframeSelectorConfig | None = None):
        self.config = config or KeyframeSelectorConfig()

    def select(self, images: list[np.ndarray]) -> KeyframeResult:
        """
        Args:
            images: ordered list of HxWx3 uint8 RGB frames

        Returns:
            KeyframeResult with selected indices and per-frame disparities
        """
        if not images:
            return KeyframeResult()

        cfg = self.config
        result = KeyframeResult()

        # Frame 0 is always a keyframe
        ref_gray = _to_gray(images[0])
        ref_pts = _detect_points(ref_gray, cfg)
        result.indices.append(0)
        result.disparities.append(0.0)
        frames_since_keyframe = 0

        H, W = ref_gray.shape
        min_disparity_px = cfg.min_disparity_frac * W

        for i, img in enumerate(images[1:], start=1):
            cur_gray = _to_gray(img)
            disparity = 0.0

            if ref_pts is not None and len(ref_pts) > 0:
                disparity = _compute_disparity(ref_gray, cur_gray, ref_pts, cfg)

            result.disparities.append(disparity)
            frames_since_keyframe += 1

            is_keyframe = (
                disparity >= min_disparity_px
                or frames_since_keyframe >= cfg.max_submap_size
            )

            if is_keyframe:
                result.indices.append(i)
                ref_gray = cur_gray
                ref_pts = _detect_points(ref_gray, cfg)
                frames_since_keyframe = 0

        return result

    def select_paths(self, image_paths: list[str]) -> KeyframeResult:
        """Convenience wrapper that loads images from disk."""
        images = [_load_rgb(p) for p in image_paths]
        return self.select(images)


# ── internal helpers ──────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _detect_points(
    gray: np.ndarray, cfg: KeyframeSelectorConfig
) -> np.ndarray | None:
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=cfg.max_corners,
        qualityLevel=cfg.quality_level,
        minDistance=cfg.min_distance,
    )
    return pts  # shape (N, 1, 2) or None


def _compute_disparity(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    ref_pts: np.ndarray,
    cfg: KeyframeSelectorConfig,
) -> float:
    """Track ref_pts from ref_gray to cur_gray, return mean displacement."""
    cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        ref_gray,
        cur_gray,
        ref_pts,
        None,
        winSize=cfg.lk_win_size,
        maxLevel=cfg.lk_max_level,
    )

    if cur_pts is None or status is None:
        return 0.0

    good = status.ravel().astype(bool)
    if good.sum() == 0:
        return 0.0

    displacement = np.linalg.norm(
        cur_pts[good] - ref_pts[good], axis=-1
    )  # (M,)
    return float(displacement.mean())
