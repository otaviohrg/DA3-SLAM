"""
Keyframe selection via Lucas-Kanade optical flow.

A frame is promoted to a keyframe when the mean displacement of tracked
feature points from the last keyframe exceeds `min_disparity_fraction` pixels.
Mirrors the strategy used in VGGT-SLAM's frame_overlap.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class KeyframeSelectorConfig:
    # Canonical values: config/default.yaml → keyframe.*

    # Minimum mean optical flow as a fraction of image width [0, 1].
    # A frame is a keyframe when mean displacement >= min_disparity_fraction * W.
    min_disparity_fraction: float

    # Maximum number of frames in a submap before forcing a new keyframe
    max_submap_size: int

    # cv2.goodFeaturesToTrack parameters
    max_corners: int
    quality_level: float
    min_distance: float
    block_size: int

    # cv2.calcOpticalFlowPyrLK parameters
    flow_window_size: tuple[int, int]
    flow_pyramid_levels: int
    flow_stop_criteria: tuple[int, int, float]  # (type, maxCount, epsilon)


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
    `config.min_disparity_fraction`, or when `config.max_submap_size` is reached.
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

        config = self.config
        result = KeyframeResult()

        # Frame 0 is always a keyframe
        reference_gray = _to_gray(images[0])
        reference_points = _detect_points(reference_gray, config)
        result.indices.append(0)
        result.disparities.append(0.0)
        frames_since_keyframe = 0

        W = reference_gray.shape[1]
        min_disparity_pixels = config.min_disparity_fraction * W

        for i, img in enumerate(images[1:], start=1):
            current_gray = _to_gray(img)
            disparity = 0.0

            if reference_points is not None and len(reference_points) > 0:
                disparity = _compute_disparity(reference_gray, current_gray,
                                               reference_points, config)

            result.disparities.append(disparity)
            frames_since_keyframe += 1

            is_keyframe = (
                disparity >= min_disparity_pixels
                or frames_since_keyframe >= config.max_submap_size
            )

            if is_keyframe:
                result.indices.append(i)
                reference_gray = current_gray
                reference_points = _detect_points(reference_gray, config)
                frames_since_keyframe = 0

        return result

    def select_paths(self, image_paths: list[str]) -> KeyframeResult:
        """Convenience wrapper that loads images from disk."""
        images = [_load_rgb(p) for p in image_paths]
        return self.select(images)


# ── online selector ───────────────────────────────────────────────────────────

class OnlineKeyframeSelector:
    """
    Stateful, frame-by-frame keyframe selector.

    Call step() for each incoming frame; returns True when the frame
    should be promoted to a keyframe. The first frame is always a keyframe.
    """

    def __init__(self, config: KeyframeSelectorConfig):
        self.config = config
        self._reference_gray: np.ndarray | None = None
        self._reference_points: np.ndarray | None = None
        self._frames_since_keyframe: int = 0

    def step(self, image: np.ndarray) -> bool:
        """
        Args:
            image: HxWx3 uint8 RGB frame

        Returns:
            True if this frame is a keyframe
        """
        config = self.config
        gray = _to_gray(image)

        if self._reference_gray is None:
            self._reference_gray = gray
            self._reference_points = _detect_points(gray, config)
            self._frames_since_keyframe = 0
            return True

        W = gray.shape[1]
        min_disparity_pixels = config.min_disparity_fraction * W
        disparity = 0.0
        if self._reference_points is not None and len(self._reference_points) > 0:
            disparity = _compute_disparity(self._reference_gray, gray,
                                           self._reference_points, config)

        self._frames_since_keyframe += 1
        is_keyframe = (
            disparity >= min_disparity_pixels
            or self._frames_since_keyframe >= config.max_submap_size
        )

        if is_keyframe:
            self._reference_gray = gray
            self._reference_points = _detect_points(gray, config)
            self._frames_since_keyframe = 0

        return is_keyframe

    def step_path(self, path: str) -> bool:
        """Convenience wrapper that loads an image from disk."""
        return self.step(_load_rgb(path))


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
    gray: np.ndarray, config: KeyframeSelectorConfig
) -> np.ndarray | None:
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        blockSize=config.block_size,
    )
    return pts  # shape (N, 1, 2) or None


def _compute_disparity(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    reference_points: np.ndarray,
    config: KeyframeSelectorConfig,
) -> float:
    """Track reference_points from reference_gray to current_gray, return mean displacement."""
    tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        current_gray,
        reference_points,
        None,
        winSize=config.flow_window_size,
        maxLevel=config.flow_pyramid_levels,
        criteria=config.flow_stop_criteria,
    )

    if tracked_points is None or status is None:
        return 0.0

    good = status.ravel().astype(bool)
    if good.sum() == 0:
        return 0.0

    displacement = np.linalg.norm(
        tracked_points[good] - reference_points[good], axis=-1
    )  # (M,)
    return float(displacement.mean())
