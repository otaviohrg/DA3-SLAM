"""
Keyframe selection via Lucas-Kanade optical flow.

A frame is promoted to a keyframe when the mean displacement of feature
points tracked from the last keyframe exceeds
`min_disparity_fraction × image width`, or when `max_submap_size` frames
have passed without one.  Mirrors the strategy used in VGGT-SLAM's
frame_overlap.py.

OnlineKeyframeSelector is the stateful frame-by-frame selector used by the
pipeline; KeyframeSelector is a thin batch wrapper around it for scripts
and offline analysis.

SegmentKeyframeSelector is an alternative policy (selected via
KeyframeSelectorConfig.selection_mode == "segment") that controls keyframe
*density* over fixed-length temporal segments instead of making per-frame
decisions, following Choi et al., "Revisiting Keyframe Selection in
Learning-Based Dense Monocular SLAM" (IEEE Access 2026).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

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

    # Downsample factor applied before running optical flow.
    # Higher = faster but coarser motion estimate. 4 is a good default.
    flow_downsample_factor: int = 4

    # ── selection policy ────────────────────────────────────────────────────
    # "disparity": frame-level threshold (VGGT-SLAM-style; OnlineKeyframeSelector).
    # "segment":   segment-level density control (SegmentKeyframeSelector),
    #              following Choi et al., "Revisiting Keyframe Selection in
    #              Learning-Based Dense Monocular SLAM" (IEEE Access 2026).
    selection_mode: str = "disparity"

    # Segment mode: number of consecutive frames accumulated into one
    # non-overlapping temporal segment (N_S in the paper).
    segment_length: int = 128

    # Segment mode: accumulated mean optical-flow displacement (px) over a
    # segment above which the denser stride is used (tau_seg in the paper).
    segment_disparity_threshold: float = 650.0

    # Segment mode: (dense, sparse) sampling strides (a, b in the paper).
    # stride = a if accumulated disparity > segment_disparity_threshold else b.
    segment_strides: tuple[int, int] = (8, 16)

    # ── blur gating ─────────────────────────────────────────────────────────
    # Motion blur poisons everything downstream at once: DA3 poses (submap
    # boundary breaks), retrieval descriptors, and re-inference confidence.
    # Fast handheld motion still yields sharp frames at direction reversals
    # and micro-pauses — prefer those.

    # Segment mode: replace each stride-picked keyframe with the sharpest
    # frame (variance of Laplacian) within ± this many buffered frames.
    # 0 = off.  The live demo uses 2.
    sharpness_window: int = 0

    # Disparity mode: defer promoting a frame to keyframe when its sharpness
    # is below this fraction of the recent median (the max_submap_size
    # force-keyframe still applies, so a long blurry stretch cannot stall
    # the pipeline).  0 = off.
    min_sharpness_ratio: float = 0.0


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


# ── online selector ───────────────────────────────────────────────────────────

class OnlineKeyframeSelector:
    """
    Stateful, frame-by-frame keyframe selector.

    Call step() for each incoming frame; returns True when the frame should
    be promoted to a keyframe.  The first frame is always a keyframe.
    The disparity measured for the most recent frame is available as
    `last_disparity` (0.0 for the first frame).
    """

    def __init__(self, config: KeyframeSelectorConfig):
        self.config = config
        self.last_disparity: float = 0.0
        self._reference_gray: np.ndarray | None = None
        self._reference_points: np.ndarray | None = None
        self._frames_since_keyframe: int = 0
        self._sharpness_history: list[float] = []

    def step(self, image: np.ndarray) -> bool:
        """
        Args:
            image: HxWx3 uint8 RGB frame

        Returns:
            True if this frame is a keyframe
        """
        config = self.config
        gray = _to_flow_gray(image, config.flow_downsample_factor)

        if self._reference_gray is None:
            self._set_reference(gray)
            self.last_disparity = 0.0
            return True

        disparity = 0.0
        if self._reference_points is not None and len(self._reference_points) > 0:
            disparity = _compute_disparity(self._reference_gray, gray,
                                           self._reference_points, config)
        self.last_disparity = disparity
        self._frames_since_keyframe += 1

        # Threshold is in downsampled-image pixels (gray is already downsampled,
        # so the fraction-of-width semantics are preserved).
        min_disparity_pixels = config.min_disparity_fraction * gray.shape[1]
        forced = self._frames_since_keyframe >= config.max_submap_size
        is_keyframe = disparity >= min_disparity_pixels or forced

        # Blur gate: defer a disparity-triggered keyframe when the frame is
        # much blurrier than recent frames (motion blur); a later, sharper
        # frame will trigger instead.  The force-keyframe path is exempt so
        # a long blurry stretch cannot stall the pipeline.
        ratio = config.min_sharpness_ratio
        if ratio > 0:
            sharpness = _sharpness(gray)
            self._sharpness_history.append(sharpness)
            del self._sharpness_history[:-30]
            if (is_keyframe and not forced
                    and sharpness < ratio * float(np.median(self._sharpness_history))):
                return False

        if is_keyframe:
            self._set_reference(gray)
        return is_keyframe

    def _set_reference(self, gray: np.ndarray) -> None:
        """Make `gray` (already downsampled) the new tracking reference."""
        self._reference_gray = gray
        self._reference_points = _detect_points(gray, self.config)
        self._frames_since_keyframe = 0


# ── segment selector ──────────────────────────────────────────────────────────

class SegmentKeyframeSelector:
    """
    Segment-level keyframe selector (Choi et al., IEEE Access 2026).

    Instead of deciding frame-by-frame whether a frame is a keyframe, the stream
    is partitioned into non-overlapping segments of `segment_length` (N_S) frames.
    Within each segment the accumulated mean optical-flow displacement
    `D_seg = sum_i d_{i,i+1}` is used as a motion proxy to pick a single regular
    sampling stride from `segment_strides = (a, b)`:

        stride = a  if D_seg > segment_disparity_threshold  else  b

    Keyframes are then sampled at that fixed stride, always including the last
    frame of the segment (which acts as the bridge/anchor to the next segment).

    Unlike OnlineKeyframeSelector, `step()` buffers frames and only returns
    keyframes at a segment boundary; call `flush()` to drain the final partial
    segment.  Each returned keyframe is a (label, image, seq_idx) tuple.
    """

    def __init__(self, config: KeyframeSelectorConfig):
        self.config = config
        self.last_disparity: float = 0.0
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._buffer: list[tuple[str, np.ndarray, int]] = []
        self._disparity_accum: float = 0.0

    def step(self, image: np.ndarray, seq_idx: int, label: str) \
            -> list[tuple[str, np.ndarray, int]]:
        """
        Buffer one incoming frame.

        Returns the keyframes selected for the segment when it just completed,
        otherwise an empty list.
        """
        config = self.config

        # Fixed-stride fast path: when both strides are equal the disparity
        # accumulator can never change the stride choice, so optical flow is
        # pure overhead — buffer the frame and emit on the segment boundary.
        # (Used by the keyframe-density sweeps for deterministic density.)
        if config.segment_strides[0] == config.segment_strides[1]:
            self._buffer.append((label, image, seq_idx))
            if len(self._buffer) >= config.segment_length:
                return self._emit_segment()
            return []

        gray = _to_flow_gray(image, config.flow_downsample_factor)
        is_segment_start = len(self._buffer) == 0

        disparity = 0.0
        if self._prev_points is not None and len(self._prev_points) > 0:
            disparity = _compute_disparity(self._prev_gray, gray,
                                           self._prev_points, config)
        self.last_disparity = disparity
        # Accumulate displacement only between frames within the same segment
        # (the cross-boundary pair does not contribute to D_seg).
        if not is_segment_start:
            self._disparity_accum += disparity

        self._buffer.append((label, image, seq_idx))
        self._prev_gray = gray
        self._prev_points = _detect_points(gray, config)

        if len(self._buffer) >= config.segment_length:
            return self._emit_segment()
        return []

    def flush(self) -> list[tuple[str, np.ndarray, int]]:
        """Emit keyframes for the final, possibly partial, segment."""
        if not self._buffer:
            return []
        return self._emit_segment()

    def _emit_segment(self) -> list[tuple[str, np.ndarray, int]]:
        config = self.config
        dense_stride, sparse_stride = config.segment_strides
        stride = (dense_stride
                  if self._disparity_accum > config.segment_disparity_threshold
                  else sparse_stride)
        stride = max(1, int(stride))

        length = len(self._buffer)
        # Regular sampling that always lands on the last frame of the segment,
        # so consecutive segments stay connected through a shared frame.
        local_indices = list(range(length - 1, -1, -stride))[::-1]

        # Blur gating: swap each stride-picked frame for the sharpest frame in
        # its neighbourhood.  Fast motion smears most frames, but sharp ones
        # survive at direction reversals and micro-pauses — those make far
        # better DA3 inputs, descriptors and anchors.
        window = max(0, int(config.sharpness_window))
        if window > 0:
            local_indices = self._sharpest_substitutes(local_indices, window)

        selected = [self._buffer[i] for i in local_indices]

        self._buffer = []
        self._disparity_accum = 0.0
        return selected

    def _buffered_sharpness(self, index: int, cache: dict[int, float]) -> float:
        """Sharpness of the buffered frame at `index`, memoised in `cache`
        (neighbourhoods of consecutive stride picks overlap)."""
        if index not in cache:
            gray = _to_flow_gray(self._buffer[index][1],
                                 self.config.flow_downsample_factor)
            cache[index] = _sharpness(gray)
        return cache[index]

    def _sharpest_substitutes(self, local_indices: list[int], window: int) -> list[int]:
        """Replace each index with the sharpest buffered frame within ±window,
        keeping the result strictly increasing (no duplicate keyframes)."""
        cache: dict[int, float] = {}
        length = len(self._buffer)
        substituted: list[int] = []
        previous = -1
        for index in local_indices:
            low = max(index - window, previous + 1)
            high = min(index + window, length - 1)
            if low > high:
                continue
            best = max(range(low, high + 1),
                       key=lambda i: self._buffered_sharpness(i, cache))
            substituted.append(best)
            previous = best
        return substituted


# ── replay selector (frozen keyframes) ────────────────────────────────────────

class ReplayKeyframeSelector:
    """Deterministic replay of a pre-recorded keyframe list.

    Emits a frame as a keyframe iff its seq_idx is in the frozen list,
    bypassing optical-flow selection entirely.  Two runs replaying the same
    list therefore see byte-identical keyframes, so any difference in the
    result comes from the network config, not from selection drift — the
    controlled-comparison foundation for the resolution / model-size sweeps.

    Shares the list-returning ``step()`` / ``flush()`` interface with
    SegmentKeyframeSelector so the frontend treats them uniformly.
    """

    def __init__(self, seq_indices: Iterable[int]):
        self._wanted: set[int] = {int(i) for i in seq_indices}
        self.last_disparity: float = 0.0

    def step(self, image: np.ndarray, seq_idx: int, label: str) \
            -> list[tuple[str, np.ndarray, int]]:
        if seq_idx in self._wanted:
            return [(label, image, seq_idx)]
        return []

    def flush(self) -> list[tuple[str, np.ndarray, int]]:
        return []


def save_keyframe_list(
    path: str | Path, keyframes: Iterable[tuple[int, str]]
) -> None:
    """Write a frozen keyframe list (one ``seq_idx<TAB>label`` line per
    keyframe).  The label is provenance only — replay keys on seq_idx."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# DA3-SLAM frozen keyframe list — one selected keyframe per line.\n")
        f.write("# seq_idx<TAB>label  (replay keys on seq_idx; label is provenance)\n")
        for seq_idx, label in keyframes:
            f.write(f"{int(seq_idx)}\t{label}\n")


def load_keyframe_list(path: str | Path) -> list[int]:
    """Read the seq_idxs from a file written by save_keyframe_list.

    Tolerates blank lines and ``#`` comments; each remaining line's first
    whitespace-separated token is the seq_idx.
    """
    seq_indices: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seq_indices.append(int(line.split()[0]))
    return seq_indices


# ── batch selector ────────────────────────────────────────────────────────────

class KeyframeSelector:
    """
    Batch wrapper around OnlineKeyframeSelector for scripts and analysis.

    Selects keyframes from an ordered sequence of RGB images and records the
    per-frame disparities.
    """

    def __init__(self, config: KeyframeSelectorConfig):
        self.config = config

    def select(self, images: list[np.ndarray]) -> KeyframeResult:
        """
        Args:
            images: ordered list of HxWx3 uint8 RGB frames

        Returns:
            KeyframeResult with selected indices and per-frame disparities
        """
        online = OnlineKeyframeSelector(self.config)
        result = KeyframeResult()
        for i, image in enumerate(images):
            if online.step(image):
                result.indices.append(i)
            result.disparities.append(online.last_disparity)
        return result

    def select_paths(self, image_paths: list[str]) -> KeyframeResult:
        """Convenience wrapper that loads images from disk."""
        return self.select([_load_rgb(p) for p in image_paths])


# ── internal helpers ──────────────────────────────────────────────────────────

def _to_flow_gray(image: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Convert to grayscale and downsample for cheaper optical flow."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    if downsample_factor > 1:
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (w // downsample_factor, h // downsample_factor),
                          interpolation=cv2.INTER_AREA)
    return gray


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap blur measure (higher =
    sharper).  Computed on the downsampled flow gray, so it costs ~nothing."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _load_rgb(path: str) -> np.ndarray:
    """Read an image from disk as HxWx3 uint8 RGB."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _detect_points(
    gray: np.ndarray, config: KeyframeSelectorConfig
) -> np.ndarray | None:
    """Shi-Tomasi corners to track from `gray`; None when the image has none
    (e.g. textureless frames — the caller then reports zero disparity)."""
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        blockSize=config.block_size,
    )
    return points  # shape (N, 1, 2) or None


def _compute_disparity(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    reference_points: np.ndarray,
    config: KeyframeSelectorConfig,
) -> float:
    """Track reference_points into current_gray; return mean displacement (px)."""
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
