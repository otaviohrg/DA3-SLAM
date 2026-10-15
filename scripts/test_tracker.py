"""
Smoke test for the per-frame tracker (da3_slam/frontend/tracker.py).

Runs without a GPU or DA3: a textured plane at known depth is translated in
front of a synthetic camera, so both the correspondences and the answer are
known.  Checks, in order:

  1. optical flow carries tracks across frames and keeps their ids
  2. a published reference turns depth into 3-D observations
  3. PnP recovers the known camera motion, in world coordinates
  4. the tracker reports nothing (rather than something wrong) when it is blind

Usage:  python scripts/test_tracker.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from da3_slam.frontend.tracker import FrameTracker, TrackerConfig, TrackingReference
from smoke_test_utils import check, header

WIDTH, HEIGHT = 640, 480
FOCAL, DEPTH = 500.0, 2.0          # px, metres — a fronto-parallel plane
INTRINSIC = np.array([[FOCAL, 0, WIDTH / 2],
                      [0, FOCAL, HEIGHT / 2],
                      [0, 0, 1]], dtype=np.float32)


def textured_plane(seed: int = 0) -> np.ndarray:
    """A high-frequency texture LK can lock onto (blurred noise + blobs)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (HEIGHT // 4, WIDTH // 4), dtype=np.uint8)
    image = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    for _ in range(120):
        centre = (int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT)))
        cv2.circle(image, centre, int(rng.integers(4, 18)),
                   int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_GRAY2RGB)


def shifted(image: np.ndarray, dx_metres: float) -> np.ndarray:
    """The same plane seen after translating the camera by dx along +x.

    A fronto-parallel plane at depth Z shifts by -f·dx/Z pixels, exactly.
    """
    shift = -FOCAL * dx_metres / DEPTH
    matrix = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (WIDTH, HEIGHT),
                          borderMode=cv2.BORDER_REFLECT)


def reference_at(seq_idx: int, to_world: np.ndarray,
                 dx_metres: float = 0.0) -> TrackingReference:
    """A reference for the view the camera had after translating `dx_metres`."""
    return TrackingReference(
        seq_idx=seq_idx,
        gray=cv2.cvtColor(shifted(textured_plane(), dx_metres), cv2.COLOR_RGB2GRAY),
        depth=np.full((HEIGHT, WIDTH), DEPTH, dtype=np.float32),
        confidence=np.ones((HEIGHT, WIDTH), dtype=np.float32),
        intrinsic=INTRINSIC,
        to_world=to_world,
        rotation_world=to_world[:3, :3].copy(),
        scale=1.0,
    )


def main() -> int:
    header("Frame tracker")
    plane = textured_plane()
    tracker = FrameTracker(TrackerConfig(enable=True))

    # ── 1. tracks survive plain translation ─────────────────────────────────
    step = 0.02                                   # metres per frame -> 5 px
    for i in range(12):
        tracker.step(shifted(plane, i * step), i)
    alive = len(tracker._ids)
    check(f"tracks alive after 12 frames: {alive} (> 300)", alive > 300)

    # ── 2. a reference becomes 3-D observations ─────────────────────────────
    # World frame == the reference camera's frame, so the answer is readable.
    tracker.set_reference(reference_at(6, np.eye(4), 6 * step))
    tracker.step(shifted(plane, 12 * step), 12)
    check(f"3-D observations adopted: {len(tracker._object_points)}",
          len(tracker._object_points) > 200)

    # ── 3. PnP recovers the known motion ────────────────────────────────────
    # Frame 6 was the reference; frame 20 is 14 steps further along +x.
    pose = None
    for i in range(13, 21):
        pose = tracker.step(shifted(plane, i * step), i)
    check("pose reported", pose is not None)
    expected = (20 - 6) * step
    error = abs(float(pose[0, 3]) - expected)
    check(f"x translation {pose[0, 3]:+.4f} m (expected {expected:+.4f}, "
          f"error {error * 100:.2f} cm)", error < 0.01)
    drift = float(np.linalg.norm(pose[:3, 3][1:]))
    check(f"no spurious y/z motion: {drift:.4f} m", drift < 0.01)
    angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))
    check(f"rotation stays near identity: {angle:.2f} deg", angle < 1.0)

    # ── 4. the reference is at DA3's processed resolution ───────────────────
    # The pipeline tracks the input frame (640x480 here) but publishes depth
    # and intrinsics at DA3's processed size (504x378), so the mapping between
    # the two is live code — and wrong scaling shows up as a low PnP inlier
    # count rather than an error.
    pw, ph = 504, 378
    scaled = np.array([[FOCAL * pw / WIDTH, 0, pw / 2],
                       [0, FOCAL * ph / HEIGHT, ph / 2],
                       [0, 0, 1]], dtype=np.float32)
    tracker2 = FrameTracker(TrackerConfig(enable=True))
    for i in range(12):
        tracker2.step(shifted(plane, i * step), i)
    reference = TrackingReference(
        seq_idx=6, gray=cv2.cvtColor(textured_plane(), cv2.COLOR_RGB2GRAY), depth=np.full((ph, pw), DEPTH, dtype=np.float32),
        confidence=np.ones((ph, pw), dtype=np.float32), intrinsic=scaled,
        to_world=np.eye(4), rotation_world=np.eye(3), scale=1.0)
    tracker2.set_reference(reference)
    pose2 = None
    for i in range(12, 21):
        pose2 = tracker2.step(shifted(plane, i * step), i)
    check("pose reported from a processed-resolution reference", pose2 is not None)
    error2 = abs(float(pose2[0, 3]) - (20 - 6) * step)
    check(f"x translation {pose2[0, 3]:+.4f} m (error {error2 * 100:.2f} cm)",
          error2 < 0.01)
    check(f"PnP inliers {tracker2.stats.inliers} of "
          f"{len(tracker2._object_points)} observations",
          tracker2.stats.inliers > 0.5 * len(tracker2._object_points))

    # ── 5. recovery after a total track loss ────────────────────────────────
    # A fast-motion burst can wipe every track; without the re-acquisition path
    # the tracker stays dead until some later adoption happens to work (a 21 s
    # blackout, measured on fr1/teddy).
    lost = FrameTracker(TrackerConfig(enable=True))
    for i in range(7):
        lost.step(shifted(plane, i * step), i)
    lost._points = np.zeros((0, 2), dtype=np.float32)      # the burst
    lost._ids = np.zeros(0, dtype=np.int64)
    lost.set_reference(reference_at(6, np.eye(4), 6 * step))
    # The baseline is deliberately far past what optical flow can chain: 54
    # frames of motion, ~270 px of image shift, with no intermediate frames.
    recovered = None
    for i in range(60, 64):
        recovered = lost.step(shifted(plane, i * step), i)
    check("relocalises after losing every track, over a 270 px baseline",
          recovered is not None)
    error5 = abs(float(recovered[0, 3]) - (63 - 6) * step)
    check(f"recovered x translation {recovered[0, 3]:+.4f} m "
          f"(error {error5 * 100:.2f} cm)", error5 < 0.03)

    # ── 6. blind tracker reports nothing ────────────────────────────────────
    blind = FrameTracker(TrackerConfig(enable=True))
    check("no pose before a reference arrives", blind.step(plane, 0) is None)

    # ── 7. the plausibility gate ────────────────────────────────────────────
    # Degenerate PnP returns a finite camera a long way from the geometry that
    # produced it (89 consecutive Replica frames reached 1e38 m).  isfinite()
    # passes those, so the cloud bound is what catches them.
    gate = FrameTracker(TrackerConfig(enable=True))
    gate._object_points = np.random.default_rng(0).normal(
        scale=0.5, size=(500, 3)).astype(np.float32)
    inside, outside = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)
    outside[:3, 3] = 1e6
    check("plausible pose inside the observation cloud accepted",
          gate._plausible(inside))
    check("absurd-but-finite pose rejected", not gate._plausible(outside))

    print("\n  All tracker checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
