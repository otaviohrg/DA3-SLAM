"""
Per-frame pose tracking between submaps.

DA3 cannot say anything about a frame until its whole batch is complete, so
poses arrive in bursts: one update per submap, each carrying ~8 s of footage
that finished being captured ~6 s ago.  The map is fine — the *pose stream* is
what stalls.  This module adds the standard SLAM answer to that (fast tracking,
slow mapping — ORB-SLAM3 and DROID are built the same way): every incoming
frame is tracked against the geometry of the most recent optimised submap, so a
pose is available at camera rate and the DA3 pipeline corrects it whenever the
next submap lands.

Mechanism, in the order it happens:

  1. `step()` runs Lucas-Kanade on every frame, carrying a single set of
     features forward and topping it up as tracks die.  Each frame's feature
     positions are kept in a bounded history.
  2. When the processing thread finishes a submap it publishes a
     `TrackingReference` — the last keyframe's depth, intrinsics and current
     global transform.  That keyframe is ~6 s in the past, which is exactly why
     the history exists: the tracks seeded before it are still alive now, so
     their positions *at that keyframe* give 2-D observations to pair with the
     depth that just arrived, with no re-detection and no wide-baseline match.
  3. Those pairs become 3-D points in the reference camera's frame, and every
     subsequent frame solves PnP against them.

The tracker never writes to the pose graph: its poses are a display/control
convenience that the next submap overwrites.  A half-batch of DA3 geometry
would be a worse map; a PnP pose is a perfectly good *pose*.

**Operating envelope.**  Step 2 is the binding constraint: the reference
keyframe is one DA3 inference old, and the tracks seeded before it have to
still be alive.  Measured survival over that gap (1200 features, TUM
fr1/teddy / Replica office3):

    gap   1.5 s   2.6 s   3.0 s   4.0 s   5.8 s   8.0 s
    teddy   272     124      94      52      17       3
    office3 110      87      81      69      53      44

`min_inliers` is 30, so fast handheld footage supports a gap of about 4 s.
That is comfortable at `submap_size` 16 (inference 2.6 s) and marginal at 32
(5.8 s), where the tracker will intermittently report None and the consumer
should hold the last pose.  Smooth or synthetic footage tracks much longer.
The two knobs therefore move together: shrinking submaps to cut pose latency
also keeps the tracker fed.

Measured end to end at `submap_size` 16, 900 frames paced at 30 fps:

    sequence          coverage   inliers   ms/frame   pose lag   error vs map
    Replica office3        64%       385        8.9      10 ms    5.1 cm / 12.6 m
    TUM fr1/teddy          31%       188       10.7      12 ms    9.9 cm / 10.0 m

Roughly 23 points of the missing coverage is the unavoidable start-up gap —
nothing can be tracked until the first submap has been optimised.  The rest is
fast-motion dropout, where the tracker reports None rather than a bad pose.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TrackerConfig:
    """Tuning for the per-frame tracker (config/default.yaml: `tracking`)."""

    # Master switch.  Off by default: benchmarks score the optimised
    # trajectory, so tracking would only cost them CPU.
    enable: bool = False

    # Feature budget carried frame to frame, and the floor at which the set is
    # topped up with fresh corners.
    # Measured on TUM fr1/teddy: a track's chance of surviving the gap between
    # the reference keyframe and now decays fast (94 of 982 tracks survive 3 s
    # of fast handheld motion), and PnP needs `min_inliers` of them — so the
    # budget is set by survival, not by how many PnP wants.
    max_features: int = 1200
    min_features: int = 900

    # goodFeaturesToTrack parameters for the top-up.
    quality_level: float = 0.01
    min_distance: float = 12.0

    # Lucas-Kanade window / pyramid, and the forward-backward reprojection
    # threshold (px) a track must satisfy to survive.  FB checking is what
    # keeps a drifting track from quietly poisoning PnP.
    flow_window_size: tuple[int, int] = (31, 31)
    flow_pyramid_levels: int = 4
    fb_threshold: float = 2.0

    # Descriptor relocalisation: ORB against the reference keyframe, used both
    # to adopt a reference the chained tracks did not survive to and to recover
    # mid-dropout.  Needs no calibration and no vocabulary — the reference is
    # known, so it is a 1-vs-1 match, and DA3 supplies the depth and intrinsics.
    relocalize: bool = True
    # "orb" (no weights, no network) or "xfeat" (learned, better under blur and
    # low texture — see da3_slam/frontend/xfeat_matcher.py).
    matcher: str = "orb"
    # cuda, not cpu: measured on this box, one XFeat call is ~6 ms on the GPU
    # and ~1055 ms on the CPU — unusable in a per-frame budget of ~11 ms.  The
    # model is 6 MB and only runs on frames that are already lost, so it barely
    # touches the budget DA3 is competing for.
    xfeat_device: str = "cuda"
    xfeat_min_cossim: float = 0.82
    orb_features: int = 2500
    # Lowe ratio.  Measured on TUM fr1/teddy across 1-7 s baselines, the
    # textbook 0.75 is far too strict for this imagery — it survives 10-75
    # geometrically consistent matches where 0.90 survives 41-218.  The cost is
    # a lower inlier FRACTION, which is RANSAC's problem, not a correctness one.
    ratio_test: float = 0.90
    # Absolute Hamming cap on a match, on top of the ratio test.
    max_hamming: int = 64
    # Frames between relocalisation attempts while lost, bounding the cost of
    # a blackout (ORB extraction is ~5-8 ms, and only runs on lost frames).
    relocalize_interval: int = 3

    ransac_iterations: int = 1000

    # Keyframes scanned per frame when replenishing (newest first).
    replenish_keyframes: int = 6

    # Multi-view agreement for replenished observations.  A track's 3-D comes
    # from ONE keyframe's depth sample, so it inherits that sample's error
    # whole — measured, tracked poses sit at ~9 cm even against a fresh
    # reference, which is the floor this attacks.  With `triangulate`, a track
    # takes the point only if `min_views` keyframes independently put it in
    # the same place (within `agreement` metres), and then averages them.
    # Not true triangulation: the tracker has no cross-keyframe feature
    # correspondences, only per-keyframe depth, so this is multi-view depth
    # agreement — which is the part that rejects bad depth.
    triangulate: bool = False
    min_views: int = 2
    agreement: float = 0.05

    # Publish EVERY keyframe of a submap to the local map, not just its last
    # frame.  With one per submap the tracker held 12 viewpoints a whole submap
    # apart and solved against 7-8% of the depth the map already had.
    publish_all_keyframes: bool = False

    # Keyframes kept as a local map.  The tracker used to solve against the
    # newest one alone, so every time the view stopped overlapping THAT
    # keyframe it went blind — while the map held ~170 others, with depth,
    # already resident.  Older keyframes usually still cover the scene.
    local_map_size: int = 12
    # How many of them relocalisation tries, nearest-pose first, before failing.
    relocalize_candidates: int = 3

    # Blur gate: skip solving a frame whose Laplacian variance is below this
    # fraction of the recent median, letting the motion model carry it.
    # MEASURED AND REJECTED — it is off (0.0) for that reason.  The premise was
    # that PnP on a smeared frame succeeds on drifted correspondences and
    # returns a confident wrong pose; on TUM fr1/teddy the gate made the live
    # trajectory WORSE (live ATE 19.75 cm with it, 12.61 cm without, at equal
    # coverage).  A blurred frame's PnP solution, outliers and all, beats
    # extrapolating over it — RANSAC is already doing the filtering the gate
    # was meant to add.  Kept configurable so the result can be re-checked.
    blur_ratio: float = 0.0
    blur_window: int = 30

    # Provisional inference: run DA3 on the first half of each batch, at
    # `provisional_resolution`, purely to refresh the tracker's geometry
    # sooner.  Measured motivation: error runs ~9 cm against a 2-4 s old
    # reference and 22-33 cm past 8 s, and most poses were landing in the
    # older buckets.  It never enters the pose graph.
    # BOOTSTRAP LADDER: before the first submap exists the tracker has no
    # depth and no intrinsics, which is 14-22% of a clip at submap 16.  With
    # this on, DA3 re-runs on the first 2 keyframes, then 4, then 8, at
    # `provisional_resolution`, purely to seed the tracker.  None of it reaches
    # the pose graph, so unlike `submap_warmup_size` the map keeps a full-size
    # first batch and the metric scale chain it anchors is untouched.
    bootstrap: bool = False

    # Frames the hand-off queue may hold before the frontend starts dropping.
    # Depth trades POSE LAG for COVERAGE: the tracker runs behind by at most
    # this many frames, but a burst no longer costs frames outright.  For an
    # offline video this is free — poses are stored by seq_idx, so late
    # delivery lands in the right place — and it never touches the map either
    # way, since the frontend never blocks on this queue.
    queue_frames: int = 8

    provisional: bool = False

    # ROLLING REFRESH: with provisional on, re-infer every this many keyframes
    # instead of once per batch, so the reference never ages past that gap.
    # 0 keeps the once-per-batch behaviour.
    refresh_keyframes: int = 0
    # Keep live observations across a reference switch instead of replacing
    # them wholesale.  Only matters once references refresh often.
    merge_observations: bool = True
    # Keyframes of history a merged observation may come from.  0 = unbounded,
    # which is the stable choice: bounding it to 4 keeps the rolling
    # refresh's accuracy (17.1 vs 25.5 cm on common frames) but gives back
    # the coverage, and teddy's dropouts go from 10% of the clip to 25%.
    merge_max_age: int = 0
    # Keyframes per refresh (the shared anchor plus the newest of these), so
    # the cost stays flat rather than growing with the batch.
    provisional_window: int = 8
    provisional_resolution: int = 392

    # Constant-velocity prediction across short gaps: at 30 fps the camera has
    # barely moved, so extrapolating is far better than reporting nothing, and
    # the prediction also warm-starts the next PnP.
    motion_model_frames: int = 10

    # PnP RANSAC: inlier threshold (px) and the minimum number of inliers for a
    # pose to be reported at all.
    reproj_error: float = 5.0
    min_inliers: int = 20

    # Reject a solve whose camera lands further from the observation cloud than
    # this many cloud radii.  Degenerate correspondences make PnP return a
    # finite but absurd camera that `isfinite` cannot catch — measured as a run
    # of 89 Replica frames at up to 1e38 m, i.e. a tenth of that sequence's
    # live poses were garbage.  Gating on the cloud rather than on velocity is
    # deliberate: a velocity gate compares against the previous accepted pose,
    # which fed a relocalisation loop back into itself when it was tried.
    max_cloud_radii: float = 10.0

    # Minimum DA3 confidence for a reference pixel's depth to be trusted as a
    # 3-D observation.  The pipeline's own point clouds use a percentile; here
    # an absolute floor is enough, since PnP has RANSAC behind it.
    min_confidence: float = 0.1

    # Radius (px) within which a track adopts a reprojected reference point,
    # and the cell size of the lookup grid used to find it.
    replenish_radius: int = 4

    # How many frames of feature history to keep.  It must comfortably exceed
    # the pipeline's pose latency (submap span + DA3 inference), or the
    # reference keyframe will have fallen out of the history by the time its
    # geometry arrives.
    history: int = 900


@dataclass
class TrackingReference:
    """Geometry the tracker solves against, published by the processing thread.

    Everything is in the units the pipeline already uses: `depth`/`intrinsic`
    at DA3's processed resolution, and `to_world` the same point transform the
    viewer uses (it maps *scaled* camera points into the global frame, so
    camera-space points must be multiplied by `scale` first).
    """

    seq_idx: int
    gray: np.ndarray             # (H, W) uint8 at the depth's resolution
    depth: np.ndarray            # (H, W) float32, metres
    confidence: np.ndarray       # (H, W) float32, [0, 1]
    intrinsic: np.ndarray        # (3, 3) float32 at the depth's resolution
    to_world: np.ndarray         # (4, 4) scaled-camera -> world
    rotation_world: np.ndarray   # (3, 3) rotation of to_world, det-normalised
    scale: float                 # submap's global metric scale

    # seq_idx -> (to_world, scale) for the keyframes the tracker may still be
    # holding.  The graph moves them on every optimisation, so without this the
    # local map would drift out of agreement with the map it came from.
    updates: dict = field(default_factory=dict)

    # Other keyframes of the same submap, carried along to be registered in the
    # local map without being adopted as THE reference.  Publishing only the
    # submap's last frame left the tracker solving against 7-8% of the depth
    # the map already held, with its viewpoints a whole submap apart.
    companions: list = field(default_factory=list)


@dataclass
class TrackStats:
    """Counters for one `step()` — reported by the runner, not the tracker."""
    tracked: int = 0             # features alive after optical flow
    inliers: int = 0             # PnP inliers backing the reported pose
    predicted: bool = False      # extrapolated by the motion model, not solved
    blurred: bool = False        # frame rejected by the blur gate
    reference_seq: int = -1      # keyframe the observations came from
    milliseconds: float = 0.0


class FrameTracker:
    """Frame-to-frame LK tracking plus PnP against the latest submap.

    `step()` runs on the frontend thread, `set_reference()` on the processing
    thread; the handover is a single slot under a lock, picked up at the start
    of the next step.  Nothing else is shared.
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._pending: TrackingReference | None = None
        self._pending_correction: tuple | None = None

        self._prev_gray: np.ndarray | None = None
        self._points = np.zeros((0, 2), dtype=np.float32)   # current positions
        self._ids = np.zeros(0, dtype=np.int64)
        self._next_id = 0
        # seq_idx -> (ids, points) for every frame seen, bounded by `history`.
        self._history: deque[tuple[int, np.ndarray, np.ndarray]] = deque(
            maxlen=config.history)

        # Active 3-D observations: id -> row in `_object_points`.
        self._object_ids = np.zeros(0, dtype=np.int64)
        self._object_cam = np.zeros((0, 3), dtype=np.float32)   # in its own keyframe
        self._object_ref = np.zeros(0, dtype=np.int64)          # which keyframe
        self._object_points = np.zeros((0, 3), dtype=np.float32)  # in WORLD
        self._references: dict[int, TrackingReference] = {}
        self._clouds: dict[int, np.ndarray] = {}                # world-space, thinned
        self._history_poses: deque = deque(maxlen=3)            # (seq, pose)
        self._reference: TrackingReference | None = None
        self._orb = cv2.ORB_create(nfeatures=config.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ref_features: tuple | None = None      # (seq_idx, points, descriptors)
        self._xfeat_reference: tuple | None = None   # (seq_idx, points, features)
        self._since_relocalize = 0
        self._sharpness: deque = deque(maxlen=config.blur_window)
        # Previous PnP solution, used to warm-start the next one.
        self._last_rvec: np.ndarray | None = None
        self._last_tvec: np.ndarray | None = None
        self.stats = TrackStats()

    # ── handover from the processing thread ─────────────────────────────────

    def set_reference(self, reference: TrackingReference) -> None:
        """Publish new geometry; adopted on the next `step()`.

        Older geometry is ignored.  Two threads publish here — the processing
        thread when a submap is optimised, the inference thread when a
        provisional half-batch lands — so without this a slow provisional pass
        could overwrite a newer, better reference.
        """
        with self._lock:
            newest = max(list(self._references) or [-1])   # dict keyed by seq_idx
            if self._pending is not None:
                newest = max(newest, self._pending.seq_idx)
            if reference.seq_idx <= newest:
                return
            self._pending = reference

    # ── per-frame tracking ──────────────────────────────────────────────────

    def step(self, image: np.ndarray, seq_idx: int) -> np.ndarray | None:
        """Track one frame; return its (4, 4) cam-to-world pose, or None.

        None means "no pose this frame" — before the first submap has been
        optimised, or when the track set has decayed past `min_inliers`.  The
        caller should hold the previous pose rather than invent one.
        """
        import time
        t0 = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        if self._prev_gray is not None and len(self._points):
            self._flow(gray)
        self._prev_gray = gray

        # Adopt a reference *after* the flow update, so the history entry for
        # its keyframe and the current positions describe the same track ids.
        with self._lock:
            pending, self._pending = self._pending, None
            correction, self._pending_correction = self._pending_correction, None
        if correction is not None:
            # Before adopting: the correction drops the bootstrap geometry the
            # new reference is about to replace.
            self._apply_correction(*correction)
        if pending is not None:
            self._adopt(pending)

        self._history.append((seq_idx, self._ids.copy(), self._points.copy()))
        if len(self._points) < self.config.min_features:
            self._top_up(gray)

        blurred = self._is_blurred(gray)
        pose = None if blurred else self._solve(seq_idx)
        predicted = False
        if pose is None and not blurred and self._references:
            # Waiting for the next submap to re-seed geometry means several
            # seconds of nothing every time a burst wipes the tracks.  Retrying
            # costs an ORB pass, and only on frames that are already lost.
            self._since_relocalize += 1
            if (self.config.relocalize
                    and self._since_relocalize >= self.config.relocalize_interval):
                self._since_relocalize = 0
                if self._relocalize_local():
                    pose = self._solve(seq_idx)
        if pose is None:
            # Nothing solved: extrapolate briefly rather than report a hole.
            # A dropout is usually a handful of blurred frames, and holding the
            # trajectory continuous through them is both more useful and more
            # honest than a gap the consumer has to paper over.
            pose = self._predict(seq_idx)
            predicted = pose is not None
        if pose is not None and not np.isfinite(pose).all():
            # The solved path already refuses non-finite poses; the motion
            # model can produce them too (a degenerate rotation delta through
            # Rodrigues), and one NaN reaching a consumer poisons whatever it
            # plots, flies, or aligns — it crashed the benchmark's SVD.
            pose, predicted = None, False
        self.stats = TrackStats(tracked=len(self._points),
                                inliers=self.stats.inliers,
                                predicted=predicted,
                                blurred=blurred,
                                reference_seq=(self._reference.seq_idx
                                               if self._reference else -1),
                                milliseconds=(time.time() - t0) * 1e3)
        return pose

    def _is_blurred(self, gray: np.ndarray) -> bool:
        """Is this frame much softer than the recent ones?

        Relative, not absolute: sharpness depends on the scene as much as on
        the motion, so the comparison is against this sequence's own recent
        median.  The keyframe selector gates on the same quantity for the same
        reason (blur poisons DA3 poses and descriptors alike).
        """
        if self.config.blur_ratio <= 0:
            return False
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self._sharpness.append(sharpness)
        if len(self._sharpness) < self.config.blur_window // 2:
            return False
        return sharpness < self.config.blur_ratio * float(np.median(self._sharpness))

    def _flow(self, gray: np.ndarray) -> None:
        """Advance every track by Lucas-Kanade, dropping the ones that fail
        the forward-backward check."""
        cfg = self.config
        lk = dict(winSize=tuple(cfg.flow_window_size),
                  maxLevel=cfg.flow_pyramid_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._points, None, **lk)
        if forward is None:
            self._points = np.zeros((0, 2), dtype=np.float32)
            self._ids = np.zeros(0, dtype=np.int64)
            return
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, forward, None, **lk)
        good = (status.ravel() == 1) & (status_b.ravel() == 1)
        if backward is not None:
            good &= (np.linalg.norm(backward - self._points, axis=1)
                     <= cfg.fb_threshold)
        h, w = gray.shape[:2]
        good &= ((forward[:, 0] >= 0) & (forward[:, 0] < w)
                 & (forward[:, 1] >= 0) & (forward[:, 1] < h))
        self._points = forward[good].astype(np.float32)
        self._ids = self._ids[good]
        # Drop observations for dead tracks: replenishment adds to this set
        # every frame, and without pruning it grew to 5980 entries of which 28
        # were still alive — pure cost in every lookup.  All FOUR arrays are
        # parallel and must be cut together; pruning only two of them left
        # _object_points longer than _object_ids as soon as an adoption failed
        # (both the chained path and relocalisation), and the next solve died
        # on a boolean index of mismatched length.
        if len(self._object_ids):
            keep = np.isin(self._object_ids, self._ids)
            if not keep.all():
                self._object_ids = self._object_ids[keep]
                self._object_cam = self._object_cam[keep]
                self._object_ref = self._object_ref[keep]
                self._object_points = self._object_points[keep]

    def _top_up(self, gray: np.ndarray) -> None:
        """Detect fresh corners away from the surviving tracks.

        New tracks carry no 3-D yet; they earn it at the next reference, which
        is what keeps the tracker alive across a long run instead of decaying
        with the first feature set.
        """
        cfg = self.config
        mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
        for x, y in self._points.astype(int):
            cv2.circle(mask, (int(x), int(y)), int(cfg.min_distance), 0, -1)
        wanted = cfg.max_features - len(self._points)
        if wanted <= 0:
            return
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=wanted, qualityLevel=cfg.quality_level,
            minDistance=cfg.min_distance, mask=mask)
        if corners is None:
            return
        corners = corners.reshape(-1, 2).astype(np.float32)
        ids = np.arange(self._next_id, self._next_id + len(corners), dtype=np.int64)
        self._next_id += len(corners)
        self._points = np.vstack([self._points, corners])
        self._ids = np.concatenate([self._ids, ids])

    # ── reference adoption ──────────────────────────────────────────────────

    def _from_history(self, reference: TrackingReference):
        """(ids, 2-D positions at the reference keyframe) for tracks still alive."""
        snapshot = next(((ids, pts) for seq, ids, pts in reversed(self._history)
                         if seq == reference.seq_idx), None)
        if snapshot is None:              # keyframe older than the history
            return None
        ref_ids, ref_points = snapshot
        keep = np.isin(ref_ids, self._ids)
        ref_ids, ref_points = ref_ids[keep], ref_points[keep]
        if len(ref_ids) < self.config.min_inliers:
            return None
        return ref_ids, ref_points

    def _match_orb(self, reference: TrackingReference, live_gray: np.ndarray):
        """ORB + Lowe ratio: (live points, matched reference points)."""
        ref_points, ref_descriptors = self._reference_features(reference)
        if ref_descriptors is None or len(ref_descriptors) < self.config.min_inliers:
            return None
        live_keypoints, live_descriptors = self._orb.detectAndCompute(live_gray, None)
        if live_descriptors is None or len(live_descriptors) < self.config.min_inliers:
            return None
        pairs = self._matcher.knnMatch(live_descriptors, ref_descriptors, k=2)
        # A match is only trusted when it is clearly better than the runner-up.
        # PnP's RANSAC is the geometric check after it.
        good = [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < self.config.ratio_test * p[1].distance
                and p[0].distance < self.config.max_hamming]
        if len(good) < self.config.min_inliers:
            return None
        live = np.array([live_keypoints[m.queryIdx].pt for m in good], dtype=np.float32)
        return live, ref_points[[m.trainIdx for m in good]]

    def _match_xfeat(self, reference: TrackingReference, live_gray: np.ndarray):
        """XFeat mutual-nearest-neighbour match, with the same return contract.

        The reference's features are cached per keyframe — it is the live frame
        that has to be described every attempt.
        """
        from da3_slam.frontend import xfeat_matcher

        cached = self._xfeat_reference
        if cached is None or cached[0] != reference.seq_idx:
            points, features = xfeat_matcher.detect(
                reference.gray, self.config.xfeat_device)
            if points is None:
                return None
            # only keypoints whose depth is usable can become observations
            usable = np.isfinite(reference.depth) & (reference.depth > 0)
            if reference.confidence is not None and reference.confidence.size:
                usable &= reference.confidence >= self.config.min_confidence
            h, w = reference.gray.shape[:2]
            rows = np.clip(points[:, 1].astype(int), 0, h - 1)
            columns = np.clip(points[:, 0].astype(int), 0, w - 1)
            keep = usable[rows, columns]
            features["descriptors"] = features["descriptors"][keep]
            cached = (reference.seq_idx, points[keep], features)
            self._xfeat_reference = cached
        _, ref_points, ref_features = cached
        if len(ref_points) < self.config.min_inliers:
            return None
        live_points, live_features = xfeat_matcher.detect(
            live_gray, self.config.xfeat_device)
        if live_points is None or len(live_points) < self.config.min_inliers:
            return None
        index_live, index_ref = xfeat_matcher.match(
            live_features, ref_features, self.config.xfeat_min_cossim)
        if len(index_live) < self.config.min_inliers:
            return None
        return live_points[index_live], ref_points[index_ref]

    def _reference_features(self, reference: TrackingReference):
        """ORB keypoints and descriptors of the reference, where depth is usable.

        Detection is masked by every condition the observation must satisfy
        later (valid depth AND confidence), so a match cannot be discarded
        downstream — that exact mismatch silently threw away 51 of 51 matches
        when the mask only encoded depth.
        """
        if self._ref_features is not None and self._ref_features[0] == reference.seq_idx:
            return self._ref_features[1], self._ref_features[2]
        usable = np.isfinite(reference.depth) & (reference.depth > 0)
        if reference.confidence is not None and reference.confidence.size:
            usable &= reference.confidence >= self.config.min_confidence
        keypoints, descriptors = self._orb.detectAndCompute(
            reference.gray, usable.astype(np.uint8) * 255)
        points = (np.array([k.pt for k in keypoints], dtype=np.float32)
                  if keypoints else np.zeros((0, 2), dtype=np.float32))
        self._ref_features = (reference.seq_idx, points, descriptors)
        return points, descriptors

    def _relocalize_local(self) -> bool:
        """Try relocalising against the local map, nearest keyframe first.

        The newest keyframe is not always the right target: after the camera
        turns away from it, an older one covers the view far better.  Ordering
        by distance from the last known pose puts that one first.
        """
        for seq in self._nearby(self.config.relocalize_candidates):
            reference = self._references[seq]
            found = self._relocalize(reference)
            if found is not None:
                self._observe(reference, *found)
                return True
        return False

    def _relocalize(self, reference: TrackingReference):
        """Match the reference keyframe into the live frame by descriptor.

        This is the recovery path, and it is deliberately not optical flow:
        once a fast-motion burst wipes the track set, LK has no way back in —
        it needs continuity it no longer has, and chaining it over the
        multi-second baseline to the reference failed on six consecutive
        attempts (a 21 s blackout, measured on fr1/teddy).  Descriptors do not
        care how the camera got here, only that it is looking at the same
        surface, so they also recover after blur and after looking away.

        Matching happens at the REFERENCE's resolution: BRIEF is not scale
        invariant, and upscaling DA3's processed image to the camera's size
        softens the corners both sides depend on.
        """
        if (not self.config.relocalize or reference.gray is None
                or self._prev_gray is None):
            return None
        h, w = reference.gray.shape[:2]
        ih, iw = self._prev_gray.shape[:2]
        live_gray = cv2.resize(self._prev_gray, (w, h))
        if self.config.matcher == "xfeat":
            found = self._match_xfeat(reference, live_gray)
        else:
            found = self._match_orb(reference, live_gray)
        if found is None:
            return None
        live, matched = found
        scale = np.array([iw / w, ih / h], dtype=np.float32)
        # register the matches as live tracks, so chained tracking resumes
        ids = np.arange(self._next_id, self._next_id + len(live), dtype=np.int64)
        self._next_id += len(ids)
        self._points = np.vstack([self._points, live * scale])
        self._ids = np.concatenate([self._ids, ids])
        self._trim_tracks(protect=ids)
        return ids, matched * scale

    def set_correction(self, transform: np.ndarray, discard_upto: int = -1) -> None:
        """Queue a gauge correction; applied on the next `step()`.

        It must NOT be applied here: this is called from the processing thread
        while the frontend is inside `step()`, and dropping references out from
        under `_replenish` raced it into a KeyError.  Same discipline as
        `set_reference`.
        """
        with self._lock:
            self._pending_correction = (np.asarray(transform, dtype=np.float64),
                                        int(discard_upto))

    def _apply_correction(self, transform: np.ndarray,
                          discard_upto: int = -1) -> None:
        """Move the tracker's own pose history into a corrected gauge.

        Observations are re-registered by `_register` from their keyframes, so
        only what the tracker remembers about ITSELF needs moving: the solved
        poses the motion model extrapolates from, and the PnP warm start, which
        would otherwise seed the next solve from the old frame.

        `discard_upto` additionally throws away every observation sampled from
        a keyframe at or below that index.  Re-projecting them is not enough:
        their CAMERA-space depth came from a 2-to-8 frame inference at
        `provisional_resolution`, and no change of frame fixes a bad depth.
        The real submap re-supplies the same keyframes at full resolution, so
        replenishment picks them up again within a few frames.
        """
        matrix = np.asarray(transform, dtype=np.float64)
        linear = matrix[:3, :3]
        scale = float(np.cbrt(max(abs(np.linalg.det(linear)), 1e-18)))
        rotation = linear / scale if scale > 0 else linear
        moved = []
        for seq, pose in self._history_poses:
            out = np.eye(4, dtype=np.float32)
            out[:3, :3] = (rotation @ pose[:3, :3].astype(np.float64)).astype(np.float32)
            out[:3, 3] = (linear @ pose[:3, 3].astype(np.float64)
                          + matrix[:3, 3]).astype(np.float32)
            moved.append((seq, out))
        self._history_poses.clear()
        self._history_poses.extend(moved)
        self._last_rvec = self._last_tvec = None
        if discard_upto >= 0 and len(self._object_ids):
            keep = self._object_ref > discard_upto
            self._object_ids = self._object_ids[keep]
            self._object_cam = self._object_cam[keep]
            self._object_ref = self._object_ref[keep]
            self._object_points = self._object_points[keep]
            for seq in [s for s in self._references if s <= discard_upto]:
                self._references.pop(seq, None)
                self._clouds.pop(seq, None)
            self._check_observations()

    def _adopt(self, reference: TrackingReference) -> None:
        """Turn a new reference's depth into 3-D observations.

        First choice is the chained path — tracks alive at that keyframe and
        still alive now — which costs nothing because the flow has been running
        all along.  When too few survive, descriptor relocalisation takes over.
        """
        self._register(reference)
        found = self._from_history(reference)
        if found is None:
            found = self._relocalize(reference)
        if found is None:
            return
        self._observe(reference, *found)

    def _register(self, reference: TrackingReference) -> None:
        """Add a keyframe (and its companions) to the local map, refreshing poses."""
        for seq, (to_world, scale) in (reference.updates or {}).items():
            held = self._references.get(int(seq))
            if held is not None:
                held.to_world, held.scale = to_world, float(scale)
                self._clouds.pop(int(seq), None)       # its world points moved
        for companion in reference.companions:
            self._references[int(companion.seq_idx)] = companion
        self._references[int(reference.seq_idx)] = reference
        while len(self._references) > self.config.local_map_size:
            oldest = min(self._references)
            self._references.pop(oldest)
            self._clouds.pop(oldest, None)
        self._reproject_observations()

    def _reproject_observations(self) -> None:
        """Rebuild world-space observations from their keyframes' current poses."""
        if not len(self._object_ids):
            self._object_points = np.zeros((0, 3), dtype=np.float32)
            return
        world = np.empty_like(self._object_cam)
        for seq in np.unique(self._object_ref):
            reference = self._references.get(int(seq))
            rows = self._object_ref == seq
            if reference is None:                      # keyframe fell out of the map
                world[rows] = self._object_points[rows] if len(
                    self._object_points) == len(world) else 0.0
                continue
            scaled = self._object_cam[rows] * np.float32(reference.scale)
            world[rows] = (scaled @ reference.to_world[:3, :3].T.astype(np.float32)
                           + reference.to_world[:3, 3].astype(np.float32))
        self._object_points = world

    def _observe(self, reference: TrackingReference, ref_ids: np.ndarray,
                 ref_points: np.ndarray) -> None:
        """Sample the reference's depth at `ref_points` to make observations."""
        # The tracker works in input-image pixels; depth and intrinsics are at
        # DA3's processed resolution.  Map the observations across.
        h, w = reference.depth.shape[:2]
        ih, iw = self._prev_gray.shape[:2]
        sx, sy = w / iw, h / ih
        px = np.clip((ref_points[:, 0] * sx).astype(int), 0, w - 1)
        py = np.clip((ref_points[:, 1] * sy).astype(int), 0, h - 1)
        depth = reference.depth[py, px]
        valid = np.isfinite(depth) & (depth > 0)
        # Lean runs (build_pointclouds off) keep depth but drop the confidence
        # map; the gate simply does not apply there — RANSAC still filters.
        if reference.confidence is not None and reference.confidence.size:
            valid &= reference.confidence[py, px] >= self.config.min_confidence
        if valid.sum() < self.config.min_inliers:
            return

        K = reference.intrinsic
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        d = depth[valid]
        x = (px[valid] - cx) / fx * d
        y = (py[valid] - cy) / fy * d
        points_cam = np.stack([x, y, d], axis=1).astype(np.float32)
        # Observations live in WORLD space, tagged with the keyframe they came
        # from: that is what lets several keyframes contribute to one PnP, and
        # what lets a later graph correction move them (see _register).
        new_ids = ref_ids[valid]
        new_ref = np.full(len(points_cam), int(reference.seq_idx), dtype=np.int64)
        merged = False
        if self.config.merge_observations and len(self._object_ids):
            # MERGE rather than replace.  Replacing was right when a reference
            # arrived once per submap: the set it threw away was stale anyway.
            # With a rolling refresh it fires several times per submap, and
            # each replacement discarded the accumulated multi-keyframe local
            # map and forced a cold solve — measured as live-pose jerk p99
            # rising 120.7 -> 176.4 mm, which reads as an erratic camera.
            # Observations are in WORLD space, so keeping the live ones across
            # a reference switch is sound; only duplicates of the ids the new
            # reference supplies are dropped, in its favour.
            alive = _index_of(self._object_ids, self._ids) >= 0
            keep = alive & ~np.isin(self._object_ids, new_ids)
            if self.config.merge_max_age > 0:
                # Keep continuity, not staleness.  Unbounded merging retained
                # geometry from references the refresh was there to replace and
                # gave back the whole accuracy gain (17.3 -> 27.3 cm on common
                # frames) in exchange for coverage.
                keep &= (int(reference.seq_idx) - self._object_ref
                         <= self.config.merge_max_age)
            if keep.any():
                self._object_cam = np.vstack([self._object_cam[keep], points_cam])
                self._object_ref = np.concatenate([self._object_ref[keep], new_ref])
                self._object_ids = np.concatenate([self._object_ids[keep], new_ids])
                merged = True
        if not merged:
            self._object_cam, self._object_ref = points_cam, new_ref
            self._object_ids = new_ids
        self._reference = reference
        self._reproject_observations()
        self._check_observations()
        if not merged:
            # Nothing survived, so the old solution describes a camera posed
            # against geometry that is gone: seed cold.
            self._last_rvec = self._last_tvec = None
            self.stats.inliers = 0

    def _trim_tracks(self, protect: np.ndarray) -> None:
        """Hold the track set to its budget, dropping the oldest unprotected ones.

        Relocalisation adds a whole fresh match set on top of whatever survived,
        which would otherwise push the population past `max_features` and make
        every later optical-flow call more expensive than it was budgeted to be.
        """
        excess = len(self._ids) - self.config.max_features
        if excess <= 0:
            return
        removable = np.flatnonzero(~np.isin(self._ids, protect))[:excess]
        if not len(removable):
            return
        keep = np.ones(len(self._ids), dtype=bool)
        keep[removable] = False
        self._points, self._ids = self._points[keep], self._ids[keep]

    @staticmethod
    def _subsample_cloud(reference: TrackingReference, stride: int = 4):
        """A thinned point cloud of the reference view, in its camera frame."""
        depth = reference.depth[::stride, ::stride]
        confidence = (reference.confidence[::stride, ::stride]
                      if reference.confidence is not None
                      and reference.confidence.size else None)
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w) * stride, np.arange(h) * stride)
        K = reference.intrinsic
        valid = np.isfinite(depth) & (depth > 0)
        if confidence is not None:
            valid &= confidence > 0
        z = depth[valid]
        x = (u[valid] - float(K[0, 2])) * z / float(K[0, 0])
        y = (v[valid] - float(K[1, 2])) * z / float(K[1, 1])
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _nearby(self, count: int) -> list[int]:
        """Local-map keyframes closest to where the camera last was.

        With one keyframe per submap, "newest" and "nearest" were the same
        thing.  With a dense local map they are not: the newest few cover a
        couple of seconds of trajectory, while the camera may be looking at
        something it passed earlier — which is exactly the case the dense map
        exists to serve.
        """
        if not self._references:
            return []
        anchor = self._history_poses[-1][1][:3, 3] if self._history_poses else None
        order = sorted(self._references, reverse=True)
        if anchor is not None:
            order.sort(key=lambda q: float(np.linalg.norm(
                self._references[q].to_world[:3, 3] - anchor)))
        return order[:count]

    def _cloud(self, seq: int):
        """Thinned point cloud of a local-map keyframe, in camera and world space."""
        cached = self._clouds.get(seq)
        if cached is not None:
            return cached
        reference = self._references[seq]
        cam = self._subsample_cloud(reference)
        scaled = cam * np.float32(reference.scale)
        world = (scaled @ reference.to_world[:3, :3].T.astype(np.float32)
                 + reference.to_world[:3, 3].astype(np.float32))
        self._clouds[seq] = (cam, world)
        return cam, world

    def _replenish(self, K: np.ndarray) -> None:
        """Give 3-D to tracks born since the observations were made.

        Runs over the WHOLE local map, not just the newest keyframe.  A track
        that appeared after the reference is still looking at surfaces the map
        holds — often surfaces an older keyframe saw rather than the newest one
        — so the reference cloud is projected into the current frame with the
        pose just solved and each track adopts the nearest projected point,
        closest surface winning its cell.  Without any replenishment the
        observation set only shrinks; without the older keyframes it shrinks
        again the moment the view leaves the newest one.
        """
        if self._last_rvec is None or not self._references:
            return
        missing = _index_of(self._ids, self._object_ids) < 0
        if not missing.any():
            return
        want, want_ids = self._points[missing], self._ids[missing]
        h, w = self._prev_gray.shape[:2]
        cell = self.config.replenish_radius
        columns = w // cell + 1
        want_keys = ((want[:, 1] // cell).astype(np.int64) * columns
                     + (want[:, 0] // cell).astype(np.int64))
        # slot -> list of (cam point, world point, keyframe) from each view
        candidates: dict[int, list] = {}
        for seq in self._nearby(self.config.replenish_keyframes):
            cam, world = self._cloud(seq)
            if not len(world):
                continue
            projected, _ = cv2.projectPoints(world, self._last_rvec,
                                             self._last_tvec, K, None)
            projected = projected.reshape(-1, 2)
            inside = ((projected[:, 0] >= 0) & (projected[:, 0] < w)
                      & (projected[:, 1] >= 0) & (projected[:, 1] < h))
            if not inside.any():
                continue
            index = np.flatnonzero(inside)
            keys = ((projected[index, 1] // cell).astype(np.int64) * columns
                    + (projected[index, 0] // cell).astype(np.int64))
            # Nearest surface per cell, vectorised: sort by (cell, depth) and
            # take each cell's first row.  The dict-and-loop version cost 12 ms
            # a frame once the local map held a dozen keyframes.
            order = np.lexsort((cam[index, 2], keys))
            sorted_keys = keys[order]
            first = np.ones(len(sorted_keys), dtype=bool)
            first[1:] = sorted_keys[1:] != sorted_keys[:-1]
            cell_keys, cell_rows = sorted_keys[first], index[order][first]
            slot = np.clip(np.searchsorted(cell_keys, want_keys), 0,
                           max(len(cell_keys) - 1, 0))
            candidate = cell_rows[slot]
            hit = (cell_keys[slot] == want_keys)
            hit &= np.linalg.norm(projected[candidate] - want, axis=1) <= cell
            for position in np.flatnonzero(hit):
                row = int(candidate[position])
                if not self.config.triangulate and int(position) in candidates:
                    continue          # newest keyframe wins, as before
                candidates.setdefault(int(position), []).append(
                    (cam[row], world[row], seq))

        taken: dict[int, tuple] = {}
        for position, views in candidates.items():
            if not self.config.triangulate:
                cam_point, world_point, seq = views[0]
            else:
                if len(views) < self.config.min_views:
                    continue
                points = np.stack([v[1] for v in views])
                centre = np.median(points, axis=0)
                agree = np.linalg.norm(points - centre, axis=1) <= self.config.agreement
                if agree.sum() < self.config.min_views:
                    continue          # the views disagree: the depth is noise
                world_point = points[agree].mean(axis=0)
                # keep the newest agreeing view's camera-space point, so a
                # later pose refresh can still re-project this observation
                cam_point, _, seq = views[int(np.flatnonzero(agree)[0])]
            taken[position] = (want_ids[position], cam_point, world_point, seq)

        if not taken:
            return
        ids, cams, worlds, seqs = zip(*taken.values())
        self._object_ids = np.concatenate([self._object_ids, np.array(ids, dtype=np.int64)])
        self._object_cam = np.vstack([self._object_cam, np.array(cams, dtype=np.float32)])
        self._object_points = np.vstack([self._object_points,
                                         np.array(worlds, dtype=np.float32)])
        self._object_ref = np.concatenate([self._object_ref, np.array(seqs, dtype=np.int64)])

    # ── motion model ────────────────────────────────────────────────────────

    def _predict(self, seq_idx: int) -> np.ndarray | None:
        """Constant-velocity extrapolation from the last two solved poses."""
        if len(self._history_poses) < 2:
            return None
        (seq_a, pose_a), (seq_b, pose_b) = self._history_poses[-2], self._history_poses[-1]
        span = seq_b - seq_a
        if span <= 0 or seq_idx - seq_b > self.config.motion_model_frames:
            return None
        step = (seq_idx - seq_b) / span
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = pose_b[:3, 3] + (pose_b[:3, 3] - pose_a[:3, 3]) * step
        delta = pose_a[:3, :3].T @ pose_b[:3, :3]
        rvec, _ = cv2.Rodrigues(delta.astype(np.float64))
        extra, _ = cv2.Rodrigues(rvec * step)
        pose[:3, :3] = (pose_b[:3, :3].astype(np.float64) @ extra).astype(np.float32)
        return pose

    # ── PnP ─────────────────────────────────────────────────────────────────

    def _intrinsic(self) -> np.ndarray | None:
        """The newest keyframe's intrinsics, scaled to the tracked image."""
        if self._reference is None or self._prev_gray is None:
            return None
        h, w = self._reference.depth.shape[:2]
        ih, iw = self._prev_gray.shape[:2]
        K = self._reference.intrinsic.astype(np.float64).copy()
        K[0, :] *= iw / w
        K[1, :] *= ih / h
        return K

    def _plausible(self, pose: np.ndarray) -> bool:
        """Is this camera anywhere near the points that produced it?

        A finite pose can still be nonsense: PnP on degenerate correspondences
        returned cameras up to 1e38 m out on Replica, for a run of 89 frames,
        and nothing downstream of `isfinite` noticed (the video renderer's
        outlier gate silently dropped them; the benchmark would have scored
        them).  The observation cloud is the only scale reference that suits
        both a 4 m room and a 250 m KITTI track.
        """
        if self.config.max_cloud_radii <= 0 or not len(self._object_points):
            return True
        centre = self._object_points.mean(axis=0)
        radius = float(np.linalg.norm(self._object_points - centre, axis=1).max())
        limit = self.config.max_cloud_radii * max(radius, 1e-3)
        return bool(np.linalg.norm(pose[:3, 3] - centre) <= limit)

    def _check_observations(self) -> None:
        """The four observation arrays are parallel; assert it cheaply."""
        n = len(self._object_ids)
        assert len(self._object_cam) == n and len(self._object_ref) == n \
            and len(self._object_points) == n, (
                f"observation arrays out of step: ids={n} "
                f"cam={len(self._object_cam)} ref={len(self._object_ref)} "
                f"world={len(self._object_points)}")

    def _solve(self, seq_idx: int) -> np.ndarray | None:
        """Pose of the current frame from world 2-D/3-D correspondences.

        Solving against WORLD points (rather than one keyframe's camera frame)
        is what makes the local map possible: observations from any number of
        keyframes enter the same PnP, and the result is the camera-to-world
        pose directly.
        """
        K = self._intrinsic()
        if K is None or not len(self._object_ids):
            return None
        self._check_observations()
        order = _index_of(self._object_ids, self._ids)
        alive = order >= 0
        if alive.sum() < self.config.min_inliers:
            return None
        object_points = self._object_points[alive].astype(np.float64)
        image_points = self._points[order[alive]].astype(np.float64)

        # Warm start ONLY when the previous frame solved.  The motion model is
        # a better seed than the last pose while tracking is healthy, but after
        # a failure it is a guess about a camera that just did something
        # unmodelled, and iterative PnP from a bad seed converges to a bad
        # local minimum where cold EPnP would have recovered — measured as a
        # coverage drop from 36% to 25%.
        rvec0 = tvec0 = None
        if self._last_rvec is not None:
            guess = self._predict(seq_idx)
            if guess is not None:
                rotation = guess[:3, :3].astype(np.float64).T
                rvec0, _ = cv2.Rodrigues(rotation)
                tvec0 = (-rotation @ guess[:3, 3].astype(np.float64)).reshape(3, 1)
            else:
                rvec0, tvec0 = self._last_rvec.copy(), self._last_tvec.copy()
        warm = rvec0 is not None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, None,
            rvec=rvec0, tvec=tvec0, useExtrinsicGuess=warm,
            flags=cv2.SOLVEPNP_ITERATIVE if warm else cv2.SOLVEPNP_EPNP,
            reprojectionError=self.config.reproj_error,
            # Relocalisation matches can run 10-20% inliers; 100 iterations
            # cannot find a consensus that sparse.  Costs nothing in the steady
            # state, where RANSAC reaches its confidence and stops.
            iterationsCount=self.config.ransac_iterations, confidence=0.999)
        n_inliers = 0 if inliers is None else len(inliers)
        self.stats.inliers = n_inliers
        if not ok or n_inliers < self.config.min_inliers:
            self._last_rvec = self._last_tvec = None
            return None

        index = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[index], image_points[index], K, None, rvec, tvec)
        self._last_rvec, self._last_tvec = rvec.copy(), tvec.copy()

        # Keep only what the geometry confirmed.  Relocalisation admits
        # candidates generously (ratio 0.90), so without this the observation
        # set silently fills with outliers, the inlier fraction stays near 20%,
        # and RANSAC then burns its full iteration budget on EVERY later frame.
        confirmed = np.flatnonzero(alive)[index]
        self._object_ids = self._object_ids[confirmed]
        self._object_cam = self._object_cam[confirmed]
        self._object_ref = self._object_ref[confirmed]
        self._object_points = self._object_points[confirmed]

        rotation, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotation.T.astype(np.float32)
        pose[:3, 3] = (-rotation.T @ tvec).ravel().astype(np.float32)
        if not np.isfinite(pose).all() or not self._plausible(pose):
            # Degenerate PnP (seen with a loose inlier gate): refuse rather
            # than hand a NaN pose to a consumer that will plot or fly it.
            self._last_rvec = self._last_tvec = None
            return None

        self._replenish(K)
        self._history_poses.append((seq_idx, pose.copy()))
        return pose


def _index_of(wanted: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Row of each `wanted` id inside `pool`, or -1 when it has died."""
    order = np.argsort(pool)
    position = np.searchsorted(pool[order], wanted)
    position = np.clip(position, 0, len(pool) - 1)
    candidate = order[position]
    return np.where(pool[candidate] == wanted, candidate, -1)
