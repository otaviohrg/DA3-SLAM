"""
Render a video of DASH-SLAM running.

Two stages in one command:

  1. **record** — the pipeline is run over an image directory with an
     ``on_update`` callback (the same hook ``scripts/live_viewer.py`` uses) that
     caches every submap's camera-space points plus the *current* pose of every
     keyframe.  Recording is cheap: it copies arrays, it never renders, so it
     does not backpressure the SLAM threads (see SLAMUpdate's contract).
  2. **render** — the cached snapshots are replayed on a wall-clock timeline and
     rasterised with a small painter's-algorithm point renderer (numpy + cv2, no
     OpenGL / no display), then encoded to H.264.

The recording is written next to the video, so a run can be re-rendered with
different camera / trail settings for free (``--render_only``).

Why the poses are stored per snapshot and the points in *camera* space: loop
closure and later optimisation move keyframes that were already drawn.  Every
video frame re-projects the whole map with the newest poses available at that
point in the timeline, so a closure visibly snaps the map together instead of
leaving a duplicated ghost copy behind (same reason LiveViewer caches).

Usage:
    # record + render (all run_slam.py flags are accepted and forwarded)
    python3 scripts/make_video.py --image_dir data/video1_30fps \
        --video outputs/video/dash_slam.mp4 --max_frames 400

    # re-render an existing recording with a different look
    python3 scripts/make_video.py --render_only \
        --recording outputs/video/dash_slam.rec.pkl.gz \
        --video outputs/video/orbit.mp4 --view orbit --trail_seconds 8

Notes:
    * ``--view fit`` (default) holds a fixed viewpoint framed on the final map,
      ``orbit`` slowly rotates around it, ``follow`` chases the camera.
    * ``--pacing capture`` (default) plays back on the input video's own clock,
      i.e. real time; ``--pacing wall`` uses the measured processing times, so
      the map appears as fast as the machine actually produced it.
"""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ── recording ─────────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """One SLAMUpdate, reduced to what the renderer needs."""
    submap_idx: int
    wall_time: float                                   # s since the run started
    # (seq_idx, points_cam (M,3) float32, colors (M,3) uint8) per new frame,
    # already downsampled.  Camera space, so they can be re-projected later.
    frames: list[tuple[int, np.ndarray, np.ndarray]]
    poses: dict[int, np.ndarray]                       # seq_idx -> (4,4) cam->world
    scales: dict[int, float]                           # seq_idx -> Sim(3) scale
    n_submaps: int
    n_loop_closures: int


@dataclass
class Recording:
    snapshots: list[Snapshot] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    source_fps: float = 30.0
    total_wall: float = 0.0
    label: str = ""

    # seq_idx -> (4, 4) pose from the per-frame tracker (frontend/tracker.py).
    # Only populated when the run was both tracked and paced: unpaced replay
    # runs the frontend to the end of the sequence before the first submap
    # exists, so nothing is ever tracked.
    tracked_poses: dict[int, np.ndarray] = field(default_factory=dict)

    # True when frames were fed at the camera's rate, which makes each
    # snapshot's wall_time directly comparable to its capture time — the
    # measurement `--pacing realtime` draws the map latency from.
    paced: bool = False


class RunRecorder:
    """``on_update`` callback that caches submap snapshots for later rendering.

    Runs inside the SLAM processing thread, so it only subsamples and copies —
    anything heavier would slow the pipeline down (and, in ``--pacing wall``
    mode, would show up as a slower video).
    """

    def __init__(self, points_per_submap: int = 40_000, seed: int = 0):
        self._budget = points_per_submap
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time()
        self._seen: set[int] = set()
        self.snapshots: list[Snapshot] = []

    def __call__(self, update) -> None:
        # The end-of-run refresh re-emits the last submap with corrected poses;
        # keep the poses, drop the points, or the map would be drawn twice.
        new = [f for f in update.frame_points_cam if int(f[0]) not in self._seen]
        self._seen.update(int(f[0]) for f in new)
        frames = []
        total = sum(len(p) for _, p, _ in new)
        keep = min(1.0, self._budget / total) if total else 1.0
        for seq_idx, points, colors in new:
            finite = np.isfinite(points).all(axis=1)
            points, colors = points[finite], colors[finite]
            if keep < 1.0 and len(points):
                # Random (not strided) subsample: the render-time budget keeps a
                # *prefix* of this array, which is only unbiased if the order is
                # already shuffled.
                sel = self._rng.choice(len(points), max(1, int(len(points) * keep)),
                                       replace=False)
                points, colors = points[sel], colors[sel]
            if len(points):
                frames.append((int(seq_idx),
                               points.astype(np.float32),
                               colors.astype(np.uint8)))
        frames.sort(key=lambda f: f[0])
        self.snapshots.append(Snapshot(
            submap_idx=int(update.submap_idx),
            wall_time=time.time() - self._t0,
            frames=frames,
            poses={int(k): np.asarray(v, dtype=np.float32).copy()
                   for k, v in update.keyframe_poses.items()},
            scales={int(k): float(v)
                    for k, v in (update.keyframe_scales or {}).items()},
            n_submaps=int(update.n_submaps),
            n_loop_closures=int(update.n_loop_closures),
        ))
        print(f"[video] recorded submap {update.submap_idx}: "
              f"{sum(len(p) for _, p, _ in frames):,} points, "
              f"{len(update.keyframe_poses)} keyframes", flush=True)


def save_recording(rec: Recording, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[video] recording saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_recording(path: Path) -> Recording:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def infer_source_fps(image_paths: list[str], default: float) -> float:
    """Recover the capture rate from numeric filenames (TUM-style timestamps).

    Falls back to `default` for frame_000123.jpg-style names, which carry no
    time information.
    """
    try:
        stamps = [float(Path(p).stem) for p in image_paths]
    except ValueError:
        return default
    span = stamps[-1] - stamps[0]
    if len(stamps) < 2 or span <= 0 or span > 1e6:
        return default
    fps = (len(stamps) - 1) / span
    return fps if 0.5 < fps < 240 else default


# ── 3-D helpers ───────────────────────────────────────────────────────────────

# DA3 works in the OpenCV convention (x right, y DOWN, z forward), so world "up"
# is -Y everywhere below.
WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """(3,4) world->camera matrix for a camera at `eye` looking at `target`."""
    z = _normalize(np.asarray(target, float) - np.asarray(eye, float))
    down = -WORLD_UP
    y = down - np.dot(down, z) * z
    if np.linalg.norm(y) < 1e-6:            # looking straight down: pick any
        y = np.cross(z, np.array([1.0, 0.0, 0.0]))
    y = _normalize(y)
    x = np.cross(y, z)                      # right-handed: x = y × z
    R = np.stack([x, y, z])
    view = np.zeros((3, 4))
    view[:3, :3] = R
    view[:3, 3] = -R @ np.asarray(eye, float)
    return view


def project(points: np.ndarray, view: np.ndarray, focal: float,
            cx: float, cy: float, near: float = 1e-3):
    """Project (N,3) world points; returns (u, v, z, valid) with u/v in pixels."""
    cam = points @ view[:3, :3].T.astype(np.float32) + view[:3, 3].astype(np.float32)
    z = cam[:, 2]
    valid = z > near
    z_safe = np.where(valid, z, 1.0)
    u = focal * cam[:, 0] / z_safe + cx
    v = focal * cam[:, 1] / z_safe + cy
    return u, v, z, valid


def slerp_pose(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    """Interpolate two (4,4) cam-to-world poses (lerp position, slerp rotation).

    Keyframes are seconds apart; without the rotation slerp the drawn frustum
    would snap between them several times a second.
    """
    out = np.eye(4, dtype=np.float32)
    out[:3, 3] = (1.0 - u) * a[:3, 3] + u * b[:3, 3]
    rel = a[:3, :3].T @ b[:3, :3]
    rvec, _ = cv2.Rodrigues(rel.astype(np.float64))
    delta, _ = cv2.Rodrigues(rvec * float(u))
    out[:3, :3] = (a[:3, :3].astype(np.float64) @ delta).astype(np.float32)
    return out


# ── timeline ──────────────────────────────────────────────────────────────────


@dataclass
class Timeline:
    """Maps video time onto the run: which snapshot's poses are current, how
    many frames of map have appeared, and where along the trajectory we are."""
    duration: float                 # video seconds (before --speed)
    snap_times: np.ndarray          # (K,) video time each snapshot appears
    snap_seq: np.ndarray            # (K,) last input-frame index of each snapshot
    frame_times: np.ndarray         # (F,) reveal time of every recorded frame
    frame_snap: np.ndarray          # (F,) snapshot index owning each frame
    source_fps: float
    pacing: str

    def seq_time(self, seq: np.ndarray | float):
        """Video time at which input frame `seq` was captured."""
        if self.pacing in ("capture", "realtime"):
            return np.asarray(seq, dtype=np.float64) / self.source_fps
        return np.interp(seq, self.snap_seq, self.snap_times)

    def state(self, t: float) -> tuple[int, int, float]:
        """(snapshot index, number of revealed map frames, input-frame position)."""
        k = int(np.searchsorted(self.snap_times, t, side="right")) - 1
        n_frames = int(np.searchsorted(self.frame_times, t, side="right"))
        if self.pacing in ("capture", "realtime"):
            seq_pos = t * self.source_fps
        else:
            seq_pos = float(np.interp(t, self.snap_times, self.snap_seq))
        return k, n_frames, seq_pos


def build_timeline(rec: Recording, pacing: str, growth: float,
                   tail: float) -> Timeline:
    """Lay the recorded snapshots out on the chosen clock.

    `growth` staggers a submap's frames over a short window after it lands, so
    the cloud grows visibly instead of popping in one block per submap (the data
    is identical — only the drawing is animated).
    """
    snaps = rec.snapshots
    snap_times, snap_seq = [], []
    frame_times, frame_snap = [], []
    for k, snap in enumerate(snaps):
        last_seq = max((seq for seq, _, _ in snap.frames), default=0)
        if pacing == "capture":
            t_k = last_seq / rec.source_fps
        else:
            # wall / realtime: when the submap actually became available.  On a
            # paced recording that shares a clock with capture time, so the gap
            # between the camera and the map on screen is the real latency.
            t_k = snap.wall_time
        snap_times.append(t_k)
        snap_seq.append(last_seq)
        n = max(len(snap.frames), 1)
        for i, _ in enumerate(snap.frames):
            frame_times.append(t_k + growth * i / n)
            frame_snap.append(k)

    # Both must be non-decreasing: the end-of-run refresh re-emits the last
    # submap with no new frames (last_seq 0), and np.interp needs monotonic xp.
    snap_times = np.maximum.accumulate(np.asarray(snap_times, dtype=np.float64))
    snap_seq = np.maximum.accumulate(np.asarray(snap_seq, dtype=np.float64))
    frame_times = np.maximum.accumulate(np.asarray(frame_times, dtype=np.float64))
    if pacing in ("capture", "realtime") and rec.image_paths:
        duration = len(rec.image_paths) / rec.source_fps
    else:
        duration = float(snap_times[-1]) if len(snap_times) else 0.0
    duration = max(duration, float(frame_times[-1]) if len(frame_times) else 0.0)
    return Timeline(duration + tail, snap_times, snap_seq, frame_times,
                    np.asarray(frame_snap, dtype=np.int32), rec.source_fps, pacing)


# ── renderer ──────────────────────────────────────────────────────────────────


class SceneRenderer:
    """Painter's-algorithm point-cloud renderer with a fading camera trail."""

    def __init__(self, rec: Recording, timeline: Timeline, args):
        self.rec = rec
        self.tl = timeline
        self.args = args
        self.width, self.height = args.width, args.height
        self.focal = 0.5 * self.height / math.tan(math.radians(args.fov) / 2)
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self.bg = np.array(_parse_color(args.bg_color), dtype=np.float32)
        self.trail_color = np.array(_parse_color(args.trail_color), dtype=np.float32)

        self._world_cache: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._follow_eye: np.ndarray | None = None
        self._follow_target: np.ndarray | None = None
        self._centre, self._distance = None, None
        self._from, self._ease_index, self._ease_start = None, -1, 0.0
        self._display: tuple[dict, dict] | None = None
        self._display_t: float | None = None
        self._dt = 0.0
        # Snap threshold for the correction ease: ~1 screen pixel, since the
        # fit scales the trajectory's extent to the frame either way.
        extent = np.array([pose[:3, 3] for snapshot in rec.snapshots
                           for pose in snapshot.poses.values()][-4096:] or [[0, 0, 0]])
        self._settle = max(float(np.ptp(extent, axis=0).max()) * 1e-3, 1e-6)
        self._azim = (self._auto_azimuth() if args.azim is None else float(args.azim))
        # Live pose stream, when the run was tracked (frontend/tracker.py).
        self._live = (timeline.pacing == "realtime" and len(rec.tracked_poses) > 0)
        self._live_path, self._live_valid = self._build_live_path()
        if args.live_camera == "auto":
            # One trail, drawn live and corrected in place, whenever there is a
            # live stream at all.  Coverage no longer decides this: the old
            # threshold existed because a dropout handed the marker back to the
            # map and teleported it by the whole latency, which `_advance`
            # removes by rate-limiting the marker instead of jumping it.
            args.live_camera = "live" if self._live else "map"
        if args.live_camera == "tracker":
            args.live_camera = "live"          # old name for the same thing
        self._latency = self._map_latency()
        self._frontier_t, self._frontier_seq = self._build_frontier()
        self._camera_path: dict[int, np.ndarray] = {}
        coverage = len(rec.tracked_poses) / max(len(rec.image_paths), 1)
        self._init_live_trail()
        self._apply_live_trust()
        if timeline.pacing == "realtime":
            print(f"[video] map latency {self._latency:.1f}s, tracker covers "
                  f"{100 * coverage:.0f}% — camera follows "
                  f"{'the live stream' if args.live_camera == 'live' else 'the map'}"
                  f", corrections eased over {args.correction_ease:.1f}s")

        # Framing is solved twice because the two choices depend on each other:
        # the inset corner is scored against a provisional framing, then the
        # final framing is fitted to the area that corner (and the HUD) leaves.
        self.tan_x, self.tan_y = self.cx / self.focal, self.cy / self.focal
        self._framings = [self._fit(k) for k in range(len(rec.snapshots))]
        self._inset_corner = (self._pick_inset_corner() if args.inset == "auto"
                              else args.inset)
        self._set_safe_area()
        self._framings = [self._fit(k) for k in range(len(rec.snapshots))]
        self.radius = self._framings[-1][2]

    # -- viewpoint ----------------------------------------------------------

    def _auto_azimuth(self) -> float:
        """Look across the trajectory, not along it.

        A corridor / street run seen end-on projects to a narrow strip that
        wastes most of the frame (and hides the shape of the path).  The
        principal horizontal axis of the trajectory is therefore put across the
        screen: the view direction is its perpendicular.
        """
        positions = np.stack([pose[:3, 3]
                              for pose in self.rec.snapshots[-1].poses.values()])
        ground = positions[:, [0, 2]] - positions[:, [0, 2]].mean(axis=0)  # x, z
        if len(ground) < 3:
            return 35.0
        principal = np.linalg.svd(ground, full_matrices=False)[2][0]
        return math.degrees(math.atan2(principal[0], principal[1])) + 90.0

    def _direction(self, t: float) -> np.ndarray:
        """Unit vector from the look-at target towards the eye."""
        spin = self.args.orbit_speed * t if self.args.view == "orbit" else 0.0
        azim, elev = math.radians(self._azim + spin), math.radians(self.args.elev)
        return np.array([math.cos(elev) * math.sin(azim),
                         -math.sin(elev),            # y is down, so -y is up
                         math.cos(elev) * math.cos(azim)])

    def _sample(self, k: int, budget: int = 8000):
        """A cheap subsample of the map as of pose version `k`, in world space.

        Framing is recomputed for every snapshot, so it has to be cheap —
        re-projecting the full cloud K times would cost more than the render.
        """
        snap = self.rec.snapshots[k]
        per_frame = max(1, budget // max(sum(len(earlier.frames)
                                             for earlier in self.rec.snapshots[:k + 1]), 1))
        out = []
        for earlier in self.rec.snapshots[:k + 1]:
            for seq_idx, points, _ in earlier.frames:
                pose = snap.poses.get(seq_idx)
                if pose is None or not len(points):
                    continue
                stride = max(1, len(points) // per_frame)
                rot = (pose[:3, :3] * np.float32(snap.scales.get(seq_idx, 1.0)))
                out.append(points[::stride] @ rot.T + pose[:3, 3])
        return (np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32))

    def _fit(self, k: int):
        """(centre, camera distance, radius) framing the map as of snapshot `k`.

        Fitted in the *view's own axes* rather than from a world bounding box:
        an elongated scene (a street, a corridor) seen from an angle otherwise
        overflows the frame or leaves most of it empty.  Percentiles, not
        min/max — DA3 emits a few very distant points and one of them would
        zoom the whole scene down to nothing.
        """
        snap = self.rec.snapshots[k]
        positions = np.stack([pose[:3, 3] for pose in snap.poses.values()]).astype(np.float64)
        if self._live and snap.poses:
            # The live trail is drawn too, and it does not sit inside the map:
            # on KITTI the tracker under-scales translation by 14%, so framing
            # on the map alone pushed the live line off the right-hand edge.
            # It runs ahead of the map by about one submap, which is what the
            # NEXT snapshot's coverage measures.
            ahead = self.rec.snapshots[min(k + 1, len(self.rec.snapshots) - 1)]
            limit = int(max(ahead.poses)) if ahead.poses else int(max(snap.poses))
            seen = self._live_ok.copy()
            seen[limit + 1:] = False
            if seen.any():
                positions = np.vstack([positions, self._live_xyz[seen].astype(np.float64)])
        pts = self._sample(k)
        R = look_at(self._direction(0.0), np.zeros(3))[:3, :3]
        cloud = (np.vstack([pts, positions]) if len(pts) else positions) @ R.T
        lo = np.percentile(cloud, 1.5, axis=0)
        hi = np.percentile(cloud, 98.5, axis=0)
        traj = positions @ R.T                      # never clip the trajectory
        lo, hi = np.minimum(lo, traj.min(axis=0)), np.maximum(hi, traj.max(axis=0))
        centre_cam = (hi + lo) / 2.0
        centre = R.T @ centre_cam
        local = cloud - centre_cam                  # camera-axes, about the centre
        tan_x, tan_y = self.tan_x, self.tan_y
        radius = float(np.percentile(np.linalg.norm(local, axis=1),
                                     self.args.fit_percentile))
        if self.args.view == "orbit":
            # Must stay framed from every azimuth, so fit the bounding sphere.
            dist = radius / min(tan_x, tan_y)
        else:
            # Exact fit: a point at camera-axes (x, y, z) is inside the frustum
            # iff dist >= |x|/tan_x - z (and likewise in y), because its depth
            # from the eye is dist + z.  Taking a high percentile rather than
            # the max keeps a handful of DA3 flyers from pushing the camera
            # back until the map is a smudge — the bounding-box bound
            # (half/tan + half_depth) is far too loose for a deep scene.
            def needed(p):
                return np.maximum(np.abs(p[:, 0]) / tan_x,
                                  np.abs(p[:, 1]) / tan_y) - p[:, 2]
            # The trajectory is the subject of the video, so it is fitted
            # exactly; the cloud only has to fit to `fit_percentile`.
            dist = max(float(np.percentile(needed(local), self.args.fit_percentile)),
                       float(needed(traj - centre_cam).max()))
        return centre, float(max(dist * self.args.zoom_out, 1e-3)), radius

    def _fitted_view(self, t: float, k: int) -> np.ndarray:
        """Framed view, eased towards the current framing.

        With `--framing grow` the view zooms out as the map does (the first
        submap fills the frame instead of being three pixels wide); the easing
        keeps that from snapping on every new submap.

        The ease runs for a fixed `--framing_ease` and then holds the target
        EXACTLY.  An exponential decay would be the obvious choice but never
        arrives: the view keeps creeping a hundredth of a pixel per frame for
        the rest of the shot, which is invisible yet flips individual points on
        and off all over the cloud — the encoder cannot skip a single block and
        the file more than doubles.  Settling makes the map pixel-identical
        between submaps, which is both cheaper and steadier to look at.
        """
        index = (max(k, 0) if self.args.framing == "grow"
                 else len(self._framings) - 1)
        centre, distance, _ = self._framings[index]
        if self._centre is None:                       # first frame: no ease
            self._from = (centre, distance)
            self._ease_index, self._ease_start = index, t
        elif index != self._ease_index:                # new framing: ease to it
            self._from = (self._centre, self._distance)
            self._ease_index, self._ease_start = index, t
        u = min(1.0, (t - self._ease_start) / max(self.args.framing_ease, 1e-6))
        u = u * u * (3.0 - 2.0 * u)                    # smoothstep in/out
        start_c, start_d = self._from
        self._centre = start_c + (centre - start_c) * u
        self._distance = start_d + (distance - start_d) * u
        return look_at(self._centre + self._direction(t) * self._distance,
                       self._centre)

    def _follow_view(self, pose: np.ndarray) -> np.ndarray:
        """Chase camera, exponentially smoothed (raw keyframe poses are jittery)."""
        dist = self.args.follow_distance or 0.35 * self.radius
        height = self.args.follow_height or 0.15 * self.radius
        forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
        eye = pose[:3, 3] - forward * dist + WORLD_UP * height
        target = pose[:3, 3] + forward * dist * 0.5
        alpha = self.args.follow_smoothing
        self._follow_eye = (eye if self._follow_eye is None
                            else alpha * self._follow_eye + (1 - alpha) * eye)
        self._follow_target = (target if self._follow_target is None
                               else alpha * self._follow_target + (1 - alpha) * target)
        return look_at(self._follow_eye, self._follow_target)

    def _set_safe_area(self) -> None:
        """Keep the map clear of the HUD bars and the input inset.

        Done by moving the *principal point* to the centre of the free area and
        fitting to its half-extents: an off-axis projection shifts the map over
        without distorting it, which a post-hoc pan could not do (the shift
        would depend on depth).
        """
        pad = 18
        x_lo, x_hi = 0.0, float(self.width)
        y_lo, y_hi = 0.0, float(self.height)
        if self.args.hud:
            y_lo, y_hi = 64.0 + pad, self.height - 44.0 - pad
        if self._inset_corner != "none" and self.rec.image_paths:
            w = self.width * self.args.inset_scale + 2 * pad
            if self._inset_corner in ("tl", "bl"):
                x_lo += w
            else:
                x_hi -= w
        self.cx, self.cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
        self.tan_x = (x_hi - x_lo) / 2.0 / self.focal
        self.tan_y = (y_hi - y_lo) / 2.0 / self.focal

    # -- map points ---------------------------------------------------------

    def _tick(self, t: float) -> float:
        """Easing factor for this video frame, 0..1.

        Exponential approach on a time constant that puts ~95% of a correction
        inside `correction_ease`.  Returns 0 on the first frame (and on a seek),
        so whatever is displayed is adopted rather than eased from nothing.
        """
        if self._display_t is None or t < self._display_t:
            self._display_t, self._dt = t, 0.0
            return 0.0
        dt, self._display_t = t - self._display_t, t
        self._dt = dt
        if self.args.correction_ease <= 0:
            return 1.0
        return 1.0 - math.exp(-3.0 * dt / self.args.correction_ease)

    def _display_poses(self, k: int, alpha: float):
        """(poses, scales, moving) for the cloud at this frame.

        A pose-graph correction arrives as a step: every keyframe moves at once,
        so the cloud teleports into its new shape.  Easing across it turns the
        correction into what it physically is — the whole estimate bending into
        a better fit.

        The ease runs from *what is currently on screen* towards the newest
        snapshot, not between consecutive snapshots: a correction landing while
        the previous one is still in flight would otherwise snap the cloud back
        to the older shape first, and the longer the ease window the bigger that
        snap-back (measured on KITTI 06: 5.61 m regardless of the window).  It
        hard-snaps once the remainder is under a pixel — an ease that never
        arrives creeps the whole cloud sub-pixel forever, which is invisible but
        doubles the encoded size.
        """
        snap = self.rec.snapshots[k]
        if self.args.correction_ease <= 0:
            return snap.poses, snap.scales, False
        if self._display is None or alpha <= 0.0:
            self._display = (dict(snap.poses), dict(snap.scales))
            return snap.poses, snap.scales, False
        held_p, held_s = self._display
        poses, scales, moving = {}, {}, False
        for seq, target in snap.poses.items():
            current = held_p.get(seq)
            goal = float(snap.scales.get(seq, 1.0))
            if current is None:          # new keyframe: nothing to morph from
                poses[seq], scales[seq] = target, goal
                continue
            have = float(held_s.get(seq, goal))
            if (np.linalg.norm(target[:3, 3] - current[:3, 3]) <= self._settle
                    and np.linalg.norm(target[:3, :3] - current[:3, :3]) <= 1e-3
                    and abs(goal - have) <= 1e-6):
                poses[seq], scales[seq] = target, goal
                continue
            poses[seq] = slerp_pose(current, target, alpha)
            scales[seq] = have + (goal - have) * alpha
            moving = True
        self._display = (poses, scales)
        return poses, scales, moving

    def _world_points(self, k: int, poses=None, scales=None):
        """World points for pose version `k`, in reveal order, plus the running
        per-frame count so a prefix slice is exactly "what has appeared so far".

        Everything is rebuilt whenever the pose version changes — that is how
        loop-closure corrections reach already-drawn geometry.  Only one version
        is kept resident.
        """
        morphing = poses is not None
        if not morphing and self._world_cache is not None and self._world_cache[0] == k:
            return self._world_cache[1], self._world_cache[2], self._world_cache[3]
        snap = self.rec.snapshots[k]
        if poses is None:
            poses, scales = snap.poses, snap.scales
        chunks_p, chunks_c, counts = [], [], []
        for earlier in self.rec.snapshots[: k + 1]:
            for seq_idx, points, colors in earlier.frames:
                pose = poses.get(seq_idx)
                if pose is None:
                    counts.append(0)
                    continue
                # world = s·R·p + t — keyframe_poses are rigid, so the Sim(3)
                # scale must be folded in or each frame sits at raw DA3 depth.
                scale = np.float32(scales.get(seq_idx, 1.0))
                rot = (pose[:3, :3] * scale).astype(np.float32)
                chunks_p.append(points @ rot.T + pose[:3, 3].astype(np.float32))
                chunks_c.append(colors)
                counts.append(len(points))
        pts = (np.concatenate(chunks_p) if chunks_p
               else np.zeros((0, 3), dtype=np.float32))
        cols = (np.concatenate(chunks_c) if chunks_c
                else np.zeros((0, 3), dtype=np.uint8))
        cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        if not morphing:          # only the settled version is worth caching
            self._world_cache = (k, pts, cols, cum)
        return pts, cols, cum

    def _draw_points(self, img: np.ndarray, pts: np.ndarray, cols: np.ndarray,
                     view: np.ndarray) -> None:
        u, v, z, valid = project(pts, view, self.focal, self.cx, self.cy)
        size = self.args.point_size
        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
        keep = (valid & (ui >= 0) & (ui < self.width - size + 1)
                & (vi >= 0) & (vi < self.height - size + 1))
        ui, vi, zk = ui[keep], vi[keep], z[keep]
        colors = cols[keep].astype(np.float32)
        if not len(ui):
            return
        # Painter's algorithm: far to near, so nearer points overwrite.
        order = np.argsort(-zk)
        ui, vi, zk, colors = ui[order], vi[order], zk[order], colors[order]
        if self.args.fog > 0:
            # Fade distant points toward the background — pure parallax is a
            # weak depth cue on a static frame.
            far = np.percentile(zk, 95) if len(zk) > 10 else zk.max()
            shade = np.clip(1.0 - self.args.fog * (zk / max(far, 1e-6)), 0.25, 1.0)
            colors = colors * shade[:, None] + self.bg * (1.0 - shade[:, None])
        colors = colors.astype(np.uint8)
        for dy in range(size):
            for dx in range(size):
                img[vi + dy, ui + dx] = colors

    # -- trajectory ---------------------------------------------------------

    def _draw_trail(self, img: np.ndarray, verts: np.ndarray, ages: np.ndarray,
                    view: np.ndarray) -> None:
        """Polyline whose opacity decays with age (newest = fully opaque).

        Segments are stamped into a single-channel alpha mask (oldest first, so
        newer segments win) and composited in one pass — per-segment
        alpha-blending of the full frame would be far too slow.
        """
        if len(verts) < 2:
            return
        u, v, _, valid = project(verts, view, self.focal, self.cx, self.cy)
        alphas = np.clip(1.0 - ages / max(self.args.trail_seconds, 1e-6), 0.0, 1.0)
        alphas = self.args.trail_min_alpha + (1.0 - self.args.trail_min_alpha) * alphas
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        thick = self.args.trail_thickness
        for i in range(len(verts) - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            a = float(min(alphas[i], alphas[i + 1]))
            if a <= 0.01:
                continue
            cv2.line(mask, (int(u[i]), int(v[i])), (int(u[i + 1]), int(v[i + 1])),
                     int(a * 255), thick, cv2.LINE_AA)
        if self.args.trail_glow > 0:
            glow = cv2.GaussianBlur(mask, (0, 0), thick * 2.5)
            mask = np.maximum(mask, (glow * self.args.trail_glow).astype(np.uint8))
        m = (mask.astype(np.float32) / 255.0)[:, :, None]
        np.copyto(img, (img.astype(np.float32) * (1 - m)
                        + self.trail_color * m).astype(np.uint8))

    def _draw_camera(self, img: np.ndarray, pose: np.ndarray,
                     view: np.ndarray, live: bool = True) -> None:
        """Wireframe frustum + a marker at the current camera position.

        When the tracker has no pose for this frame the marker is driven by the
        map instead, which means it holds and then follows the frontier.  That
        motion is honest but it reads as a frozen video unless the marker says
        so, so a map-driven marker is drawn dimmed and hollow.  Dropouts are
        multi-second when they happen at all (teddy 3.7-4.0 s, KITTI up to
        32 s), so there is nothing short to interpolate across — the only
        honest options are to show the hold or to invent motion.
        """
        # Sized against the viewing distance, so the frustum keeps the same
        # apparent size while the framing zooms out with the map.
        d = self.args.frustum_scale * (self._distance or self.radius)
        corners = np.array([[0, 0, 0],
                            [-0.7 * d, -0.5 * d, d], [0.7 * d, -0.5 * d, d],
                            [0.7 * d, 0.5 * d, d], [-0.7 * d, 0.5 * d, d]],
                           dtype=np.float32)
        world = corners @ pose[:3, :3].T.astype(np.float32) + pose[:3, 3].astype(np.float32)
        u, v, _, valid = project(world, view, self.focal, self.cx, self.cy)
        color = _parse_color(self.args.camera_color)
        if not live:
            color = tuple(int(round(c * 0.45 + 18)) for c in color)
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        for i, j in edges:
            if valid[i] and valid[j]:
                cv2.line(img, (int(u[i]), int(v[i])), (int(u[j]), int(v[j])),
                         color, 2, cv2.LINE_AA)
        if valid[0]:
            if live:
                cv2.circle(img, (int(u[0]), int(v[0])), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(u[0]), int(v[0])), 9, color, 1, cv2.LINE_AA)

    # -- overlays -----------------------------------------------------------

    def _inset_rect(self, corner: str, w: int, h: int) -> tuple[int, int]:
        pad = 18
        bar = 64 + pad if self.args.hud else pad      # clear of the HUD bars
        x0 = pad if corner in ("tl", "bl") else self.width - w - pad
        y0 = bar if corner in ("tl", "tr") else self.height - h - bar
        return x0, y0

    def _pick_inset_corner(self) -> str:
        """Put the input inset in whichever corner the map uses least.

        The framing fills the frame with the map, so a fixed corner reliably
        ends up covering the part that matters — usually the live camera, since
        that is wherever the trajectory currently ends.  Every snapshot's
        framing is scored, not just the final one: the map is smaller (and the
        path elsewhere) for most of the video.
        """
        w = int(self.width * self.args.inset_scale)
        h = int(w * 4 / 3)                            # portrait: the worst case
        rects = {c: self._inset_rect(c, w, h) for c in ("br", "tr", "bl", "tl")}
        cost = dict.fromkeys(rects, 0.0)
        for k in range(len(self.rec.snapshots)):
            centre, distance, _ = self._framings[k]
            view = look_at(centre + self._direction(0.0) * distance, centre)
            pts = self._sample(k, budget=4000)
            traj = np.stack([pose[:3, 3]
                             for pose in self.rec.snapshots[k].poses.values()])
            for cloud, weight in ((pts, 1.0), (traj, 60.0)):
                u, v, _, ok = project(cloud, view, self.focal, self.cx, self.cy)
                for corner, (x0, y0) in rects.items():
                    hit = ok & (u >= x0) & (u < x0 + w) & (v >= y0) & (v < y0 + h)
                    cost[corner] += weight * float(hit.sum()) / max(len(cloud), 1)
        return min(cost, key=cost.get)

    def _draw_inset(self, img: np.ndarray, seq: int) -> None:
        paths = self.rec.image_paths
        if not paths or self._inset_corner == "none":
            return
        seq = int(np.clip(seq, 0, len(paths) - 1))
        path = paths[seq]
        if self.args.inset_dir:
            path = str(Path(self.args.inset_dir) / Path(path).name)
        frame = cv2.imread(path)
        if frame is None:
            return
        w = int(self.width * self.args.inset_scale)
        h = int(w * frame.shape[0] / frame.shape[1])
        frame = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
        x0, y0 = self._inset_rect(self._inset_corner, w, h)
        img[y0:y0 + h, x0:x0 + w] = frame
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (210, 210, 220), 1)
        _label(img, "input", (x0 + 6, y0 + h - 8), 0.45, (230, 230, 240))

    def _draw_hud(self, img: np.ndarray, t: float, stats: dict) -> None:
        if not self.args.hud:
            return
        _panel(img, 0, 0, self.width, 64, 0.55)
        _label(img, self.args.title, (22, 42), 0.95, (255, 255, 255), 2)
        # realtime shares the capture clock (only the map's arrival differs),
        # so it reads as elapsed time; only 'wall' is a processing-time replay.
        right = (f"wall = {t:6.2f}s" if self.tl.pacing == "wall"
                 else f"t = {t:6.2f}s")
        _label(img, right, (self.width - 190, 40), 0.7, (200, 210, 230), 1)
        # Loop closures are deliberately not reported on screen (neither a
        # per-event banner nor a count) — the map snapping into place says it.
        line = (f"keyframes {stats['keyframes']:4d}   "
                f"submaps {stats['submaps']:3d}   "
                f"points {stats['points']:,}")
        _panel(img, 0, self.height - 44, self.width, 44, 0.55)
        _label(img, line, (22, self.height - 16), 0.62, (210, 220, 235), 1)

    # -- one frame ----------------------------------------------------------

    def render(self, t: float) -> np.ndarray:
        k, n_frames, seq_pos = self.tl.state(t)
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.bg.astype(np.uint8)
        if k < 0:
            # Before the first submap there is no map to draw — but with the
            # bootstrap ladder there IS a live estimate, and this is precisely
            # the window the ladder exists to fill.  Drawing nothing here threw
            # the whole gain away: on chess the tracker is solving from 4.3 s
            # while the first submap lands at 7.3 s.
            if self._live:
                verts, ages, pose = self._advance(
                    self.rec.snapshots[0], seq_pos, {}, self._tick(t))
                if len(verts):
                    view = self._fitted_view(t, 0)
                    self._draw_trail(img, verts, ages, view)
                    if pose is not None:
                        self._draw_camera(img, pose, view, self._marker_live)
            self._draw_inset(img, seq_pos)
            self._draw_hud(img, t, {"keyframes": 0, "submaps": 0, "points": 0})
            return img

        snap = self.rec.snapshots[k]
        alpha = self._tick(t)
        poses, scales, morphing = self._display_poses(k, alpha)
        pts, cols, cum = (self._world_points(k, poses, scales) if morphing
                          else self._world_points(k))
        n_frames = min(n_frames, len(cum) - 1)
        end = int(cum[n_frames])
        if self.args.max_points and end > self.args.max_points:
            # Stable thinning: keep a *prefix* of each frame's (already
            # shuffled) points so the cloud thins smoothly instead of flickering
            # a new random subset every frame.  Cheap approximation: stride the
            # concatenated prefix, which is stable for a fixed `end`.
            stride = int(math.ceil(end / self.args.max_points))
            vis_pts, vis_cols = pts[:end:stride], cols[:end:stride]
        else:
            vis_pts, vis_cols = pts[:end], cols[:end]

        if self._live:
            verts, ages, pose = self._advance(snap, seq_pos, poses, alpha)
            camera_seq = self._cam_seq
        else:
            camera_seq = self._camera_seq(snap, seq_pos)
            pose = (self._camera_path.get(int(round(t * self.args.fps / self.args.speed)))
                    if self._camera_path else None)
            if pose is None:
                pose = self._camera_pose(snap, camera_seq)
        if self.args.view == "follow" and pose is not None:
            view = self._follow_view(pose)
        else:
            view = self._fitted_view(t, k)

        self._draw_points(img, vis_pts, vis_cols, view)
        if not self._live:
            verts, ages = self._trail(snap, camera_seq, t, pose, poses)
        self._draw_trail(img, verts, ages, view)
        if pose is not None:
            self._draw_camera(img, pose, view,
                              self._marker_live if self._live else True)
        self._draw_inset(img, seq_pos)

        self._draw_hud(img, t, {
            "keyframes": sum(1 for seq in snap.poses if seq <= seq_pos),
            "submaps": snap.n_submaps,
            "points": len(vis_pts),
        })
        return img

    def _build_camera_path(self, n_frames: int):
        """Camera pose for every output frame, smoothed over the whole run.

        Built ahead of time rather than per frame because the two things that
        move the marker are both discontinuous: it would otherwise clamp to the
        newest keyframe and sit frozen until the next submap lands, and every
        pose-graph correction (a loop closure, a boundary rescale) teleports it
        by however much the map just moved — up to 2.8 m, measured.  A CENTRED
        smoothing pass turns both into short glides without adding lag, and it
        cannot mask the corrections themselves: the cloud and the trail still
        snap, which is what a correction looks like.
        """
        step = self.args.speed / self.args.fps
        poses, index = [], []
        for i in range(n_frames):
            k, _, seq_pos = self.tl.state(i * step)
            if k < 0:
                continue
            snap = self.rec.snapshots[k]
            pose = self._camera_pose(snap, self._camera_seq(snap, seq_pos))
            if pose is not None:
                poses.append(pose)
                index.append(i)
        if not poses:
            return {}
        translations = np.stack([pose[:3, 3] for pose in poses])
        rotations = np.stack([pose[:3, :3] for pose in poses])
        window = max(1, int(self.args.live_smoothing * self.args.fps) | 1)
        if window > 1 and len(poses) > window:
            pad = window // 2
            kernel = np.ones(window) / window
            padded = np.pad(translations, ((pad, pad), (0, 0)), mode="edge")
            translations = np.stack(
                [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)], 1)
            padded_r = np.pad(rotations, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            smoothed = np.empty_like(rotations)
            for j in range(len(rotations)):
                mean = padded_r[j:j + window].mean(axis=0)
                u, _, vt = np.linalg.svd(mean)          # nearest rotation
                smoothed[j] = u @ vt
            rotations = smoothed
        path = {}
        for j, i in enumerate(index):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rotations[j]
            pose[:3, 3] = translations[j]
            path[i] = pose
        return path

    def _map_latency(self) -> float:
        """Median delay between a frame being captured and its submap landing.

        On a paced recording wall_time and capture time share a clock, so this
        is the pipeline's real lag — what the gap between the camera marker and
        the input inset is showing.
        """
        lags = [snap.wall_time - max(q for q, _, _ in snap.frames) / self.rec.source_fps
                for snap in self.rec.snapshots if snap.frames]
        return float(np.median(lags)) if lags else 0.0

    def _build_frontier(self):
        """Knots for a continuous sweep of the map's own frontier.

        What the map knows advances as a staircase: nothing new until a submap
        lands, then a whole submap's worth at once.  Parking the camera at the
        frontier therefore freezes it between submaps and jerks it forward on
        each one, and a fixed time offset cannot fix that — set it small and
        the camera outruns the map and clamps anyway, set it large enough to
        never clamp and it trails by a submap interval on top of the latency.
        Instead the marker spends each interval sweeping the segment the
        previous submap revealed: continuous, always on geometry the map holds,
        and still strictly behind what the system knew.
        """
        times, newest = [], []
        for snap in self.rec.snapshots:
            if snap.poses:
                times.append(snap.wall_time)
                newest.append(max(snap.poses))
        if len(times) < 2:
            return np.array([0.0]), np.array(newest or [0.0], dtype=float)
        # value at the start of interval k is what interval k-1 revealed
        knots_t = np.array(times[1:] + [self.tl.duration], dtype=float)
        knots_seq = np.array(newest[:-1] + [newest[-1]], dtype=float)
        return knots_t, knots_seq

    def _build_live_path(self):
        """Per-frame camera path from the tracked poses: cleaned, filled, smoothed.

        Three things the raw stream needs before it can be looked at.  Gross
        PnP outliers (a single-frame 3.7 m jump was measured) are rejected
        against a robust speed bound.  Short dropouts are interpolated, since
        they are transient and the map replaces that stretch seconds later;
        long ones are left invalid so the camera falls back to the map's own
        (delayed) estimate rather than freezing or gliding through a fiction.
        Finally the path is smoothed with a CENTRED window — offline rendering
        can use the future, so the jitter goes away without the lag a causal
        filter would add.
        """
        if not self._live:
            return {}, np.zeros(0, dtype=bool)
        seqs = np.array(sorted(self.rec.tracked_poses), dtype=np.int64)
        positions = np.stack([self.rec.tracked_poses[int(seq)][:3, 3] for seq in seqs])
        rotations = np.stack([self.rec.tracked_poses[int(seq)][:3, :3] for seq in seqs])

        # reject isolated jumps: a robust bound on per-frame displacement
        step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        span = np.maximum(np.diff(seqs), 1)
        speed = step / span
        bound = max(6.0 * float(np.median(speed)), 0.05)
        keep = np.ones(len(seqs), dtype=bool)
        keep[1:] &= speed <= bound
        keep[:-1] &= speed <= bound
        seqs, positions, rotations = seqs[keep], positions[keep], rotations[keep]
        if len(seqs) < 2:
            return {}, np.zeros(0, dtype=bool)

        # resample onto every frame, marking which ones are actually supported
        first, last = int(seqs[0]), int(seqs[-1])
        grid = np.arange(first, last + 1)
        filled = np.stack([np.interp(grid, seqs, positions[:, i]) for i in range(3)], 1)
        gap_limit = self.args.live_gap * self.rec.source_fps
        valid = np.ones(len(grid), dtype=bool)
        for a, b in zip(seqs[:-1], seqs[1:]):
            if b - a > gap_limit:
                valid[int(a) - first + 1:int(b) - first] = False

        # centred smoothing of translation; rotation follows the nearest sample
        window = max(1, int(self.args.live_smoothing * self.rec.source_fps) | 1)
        if window > 1:
            pad = window // 2
            padded = np.pad(filled, ((pad, pad), (0, 0)), mode="edge")
            kernel = np.ones(window) / window
            filled = np.stack([np.convolve(padded[:, i], kernel, mode="valid")
                               for i in range(3)], axis=1)
        nearest = np.searchsorted(seqs, grid).clip(0, len(seqs) - 1)
        frames = rotations[nearest]
        if window > 1 and len(frames) > window:
            # The marker takes its orientation from here, so nearest-sample
            # rotations make the frustum flick between neighbouring keyframes.
            # Centred mean + nearest rotation (SVD), same window as the path.
            pad = window // 2
            padded = np.pad(frames, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            smoothed = np.empty_like(frames)
            for j in range(len(frames)):
                u_, _, vt = np.linalg.svd(padded[j:j + window].mean(axis=0))
                smoothed[j] = u_ @ vt
            frames = smoothed
        path = {}
        for i, seq in enumerate(grid):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = frames[i]
            pose[:3, 3] = filled[i]
            path[int(seq)] = pose
        return path, valid

    # -- the live trail -----------------------------------------------------

    def _init_live_trail(self) -> None:
        """Per-frame arrays backing the single, continuously corrected trail.

        The trail is one vertex per *input frame*, not per keyframe, and a
        vertex is never replaced — only moved.  While the pipeline is still
        working on a stretch, its vertices sit where the tracker put them in
        real time; when the submap covering them finally lands, their target
        becomes the optimised estimate and they ease into it.  That is the
        whole point: the line the viewer watched being drawn live is the same
        line that then bends into the corrected shape, instead of the live
        stretch being deleted and an optimised one appearing in its place.
        """
        n = len(self.rec.image_paths)
        self._n_input = n
        self._live_xyz = np.zeros((n, 3), dtype=np.float32)
        self._live_ok = np.zeros(n, dtype=bool)
        if self._live and self._live_path:
            first = min(self._live_path)
            for seq, pose in self._live_path.items():
                if not 0 <= seq < n:
                    continue
                index = seq - first
                if 0 <= index < len(self._live_valid) and not self._live_valid[index]:
                    continue
                self._live_xyz[seq], self._live_ok[seq] = pose[:3, 3], True
        self._vert_xyz = np.zeros((n, 3), dtype=np.float32)
        self._vert_on = np.zeros(n, dtype=bool)
        self._cam_seq: float | None = None
        self._marker_live = True
        self._prev_seq_pos: float | None = None

    def _apply_live_trust(self) -> None:
        """Drop the live stream when it is worse than the gap it fills.

        The tail exists to show the stretch the map has not reached — one map
        latency of motion.  If the tracker's own error is bigger than that
        stretch, the tail conveys nothing and actively misleads: on KITTI the
        live line sits a median 59.7 m from where the map later puts the same
        frames, against the 48.8 m the camera actually covers in that time, so
        the marker rides a position the map then contradicts.  Measured ratios:
        chess 0.02, Replica 0.09, teddy 0.15, KITTI 1.22.

        The map always wins once it arrives — that is already true per vertex —
        and this extends the same rule to the part of the trail the map has not
        reached yet: do not show a live estimate that cannot be believed.
        """
        if not self._live or self.args.live_trust == "always":
            return
        drop = self.args.live_trust == "never"
        ratio = float("nan")
        if not drop:
            final = self.rec.snapshots[-1].poses if self.rec.snapshots else {}
            seen = np.flatnonzero(self._live_ok)
            if len(final) >= 2 and len(seen) >= 8:
                keys = np.array(sorted(final), dtype=np.float64)
                pts = np.stack([final[int(k)][:3, 3] for k in keys]).astype(np.float64)
                mapped = np.stack([np.interp(seen, keys, pts[:, i])
                                   for i in range(3)], axis=1)
                error = float(np.median(np.linalg.norm(
                    self._live_xyz[seen] - mapped, axis=1)))
                speed = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)
                                        / np.maximum(np.diff(keys), 1)))
                span = speed * self._latency * self.rec.source_fps
                if span > 1e-9:
                    ratio = error / span
                    drop = ratio > self.args.live_trust_ratio
        if drop:
            self._live_ok[:] = False               # map-only trail and marker
            self.args.live_camera = "map"
            print(f"[video] live stream rejected (error/span {ratio:.2f} > "
                  f"{self.args.live_trust_ratio:.2f}) — the tracker is wrong by "
                  f"more than the gap it would fill, so the map drives "
                  f"everything")

    def _map_xyz(self, poses: dict, upto: int) -> np.ndarray:
        """The map's own position for every input frame 0..upto.

        Keyframes are sparse, so frames between them are interpolated along the
        optimised trajectory — the map has no opinion about them, and a
        straight segment is what the trail already drew between keyframes.
        """
        keys = np.array(sorted(poses), dtype=np.float64)
        pts = np.stack([poses[int(k)][:3, 3] for k in keys]).astype(np.float64)
        grid = np.arange(upto + 1, dtype=np.float64)
        return np.stack([np.interp(grid, keys, pts[:, i])
                         for i in range(3)], axis=1).astype(np.float32)

    def _advance(self, snap: Snapshot, seq_pos: float, poses: dict, alpha: float):
        """Move the trail one video frame on; return (verts, ages, head pose).

        Each vertex has exactly one target: the optimised estimate where the
        map has reached, the tracker's live pose ahead of that.  Switching a
        vertex's target from the second to the first is what a correction *is*,
        and easing every vertex at the same rate is what makes the whole line
        bend at once rather than being re-drawn.
        """
        n = min(int(seq_pos) + 1, self._n_input)
        if n <= 0:
            return np.zeros((0, 3), np.float32), np.zeros(0), None
        covered = min(int(max(poses)), n - 1) if poses else -1

        target = np.zeros((n, 3), dtype=np.float32)
        on = np.zeros(n, dtype=bool)
        if covered >= 0:
            target[:covered + 1] = self._map_xyz(poses, covered)
            on[:covered + 1] = True
        live = self._live_ok[:n].copy()
        live[:covered + 1] = False                 # the map outranks the tracker
        # A rolling live->map similarity was tried here and removed: the
        # disagreement is accumulating DRIFT, not a fixed gauge offset, so a
        # global fit did not reduce it (KITTI 59.7 -> 61.0 m) while every
        # re-fit shifted the whole tail at once (worst vertex move 7.4 ->
        # 14.8 m).  What the tail needs is the trust test below, not a fit.
        target[live] = self._live_xyz[:n][live]
        on |= live

        # (2) Interior gaps — frames with neither a live pose nor map coverage
        # — are interpolated between the vertices that do exist, so the line
        # stays continuous instead of popping into shape when the map finally
        # reaches them.  Only INTERIOR: never extrapolated past the newest
        # vertex, which would invent a head.
        if self.args.live_fill and on.sum() >= 2:
            known = np.flatnonzero(on)
            lo, hi = int(known[0]), int(known[-1])
            missing = ~on
            missing[:lo] = False
            missing[hi + 1:] = False
            if missing.any():
                holes = np.flatnonzero(missing)
                for axis in range(3):
                    target[holes, axis] = np.interp(holes, known,
                                                    target[known, axis])
                on |= missing

        fresh = on & ~self._vert_on[:n]            # no history: start on target
        self._vert_xyz[:n][fresh] = target[fresh]
        moving = on & self._vert_on[:n]
        if alpha > 0 and moving.any():
            current = self._vert_xyz[:n][moving]
            goal = target[moving]
            delta = goal - current
            near = np.linalg.norm(delta, axis=1) <= self._settle
            current = current + delta * alpha
            current[near] = goal[near]             # finite-time settle
            self._vert_xyz[:n][moving] = current
        self._vert_on[:n] |= on

        # The marker never moves backwards and never teleports.  It rides the
        # live head while the tracker has one; through a dropout it follows the
        # map's FRONTIER SWEEP rather than the map's coverage, because coverage
        # advances as a staircase — one submap at a time — and following it
        # directly freezes the marker for a whole submap interval (11 s on
        # KITTI, measured) and then lurches.  When tracking resumes the marker
        # closes the gap on a time constant instead of jumping the whole
        # latency (0.7-2.9 m, measured, on every drop and resume).
        step = 0.0 if self._prev_seq_pos is None else max(seq_pos - self._prev_seq_pos, 0.0)
        self._prev_seq_pos = seq_pos
        self._marker_live = bool(self.args.live_camera == "live"
                                 and self._live_ok[min(int(seq_pos), n - 1)])
        if self._marker_live:
            goal_seq = float(seq_pos)
        else:
            goal_seq = min(float(np.interp(self.tl.seq_time(seq_pos),
                                           self._frontier_t, self._frontier_seq)),
                           float(covered))
        goal_seq = min(goal_seq, float(n - 1))     # never past the last frame
        if self._cam_seq is None:
            self._cam_seq = max(goal_seq, 0.0)
        elif goal_seq > self._cam_seq:
            close = (goal_seq - self._cam_seq) * (
                1.0 - math.exp(-self._dt / max(self.args.live_catchup, 1e-6)))
            # Ceiling on the slide: an exponential close starts fast, and a
            # multi-second gap made the marker visibly sprint.  Capping it
            # trades a longer catch-up for no leap.
            if self.args.live_catchup_max > 0:
                close = min(close, step * max(self.args.live_catchup_max - 1.0, 0.0))
            self._cam_seq = min(goal_seq, self._cam_seq + step + close)

        # The trail may not run ahead of the marker.  It used to be drawn to
        # `seq_pos` while the marker sat at `_cam_seq`, so the line visibly
        # grew first and the camera chased it — the trail was showing poses the
        # marker had not reached.
        head = min(int(math.floor(self._cam_seq)) + 1, n)
        drawn = np.flatnonzero(self._vert_on[:head])
        verts = self._vert_xyz[:n][drawn]
        now = float(self.tl.seq_time(self._cam_seq))
        ages = now - np.asarray(self.tl.seq_time(drawn), dtype=np.float64)
        return verts, np.maximum(ages, 0.0), self._head_pose(snap, n)

    def _head_pose(self, snap: Snapshot, n: int) -> np.ndarray | None:
        """Camera pose at the marker: the trail's own vertex, plus a rotation.

        The position has to come from the trail, not from the pose source, or
        the marker drifts off the line it is supposed to be leading.
        """
        low = int(math.floor(self._cam_seq))
        if not 0 <= low < n or not self._vert_on[low]:
            return None
        high = min(low + 1, n - 1)
        u = self._cam_seq - low
        position = self._vert_xyz[low]
        if high > low and self._vert_on[high]:
            position = position * (1 - u) + self._vert_xyz[high] * u
        source = self._live_path.get(low) if self._live_ok[low] else None
        if source is None:
            source = self._camera_pose(snap, self._cam_seq)
        pose = np.eye(4, dtype=np.float32)
        if source is not None:
            pose[:3, :3] = source[:3, :3]
        pose[:3, 3] = position
        return pose

    def _camera_seq(self, snap: Snapshot, seq_pos: float) -> float:
        """Input-frame index the camera marker should sit at.

        `--live_camera map` puts it on the optimised trajectory, one measured
        latency behind the live frame.  That is deliberately NOT the tracker's
        answer: mixing the two means the marker teleports by the whole latency
        (0.7-2.9 m, measured) every time tracking drops or resumes, and no
        amount of smoothing hides a jump between two estimates seconds apart.
        Following one estimate keeps the marker continuous, keeps it on
        geometry the map actually has, and makes the gap to the input inset a
        readable measure of the lag.
        """
        return float(np.interp(self.tl.seq_time(seq_pos),
                               self._frontier_t, self._frontier_seq))

    def _camera_pose(self, snap: Snapshot, seq_pos: float) -> np.ndarray | None:
        """Current camera pose, interpolated between the bracketing keyframes."""
        seqs = sorted(seq for seq in snap.poses if seq <= seq_pos + 1e-9)
        if not seqs:
            return None
        last = seqs[-1]
        following = [seq for seq in snap.poses if seq > last]
        if not following:
            return snap.poses[last]
        nxt = min(following)
        u = float(np.clip((seq_pos - last) / max(nxt - last, 1e-9), 0.0, 1.0))
        return slerp_pose(snap.poses[last], snap.poses[nxt], u)

    def _trail(self, snap: Snapshot, seq_pos: float, t: float,
               pose: np.ndarray | None, poses: dict | None = None):
        """Trajectory vertices up to now, with each vertex's age in video seconds.

        Vertices are read from the *current* pose set, so the whole trail (not
        just its tip) moves when the graph is re-optimised.
        """
        poses = snap.poses if poses is None else poses
        seqs = np.array(sorted(seq for seq in poses if seq <= seq_pos + 1e-9))
        if not len(seqs):
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0)
        verts = np.stack([poses[seq][:3, 3] for seq in seqs]).astype(np.float32)
        # Age is measured back along the TRAJECTORY from wherever the camera
        # is, not from the wall clock: under realtime pacing the camera sits a
        # latency behind, so a clock-based age would put the whole trail past
        # the fade floor and lose the bright head entirely.
        now = float(self.tl.seq_time(seq_pos))
        ages = now - np.asarray(self.tl.seq_time(seqs), dtype=np.float64)
        if pose is not None:
            verts = np.vstack([verts, pose[:3, 3][None, :]])
            ages = np.append(ages, 0.0)
        return verts, np.maximum(ages, 0.0)


def _similarity(source: np.ndarray, target: np.ndarray):
    """Umeyama similarity [sR | t] taking `source` points onto `target`."""
    if len(source) < 3 or not (np.isfinite(source).all() and np.isfinite(target).all()):
        return None
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, sigma, vt = np.linalg.svd((b.T @ a) / len(source))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    variance = float((a ** 2).sum() / len(source))
    if variance <= 1e-12:
        return None
    scale = float((sigma * np.diag(d)).sum() / variance)
    if not np.isfinite(scale) or scale <= 0:
        return None
    out = np.eye(4)
    out[:3, :3] = scale * rotation
    out[:3, 3] = mu_t - scale * rotation @ mu_s
    return out


def _parse_color(spec) -> tuple[int, int, int]:
    if isinstance(spec, (tuple, list)):
        return tuple(int(c) for c in spec)
    parts = [int(c) for c in str(spec).replace("#", "").split(",")]
    return tuple(parts)


def _label(img, text, org, scale, color, thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                tuple(int(c) for c in color), thickness, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha) -> None:
    region = img[y:y + h, x:x + w]
    region[:] = (region.astype(np.float32) * (1 - alpha)).astype(np.uint8)


# ── encoding ──────────────────────────────────────────────────────────────────


class VideoOut:
    """H.264 through the ffmpeg binary when present, else OpenCV's mp4v.

    (The CUDA image has OpenCV but no ffmpeg CLI; the host usually has both.)
    """

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 crf: int = 18, frames_dir: Path | None = None,
                 bitrate_kbps: int | None = None, preset: str = "slow"):
        self.path, self.size, self.fps = path, size, fps
        self.frames_dir = frames_dir
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        try:  # fail here, not as a broken ffmpeg pipe 1500 frames later
            path.touch()
        except OSError as exc:
            raise SystemExit(
                f"[video] cannot write {path}: {exc}.  (outputs/ written from "
                f"Docker is root-owned — render inside the container, or pass "
                f"a --video path you own.)") from exc
        if frames_dir:
            frames_dir.mkdir(parents=True, exist_ok=True)
        self.proc = self.writer = None
        if shutil.which("ffmpeg"):
            # Capped CRF: encode for constant quality, but never let the
            # average bitrate exceed what --target_mb allows.  Plain ABR would
            # *spend* the whole budget even on the long static stretches this
            # renderer produces; capped CRF gives the smallest file that still
            # holds the requested quality, and only degrades where the content
            # is genuinely busy.
            rate = ["-crf", str(crf)]
            self.backend = f"ffmpeg/libx264 crf {crf}"
            if bitrate_kbps:
                # A generous VBV buffer (4 s) matters here: the bits are not
                # spread evenly at all — the map is pixel-static between
                # submaps and then every pixel moves for ~1 s while the framing
                # eases.  A tight buffer starves exactly those frames and
                # smears the points; over a whole file the average still holds.
                rate += ["-maxrate", f"{bitrate_kbps}k",
                         "-bufsize", f"{int(bitrate_kbps * 4)}k"]
                self.backend += f", capped at {bitrate_kbps} kbps"
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-",
                 "-an", "-c:v", "libx264", "-preset", preset, *rate,
                 # yuv420p progressive, and explicitly not interlaced: some
                 # submission portals reject anything else.
                 "-pix_fmt", "yuv420p", "-field_order", "progressive",
                 "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            if bitrate_kbps:
                print("[video] --target_mb needs the ffmpeg binary; OpenCV's "
                      "mp4v writer ignores it (render on the host instead)")
            self.writer = cv2.VideoWriter(str(path),
                                          cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open a video writer for {path}")
            self.backend = "opencv/mp4v"

    def write(self, rgb: np.ndarray) -> None:
        if self.frames_dir:
            cv2.imwrite(str(self.frames_dir / f"{self.count:06d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self.proc:
            self.proc.stdin.write(rgb.tobytes())
        else:
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.count += 1

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        if self.writer:
            self.writer.release()


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_video_args(argv: list[str]):
    """Parse the video-only flags; everything else is forwarded to run_slam."""
    p = argparse.ArgumentParser(
        description="Render a video of DASH-SLAM running (point cloud + fading "
                    "camera trail). Unknown flags are forwarded to run_slam.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", default="outputs/video/dash_slam.mp4",
                   help="Output .mp4 path")
    p.add_argument("--recording", default=None,
                   help="Snapshot cache (default: <video>.rec.pkl.gz). Written "
                        "by the run, reusable with --render_only")
    p.add_argument("--render_only", action="store_true",
                   help="Skip SLAM and render an existing --recording")
    p.add_argument("--record_only", action="store_true",
                   help="Run SLAM and save the recording without rendering")
    p.add_argument("--frames_dir", default=None,
                   help="Also dump every rendered frame as a PNG here")

    g = p.add_argument_group("timing")
    g.add_argument("--fps", type=float, default=30.0, help="Output video fps")
    g.add_argument("--source_fps", type=float, default=None,
                   help="Capture rate of the input frames (default: inferred "
                        "from numeric filenames, else 30)")
    g.add_argument("--pacing", choices=["auto", "capture", "wall", "realtime"],
                   default="auto",
                   help="'capture' plays the input's own clock and reveals the "
                        "map with zero latency; 'wall' replays the measured "
                        "processing times; 'realtime' is the honest one — the "
                        "input and the tracked camera run on the capture clock "
                        "while each submap appears when it really arrived, so "
                        "the map visibly trails the camera (needs a recording "
                        "made with --realtime --tracking).  'auto' picks "
                        "realtime when the recording supports it, else capture")
    g.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (2 = twice as fast)")
    g.add_argument("--live_camera", choices=["auto", "map", "live", "tracker"],
                   default="auto",
                   help="What the camera marker follows under --pacing "
                        "realtime: 'live' rides the tracker's head, holding "
                        "through dropouts and following the map's frontier "
                        "until tracking resumes; 'map' stays one measured "
                        "latency behind on the optimised trajectory. 'auto' "
                        "picks 'live' whenever the recording has a pose stream "
                        "('tracker' is the old name for 'live')")
    g.add_argument("--live_catchup", type=float, default=2.0,
                   help="Time constant (s) for closing the gap after a tracking "
                        "dropout, on top of real-time advance. Small = the "
                        "marker snaps forward, large = it stays behind")
    g.add_argument("--correction_ease", type=float, default=0.5,
                   help="Seconds over which a pose-graph correction is eased "
                        "into the trail and the cloud, instead of snapping. "
                        "0 = snap (the old behaviour)")
    g.add_argument("--live_trust", choices=["auto", "always", "never"],
                   default="auto",
                   help="Whether to draw the live stream at all. 'auto' keeps "
                        "it only while its error is smaller than the span it "
                        "is meant to reveal (see --live_trust_ratio)")
    g.add_argument("--live_trust_ratio", type=float, default=0.5,
                   help="Trust threshold: median live-vs-map error divided by "
                        "the distance the camera covers during one map latency")
    g.add_argument("--live_fill", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Interpolate trail vertices that have neither a live "
                        "pose nor map coverage yet, so the line stays "
                        "continuous instead of popping into shape")
    g.add_argument("--live_catchup_max", type=float, default=3.0,
                   help="Ceiling on how fast the marker closes a gap, as a "
                        "multiple of real time. 0 = uncapped")
    g.add_argument("--live_smoothing", type=float, default=0.25,
                   help="Centred smoothing window (s) on the tracked camera "
                        "path; 0 = raw. Offline rendering can use the future, "
                        "so this costs no lag")
    g.add_argument("--live_gap", type=float, default=1.0,
                   help="Tracker dropouts shorter than this (s) are "
                        "interpolated; longer ones fall back to the map's own "
                        "delayed pose")
    g.add_argument("--growth", type=float, default=0.6,
                   help="Seconds over which a new submap's points stream in")
    g.add_argument("--tail", type=float, default=2.0,
                   help="Extra seconds held on the finished map")

    g = p.add_argument_group("view")
    g.add_argument("--view", choices=["fit", "orbit", "follow"], default="fit",
                   help="fit: fixed framing on the final map; orbit: slow "
                        "rotation; follow: chase the camera")
    g.add_argument("--width", type=int, default=1280)
    g.add_argument("--height", type=int, default=720)
    g.add_argument("--fov", type=float, default=55.0, help="Vertical FOV (deg)")
    g.add_argument("--elev", type=float, default=28.0,
                   help="Viewpoint elevation above the scene (deg)")
    g.add_argument("--azim", type=float, default=None,
                   help="Viewpoint azimuth (deg); default: across the "
                        "trajectory's principal axis")
    g.add_argument("--fit_percentile", type=float, default=95.0,
                   help="Fraction of the cloud the framing must contain; the "
                        "rest (DA3 flyers, points next to the virtual camera) "
                        "is allowed off-frame")
    g.add_argument("--orbit_speed", type=float, default=6.0,
                   help="Orbit rate (deg per video second, --view orbit)")
    g.add_argument("--zoom_out", type=float, default=1.15,
                   help="Framing margin around the fitted map (1 = tight)")
    g.add_argument("--framing", choices=["grow", "final"], default="grow",
                   help="grow: zoom out as the map grows; final: hold the "
                        "framing of the finished map from the first frame")
    g.add_argument("--framing_ease", type=float, default=1.2,
                   help="Time constant of the framing easing, in timeline "
                        "seconds (--framing grow)")
    g.add_argument("--follow_distance", type=float, default=None)
    g.add_argument("--follow_height", type=float, default=None)
    g.add_argument("--follow_smoothing", type=float, default=0.93,
                   help="EMA factor for the chase camera (higher = smoother)")

    g = p.add_argument_group("look")
    g.add_argument("--point_size", type=int, default=2, help="Point size (px)")
    g.add_argument("--max_points", type=int, default=900_000,
                   help="Render-time cap on drawn points (0 = no cap)")
    g.add_argument("--points_per_submap", type=int, default=45_000,
                   help="Record-time point budget per submap")
    g.add_argument("--fog", type=float, default=0.45,
                   help="Distance fade of the cloud, 0 = off")
    g.add_argument("--bg_color", default="14,16,22")
    g.add_argument("--trail_color", default="90,210,255",
                   help="R,G,B of the trajectory trail")
    g.add_argument("--camera_color", default="255,214,110",
                   help="R,G,B of the camera frustum")
    g.add_argument("--frustum_scale", type=float, default=0.05,
                   help="Camera frustum size as a fraction of the view distance")
    g.add_argument("--trail_seconds", type=float, default=5.0,
                   help="Fade length: a position this old reaches minimum opacity")
    g.add_argument("--trail_min_alpha", type=float, default=0.18,
                   help="Opacity floor for old positions (0 = fade to nothing)")
    g.add_argument("--trail_thickness", type=int, default=3)
    g.add_argument("--trail_glow", type=float, default=0.45,
                   help="Soft glow around the trail, 0 = off")
    g.add_argument("--inset", choices=["auto", "none", "tl", "tr", "bl", "br"],
                   default="auto",
                   help="Corner for the input-frame inset; auto picks the one "
                        "the map covers least")
    g.add_argument("--inset_scale", type=float, default=0.20,
                   help="Inset width as a fraction of the video width")
    g.add_argument("--inset_dir", default=None,
                   help="Re-resolve the recorded input frames against this "
                        "directory (paths recorded inside Docker differ from "
                        "the host ones)")
    g.add_argument("--hud", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--title", default="DASH-SLAM")
    g.add_argument("--crf", type=int, default=18,
                   help="x264 quality (lower = better, 18 is visually lossless)")
    g.add_argument("--preset", default="slow",
                   help="x264 preset (slower = smaller file at the same "
                        "quality; encoding is not the bottleneck here)")
    g.add_argument("--target_mb", type=float, default=None,
                   help="Cap the file at this size (MB) — encodes at --crf and "
                        "only lowers quality if that would overshoot, so the "
                        "file stays as small as the quality allows. Needs the "
                        "ffmpeg binary (host, not the container)")

    return p.parse_known_args(argv)


def record_run(slam_argv: list[str], args) -> Recording:
    """Run the pipeline, capturing a snapshot per submap.

    run_slam's own parser is reused verbatim, so every SLAM flag it accepts
    (--submap_size, --no_loop_closure, --max_frames, ...) works here unchanged.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import yaml
    import run_slam
    globals()["run_slam"] = run_slam

    pre = argparse.ArgumentParser(add_help=False)
    from da3_slam.config import DEFAULT_YAML
    pre.add_argument("--config", default=str(DEFAULT_YAML))
    known, _ = pre.parse_known_args(slam_argv)
    with open(known.config) as f:
        yaml_config = yaml.safe_load(f)

    argv_backup = sys.argv
    sys.argv = [argv_backup[0]] + slam_argv
    try:
        slam_args = run_slam.parse_args(yaml_config)
    finally:
        sys.argv = argv_backup

    image_paths = run_slam.collect_image_paths(slam_args.image_dir,
                                               slam_args.max_frames)
    if not image_paths:
        sys.exit(f"No images found in {slam_args.image_dir}")
    print(f"[video] {len(image_paths)} images from {slam_args.image_dir}")

    config = run_slam.build_config(slam_args)
    config.build_pointclouds = True        # the video *is* the point cloud

    from da3_slam.slam import DA3SLAM
    recorder = RunRecorder(points_per_submap=args.points_per_submap)
    tracked: dict[int, np.ndarray] = {}

    def on_pose(seq_idx, pose, stats):
        tracked[int(seq_idx)] = np.asarray(pose, dtype=np.float32).copy()

    def on_pose_correction(transform, bound):
        """The bootstrap ladder's gauge has been reconciled with the map.

        Poses solved before the first submap existed are in DA3's units for a
        2-frame batch.  Without this the opening stretch of the live trail is
        drawn in the wrong frame and then yanked into place when the map
        reaches it, which is the one thing the trail is supposed not to do.
        """
        matrix = np.asarray(transform, dtype=np.float64)
        linear, shift = matrix[:3, :3], matrix[:3, 3]
        scale = float(np.cbrt(max(abs(np.linalg.det(linear)), 1e-18)))
        rotation = linear / scale if scale > 0 else linear
        for seq, pose in list(tracked.items()):
            fixed = np.eye(4, dtype=np.float32)
            fixed[:3, :3] = (rotation @ pose[:3, :3].astype(np.float64)).astype(np.float32)
            fixed[:3, 3] = (linear @ pose[:3, 3].astype(np.float64)
                            + shift).astype(np.float32)
            tracked[seq] = fixed
        print(f"[video] bootstrap gauge reconciled after frame {bound} "
              f"(scale {scale:.3f}, {len(tracked)} poses rewritten)")

    slam = DA3SLAM(config)
    fps = (args.source_fps or infer_source_fps(image_paths, 30.0))
    paced = slam_args.realtime is not None
    if paced:
        if slam_args.realtime > 0:
            fps = slam_args.realtime
        print(f"[video] pacing input at {fps:.2f} fps — the map's arrival times "
              f"become real measurements, and the tracker gets frames on the "
              f"camera's clock")
    t0 = time.time()
    recorder._t0 = t0                      # measure from the first frame, not model load
    if paced:
        result = slam.run_stream(run_slam._paced(slam._disk_frame_source(image_paths), fps),
                                 on_update=recorder, on_pose=on_pose,
                                 on_pose_correction=on_pose_correction)
    else:
        result = slam.run(image_paths, on_update=recorder, on_pose=on_pose,
                          on_pose_correction=on_pose_correction)
    wall = time.time() - t0
    print(f"[video] pipeline finished in {wall:.1f}s — "
          f"{result.n_keyframes} keyframes, {len(result.submaps)} submaps, "
          f"{len(result.loop_closures)} loop closures")

    if config.tracking.enable:
        print(f"[video] tracker: {len(tracked)} poses "
              f"({100 * len(tracked) / max(len(image_paths), 1):.0f}% of frames)")
    return Recording(
        snapshots=recorder.snapshots,
        image_paths=image_paths,
        source_fps=fps,
        total_wall=wall,
        label=Path(slam_args.image_dir).name,
        tracked_poses=tracked,
        paced=paced,
    )


def render(rec: Recording, args) -> None:
    timeline = build_timeline(rec, args.pacing, args.growth, args.tail)
    renderer = SceneRenderer(rec, timeline, args)
    n_frames = max(int(timeline.duration / args.speed * args.fps), 1)
    if not renderer._live:
        # The live trail builds its own marker, continuous by construction; the
        # pre-smoothed path exists to hide the map camera's staircase.
        renderer._camera_path = renderer._build_camera_path(n_frames)
    bitrate = None
    if args.target_mb:
        # 5% held back for container overhead.
        bitrate = int(args.target_mb * 8 * 1024 / (n_frames / args.fps) * 0.95)
    out = VideoOut(Path(args.video), (args.width, args.height), args.fps,
                   crf=args.crf, bitrate_kbps=bitrate, preset=args.preset,
                   frames_dir=Path(args.frames_dir) if args.frames_dir else None)
    print(f"[video] rendering {n_frames} frames "
          f"({timeline.duration / args.speed:.1f}s @ {args.fps:g} fps, "
          f"{args.width}x{args.height}, {out.backend})")
    start = time.time()
    for i in range(n_frames):
        out.write(renderer.render(i / args.fps * args.speed))
        if i % 25 == 0 or i == n_frames - 1:
            done = i + 1
            rate = done / max(time.time() - start, 1e-6)
            print(f"\r[video] frame {done}/{n_frames}  "
                  f"{rate:4.1f} fps  eta {(n_frames - done) / max(rate, 1e-6):5.0f}s",
                  end="", flush=True)
    out.close()
    print(f"\n[video] wrote {args.video} "
          f"({Path(args.video).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args, slam_argv = parse_video_args(sys.argv[1:])
    # yuv420p needs even dimensions.
    args.width -= args.width % 2
    args.height -= args.height % 2
    rec_path = Path(args.recording or
                    str(Path(args.video).with_suffix("")) + ".rec.pkl.gz")

    if args.render_only:
        if not rec_path.exists():
            sys.exit(f"No recording at {rec_path} — run without --render_only first")
        rec = load_recording(rec_path)
        print(f"[video] loaded {len(rec.snapshots)} snapshots from {rec_path}")
        if args.source_fps:
            rec.source_fps = args.source_fps
    else:
        rec = record_run(slam_argv, args)
        save_recording(rec, rec_path)

    if not rec.snapshots:
        sys.exit("[video] the run produced no submaps — nothing to render")
    if args.pacing == "auto":
        args.pacing = ("realtime" if rec.paced and rec.tracked_poses else "capture")
        print(f"[video] pacing: {args.pacing}"
              + ("" if args.pacing == "realtime" else
                 " (recording has no paced/tracked pose stream)"))
    if args.record_only:
        return
    render(rec, args)


if __name__ == "__main__":
    main()
