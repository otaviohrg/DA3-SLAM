"""
Rerun-based live viewer for DA3-SLAM.

A LiveViewer instance is a callable (SLAMUpdate → None) suitable for passing as
`on_update` to DA3SLAM.run_stream().  Each submap it receives is logged to a
Rerun recording: the growing point cloud, the camera trajectory, and the latest
camera pose.

Why Rerun (and how it works in Docker): the SDK in this process is a *client*
that streams log data to a *viewer* process over gRPC.  For the containerised
demo you run the `rerun` viewer on the host and the container connects out to
it — no X11 / OpenGL passthrough required.  Modes:

  * connect  (default) — connect to a viewer already running on the host at
               `addr` (e.g. `rerun+http://host.docker.internal:9876/proxy`).
               Start the host viewer with:  rerun
  * serve    — serve a web viewer from *this* process; open it in a browser on
               the host (expose the web + gRPC ports from the container).
  * spawn    — spawn a native viewer window in this process (bare-metal only;
               needs a display — not for headless Docker).

Requires `rerun-sdk` (see requirements-realsense.txt).  Imported lazily.
Rerun's Python API has shifted across releases; this targets rerun-sdk >= 0.20
(the gRPC-connect era) and degrades gracefully across nearby versions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Type-only: importing da3_slam.slam pulls in the full GPU stack, which the
    # viewer itself does not need.
    from da3_slam.slam import SLAMUpdate


class LiveViewer:
    """Callable that logs SLAMUpdate snapshots to a Rerun recording.

    Camera-space points of every submap are cached (downsampled) so that when
    a loop closure or later optimisation shifts already-drawn poses, the
    affected submaps are *re-projected and re-logged* — without this, corrected
    geometry would stay frozen at its pre-correction position and every loop
    closure would look like a duplicated map.
    """

    def __init__(
        self,
        mode: str = "connect",
        addr: str | None = None,
        max_points_per_submap: int = 60_000,
        app_id: str = "da3-slam",
        repose_threshold: float = 0.05,
    ):
        import rerun as rr  # lazy: only needed when a viewer is requested

        self._rr = rr
        self._max_points = max_points_per_submap
        self._rng = np.random.default_rng(0)

        # Minimum pose shift (translation norm + rotation Frobenius norm, in
        # scene units / ~metres) before a cached submap is re-logged.
        self._repose_threshold = repose_threshold
        # submap_idx → [(seq_idx, points_cam, colors)], downsampled at cache time
        self._submap_cache: dict[int, list[tuple[np.ndarray, ...]]] = {}
        # submap_idx → {seq_idx: (4, 4) pose the submap was last drawn with}
        self._logged_poses: dict[int, dict[int, np.ndarray]] = {}

        rr.init(app_id)
        if mode == "spawn":
            rr.spawn()
        elif mode == "serve":
            self._serve()
        elif mode == "connect":
            self._connect(addr)
        else:
            raise ValueError(f"unknown viewer mode: {mode!r}")

        # DA3 uses the OpenCV camera convention (x right, y down, z forward).
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self._set_time = self._resolve_set_time()
        print(f"[viewer] Rerun ready (mode={mode})")

    def _resolve_set_time(self):
        """Return a fn(seq_idx) that sets the 'submap' timeline, across rerun
        versions.  rerun >= 0.23 uses set_time(timeline, sequence=…); older
        builds use set_time_sequence(timeline, seq) (removed in 0.24).  Falls
        back to a no-op so a missing timeline API never blocks logging."""
        rr = self._rr
        if hasattr(rr, "set_time"):
            return lambda seq: rr.set_time("submap", sequence=seq)
        if hasattr(rr, "set_time_sequence"):
            return lambda seq: rr.set_time_sequence("submap", seq)
        return lambda seq: None

    # ── connection helpers ──────────────────────────────────────────────────

    def _connect(self, addr: str | None) -> None:
        """Connect to a running viewer, tolerating API renames across versions."""
        rr = self._rr
        # rerun >= 0.20 exposes connect_grpc (URL form); older builds used
        # connect_tcp / connect (host:port).  Try newest first.
        errors = []
        for fn_name in ("connect_grpc", "connect_tcp", "connect"):
            fn = getattr(rr, fn_name, None)
            if fn is None:
                continue
            try:
                fn(addr) if addr else fn()
                return
            except Exception as exc:  # signature/version mismatch → try next
                errors.append(f"{fn_name}: {exc!r}")
        raise RuntimeError(
            "Could not connect to a Rerun viewer. Start one on the host with "
            "`rerun` and check --viewer_addr. Attempts: " + "; ".join(errors)
        )

    def _serve(self) -> None:
        """Serve a web viewer from this process (browse it from the host)."""
        rr = self._rr
        for fn_name in ("serve_web", "serve"):
            fn = getattr(rr, fn_name, None)
            if fn is not None:
                fn()
                print("[viewer] serving Rerun web viewer — open the forwarded "
                      "web port in a browser on the host")
                return
        raise RuntimeError("This rerun-sdk has no serve_web()/serve().")

    # ── point cache / re-projection helpers ─────────────────────────────────

    @staticmethod
    def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Apply a (4, 4) cam-to-world pose to (M, 3) camera-space points."""
        return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)

    def _cache_submap(self, submap_idx: int, frame_points) -> None:
        """Store a submap's per-frame camera-space points, downsampled to the
        per-submap budget (proportionally across frames) with non-finite
        points dropped (DA3 can emit NaN/inf depth)."""
        total = sum(len(points) for _, points, _ in frame_points)
        keep_fraction = min(1.0, self._max_points / total) if total else 1.0
        cached = []
        for seq_idx, points, colors in frame_points:
            finite = np.isfinite(points).all(axis=1)
            points, colors = points[finite], colors[finite]
            if keep_fraction < 1.0 and len(points):
                n_keep = max(1, int(len(points) * keep_fraction))
                sel = self._rng.choice(len(points), n_keep, replace=False)
                points, colors = points[sel], colors[sel]
            if len(points):
                cached.append((seq_idx, points, colors))
        self._submap_cache[submap_idx] = cached

    def _log_submap(self, submap_idx: int, poses: dict[int, np.ndarray]) -> None:
        """(Re-)project a cached submap with the current poses and log it.

        One entity per submap on the "submap" timeline (NOT static): all
        submaps' clouds are visible together at the latest time cursor, so the
        map accumulates — but because the data is temporal, `rerun
        --memory-limit …` (Makefile `viewer` target) can drop the oldest
        submaps instead of OOM-crashing on a long run.  Re-logging the same
        entity replaces what is shown at the current cursor, which is exactly
        how corrected submaps snap into place after a loop closure.
        """
        cached = self._submap_cache.get(submap_idx) or []
        parts = [(self._transform(points, poses[seq_idx]), colors)
                 for seq_idx, points, colors in cached if seq_idx in poses]
        if parts:
            self._rr.log(
                f"world/points/submap_{submap_idx}",
                self._rr.Points3D(
                    np.concatenate([p for p, _ in parts]),
                    colors=np.concatenate([c for _, c in parts]),
                ),
            )
        self._logged_poses[submap_idx] = {
            seq_idx: poses[seq_idx].copy()
            for seq_idx, _, _ in cached if seq_idx in poses
        }

    def _pose_shift(self, submap_idx: int, poses: dict[int, np.ndarray]) -> float:
        """Max shift of this submap's frames since it was last drawn:
        translation delta plus rotation Frobenius delta (≈ metres for
        room-scale scenes)."""
        shift = 0.0
        for seq_idx, logged in self._logged_poses.get(submap_idx, {}).items():
            current = poses.get(seq_idx)
            if current is None:
                continue
            shift = max(
                shift,
                float(np.linalg.norm(current[:3, 3] - logged[:3, 3]))
                + float(np.linalg.norm(current[:3, :3] - logged[:3, :3])),
            )
        return shift

    # ── the on_update callback ──────────────────────────────────────────────

    def __call__(self, update: SLAMUpdate) -> None:
        rr = self._rr
        try:
            self._set_time(update.submap_idx)  # cosmetic — never block logging
        except Exception:
            pass

        poses = update.keyframe_poses
        print(f"[viewer] logging submap {update.submap_idx}: "
              f"{len(update.new_points_world)} points, {len(poses)} poses",
              flush=True)
        if poses:
            order = sorted(poses)
            positions = np.stack([poses[k][:3, 3] for k in order]).astype(np.float32)
            # static: overwrite the full path each time (no per-submap history
            # to accumulate) and keep the latest camera pose.
            rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)
            latest = poses[order[-1]]
            rr.log(
                "world/camera",
                rr.Transform3D(translation=latest[:3, 3].astype(np.float32),
                               mat3x3=latest[:3, :3].astype(np.float32)),
                static=True,
            )

        # Points are placed with [sR | t]: keyframe_poses are rigid, and under
        # Sim(3) dropping s leaves each frame's geometry at raw depth (ghost
        # copies).  Scaling the rotation block also lets _pose_shift see a
        # pure scale correction and re-project for it.
        scales = getattr(update, "keyframe_scales", None) or {}
        point_poses = {}
        for k, pose in poses.items():
            scaled = pose.copy()
            scaled[:3, :3] *= scales.get(k, 1.0)
            point_poses[k] = scaled

        if update.frame_points_cam:
            # Cache once (the final refresh re-emits the last submap — the
            # cache is kept, only the projection is redone), then draw with
            # the current poses.
            if update.submap_idx not in self._submap_cache:
                self._cache_submap(update.submap_idx, update.frame_points_cam)
            self._log_submap(update.submap_idx, point_poses)

        # Re-project earlier submaps whose poses moved since they were drawn
        # (loop closure / later optimisation) — otherwise corrected geometry
        # stays frozen at its pre-correction position.
        relogged = [
            submap_idx for submap_idx in sorted(self._submap_cache)
            if submap_idx != update.submap_idx
            and self._pose_shift(submap_idx, point_poses) > self._repose_threshold
        ]
        for submap_idx in relogged:
            self._log_submap(submap_idx, point_poses)
        if relogged:
            print(f"[viewer] re-projected {len(relogged)} submap(s) after "
                  f"pose correction: {relogged}", flush=True)

        rr.log(
            "status",
            rr.TextLog(
                f"submaps={update.n_submaps}  "
                f"loop_closures={update.n_loop_closures}  "
                f"keyframes={len(poses)}"
            ),
        )
