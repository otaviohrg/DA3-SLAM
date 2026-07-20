"""
Periodic + event-triggered trajectory snapshots for SLAM runs.

TrajectorySnapshotter plugs into the two DA3SLAM callback hooks:

  * ``on_update`` (fired per optimised submap): saves a top-down trajectory
    PNG at most every ``interval_s`` seconds of wall-clock time, so a long
    run leaves a time-lapse of the growing map.
  * ``on_loop_closure`` (fired twice per loop-closure drain): saves a
    pre/post snapshot pair with red lines connecting each loop-closure frame
    pair at their graph poses of that instant.  phase="pre" (factors
    inserted, optimisation not yet run): the endpoints still carry the
    accumulated drift the closure is about to correct, so the chord shows
    what the closure claims.  phase="post" (right after that optimisation):
    the same chords again — a converged closure has coincident endpoints
    (chord collapses to a point), so the pair shows exactly what the
    correction did to the map.

Filenames carry a zero-padded global sequence number (``012_t0095s_...``,
``013_loop_pre_opt_...``) so a lexical sort of the snapshot directory replays
the run in exact chronological order, loop-closure events interleaved.

With ground truth supplied (``gt`` + ``timestamps``), every snapshot Sim3-
aligns the current partial estimate onto GT (Umeyama over the timestamp-
matched poses, same convention as the benchmark scorer) and draws both; until
enough matched poses exist (early in a run, or GT starting late) it falls
back to the raw estimate frame and says so in the title.

Rendering uses the matplotlib OO API rather than pyplot: both callbacks run
inside the SLAM processing thread, and pyplot's global figure state is not
thread-safe.  Failures are printed and swallowed (the callbacks already
guarantee they never propagate into the SLAM run, this just keeps the
messages readable).

Usage (wired automatically by da3_runner.run_da3 via --snapshot_interval):

    snap = TrajectorySnapshotter(out_dir / "snapshots", interval_s=10.0,
                                 gt=gt_all, timestamps=timestamps)
    slam.run(paths, on_update=snap.on_update, on_loop_closure=snap.on_loop_closure)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from benchmark_common import associate, sim3_align

_MIN_ALIGN_POSES = 3


class TrajectorySnapshotter:
    """Renders top-down trajectory PNGs from the SLAM callback hooks."""

    def __init__(
        self,
        out_dir: str | Path,
        interval_s: float = 10.0,
        gt: list[tuple[float, np.ndarray]] | None = None,
        timestamps: list[float] | None = None,
        max_diff: float = 0.02,
    ) -> None:
        """
        Args:
            out_dir:     directory for the PNGs (created if missing)
            interval_s:  minimum wall-clock spacing of periodic snapshots
            gt:          ground truth as (timestamp, 4x4 cam-to-world), sorted
                         (benchmark_common.load_groundtruth output); None
                         disables the GT overlay
            timestamps:  timestamps[i] = timestamp (s) of input frame seq_idx i
                         (required for the GT overlay)
            max_diff:    estimate↔GT association tolerance in seconds
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self._gt = gt if gt and timestamps else None
        self._timestamps = timestamps
        self._max_diff = max_diff
        self._t0 = time.monotonic()
        self._last_periodic = float("-inf")
        self._seq = 0  # global snapshot counter (chronological filename order)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def on_update(self, update) -> None:
        """Per-submap hook: save a snapshot every `interval_s` wall seconds."""
        now = time.monotonic()
        if now - self._last_periodic < self.interval_s:
            return
        self._last_periodic = now
        elapsed = now - self._t0
        self._render(
            update.keyframe_poses,
            name=f"t{elapsed:05.0f}s_submap{update.submap_idx:03d}.png",
            title=(f"t=+{elapsed:.0f}s  submap {update.submap_idx}  "
                   f"{len(update.keyframe_poses)} keyframes  "
                   f"{update.n_loop_closures} loop closure(s)"),
        )

    def on_loop_closure(self, keyframe_poses: dict, pairs: list,
                        phase: str = "pre") -> None:
        """Loop hook: save pre/post-optimisation snapshots with loop chords."""
        elapsed = time.monotonic() - self._t0
        if phase == "pre":
            subtitle = "PRE-optimisation (chord = drift about to be corrected)"
        else:
            subtitle = "POST-optimisation (converged chord collapses to a point)"
        self._render(
            keyframe_poses,
            name=f"loop_{phase}_opt_t{elapsed:05.0f}s.png",
            title=f"loop closure at t=+{elapsed:.0f}s — {subtitle}",
            chords=pairs,
        )

    # ── rendering ─────────────────────────────────────────────────────────────

    def _align_to_gt(self, order: list, pos: np.ndarray) -> np.ndarray | None:
        """Sim3-align the current estimate positions onto ground truth.

        Umeyama over the timestamp-matched pose pairs (the benchmark scorer's
        convention).  Returns aligned copies of `pos`, or None when fewer than
        _MIN_ALIGN_POSES estimate poses match a GT stamp (early in the run, or
        GT coverage starting late) — the caller then plots the raw estimate.
        """
        est_stamps = [self._timestamps[i] for i in order]
        gt_stamps = [ts for ts, _ in self._gt]
        pairs = associate(est_stamps, gt_stamps, max_diff=self._max_diff)
        if len(pairs) < _MIN_ALIGN_POSES:
            return None
        src = pos[[ia for ia, _ in pairs]]
        dst = np.array([self._gt[ib][1][:3, 3] for _, ib in pairs])
        T = sim3_align(src, dst)
        pos_h = np.hstack([pos, np.ones((len(pos), 1))])
        return (T @ pos_h.T).T[:, :3]

    def _render(self, poses: dict, name: str, title: str,
                chords: list | None = None) -> None:
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            order = sorted(poses)
            row = {seq: k for k, seq in enumerate(order)}
            pos = np.array([poses[i][:3, 3] for i in order], dtype=np.float64)

            aligned = self._align_to_gt(order, pos) if self._gt else None
            if aligned is not None:
                pos = aligned

            fig = Figure(figsize=(9, 9))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            if aligned is not None:
                gt_pos = np.array([T[:3, 3] for _, T in self._gt])
                ax.plot(gt_pos[:, 0], gt_pos[:, 2], "g-", linewidth=1.5,
                        label="Ground truth")
                est_label = "estimate (Sim3-aligned)"
            else:
                est_label = ("estimate (raw frame — no GT alignment yet)"
                             if self._gt else "estimate (est frame)")
            ax.plot(pos[:, 0], pos[:, 2], "b--", linewidth=1.2, label=est_label)
            ax.scatter(pos[0, 0], pos[0, 2], color="green", s=60, zorder=5)
            ax.scatter(pos[-1, 0], pos[-1, 2], color="blue", s=80,
                       marker="*", zorder=5)
            drew_chord = False
            for seq_b, seq_a in chords or []:
                if seq_b not in row or seq_a not in row:
                    continue
                seg = pos[[row[seq_b], row[seq_a]]]
                ax.plot(seg[:, 0], seg[:, 2], "r-o", linewidth=1.5,
                        markersize=6, markerfacecolor="none", zorder=6,
                        label=None if drew_chord else "loop closure")
                drew_chord = True
            ax.set_xlabel("X (m)" if aligned is not None else "X")
            ax.set_ylabel("Z (m)" if aligned is not None else "Z")
            ax.set_title(title)
            ax.legend()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            self._seq += 1
            path = self.out_dir / f"{self._seq:03d}_{name}"
            fig.savefig(str(path), dpi=120, bbox_inches="tight")
            print(f"[Snapshot] saved {path}")
        except Exception as exc:
            print(f"[Snapshot] failed for {name} ({exc!r})")
