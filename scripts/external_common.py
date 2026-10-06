"""
Shared adapter for benchmarking EXTERNAL SLAM systems against the shared standard.

Copied verbatim into each baseline repo's scripts/ alongside benchmark_common.py
(same lockstep rule: edit it in one place or not at all).  It exists so that
adding a baseline is a ~40-line spec rather than a port: `benchmark_common.py`
already owns dataset IO and scoring, and every system here writes its trajectory
to disk, so the only per-system knowledge is *how to invoke it* and *where the
poses land*.

Contract for a baseline driver:

    from external_common import DATASETS, run_external, main_loop
    def build_cmd(ctx) -> list[str]:   ...      # how to invoke the system
    def load_poses(ctx) -> dict[float, np.ndarray]: ...   # where poses land
    main_loop(SYSTEM, build_cmd, load_poses)

DATASET PROTOCOLS (matched to what the published tables use — see notes below):
  tum      9 x freiburg1_*            MASt3R-SLAM eval_tum.sh, ViSTA-SLAM
  replica  office0-4, room0-2 (8)     ViSTA-SLAM evaluation_replica.py:38
  7scenes  <scene>/seq-01 x 7         MASt3R-SLAM dataloader.py:142,
                                      ViSTA-SLAM evaluation_7scenes.py:51
7-Scenes uses seq-01 ONLY: the dataset ships 2-12 sequences per scene
(redkitchen 12, heads 2), so averaging over all 46 would weight redkitchen 26%
of the mean and heads 4% despite each being one scene.  One-per-scene weights
the seven environments equally, which is what a "7-Scenes average" means.

POSE CONVENTIONS FAIL SILENTLY.  A camera-to-world/world-to-camera flip or a
quaternion-order mistake yields a plausible-looking-but-wrong ATE rather than an
error, so `check_sanity()` is called on every trajectory and warns when the
estimate looks degenerate (no motion, huge scale, non-finite).  Validate each new
baseline against its own published number on one sequence before trusting a table.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import benchmark_common as bc

# ── dataset protocols ─────────────────────────────────────────────────────────

TUM_SEQUENCES = [
    "rgbd_dataset_freiburg1_360", "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2", "rgbd_dataset_freiburg1_floor",
    "rgbd_dataset_freiburg1_plant", "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg1_rpy", "rgbd_dataset_freiburg1_teddy",
    "rgbd_dataset_freiburg1_xyz",
]
REPLICA_SCENES = ["office0", "office1", "office2", "office3", "office4",
                  "room0", "room1", "room2"]
SEVENSCENES_SCENES = ["chess", "fire", "heads", "office", "pumpkin",
                      "redkitchen", "stairs"]

DATASETS = ("tum", "replica", "7scenes")

# ── canonical table ordering ──────────────────────────────────────────────────
# The order rows appear in the paper's tables.  Kept here so every report script
# renders the same sequence instead of whatever order results happened to be
# produced in.  `*` marks a system run WITHOUT intrinsics; MASt3R-SLAM and
# DROID-SLAM appear in both tables because they are the same system given
# different intrinsics sources, which is the convention the recent uncalibrated
# papers use (UniSim-SLAM writes the uncalibrated one `MASt3R-SLAM*`).
CALIBRATED_ORDER = [
    "ORB-SLAM3",
    "DeepV2D",
    "DPV-SLAM",
    "DPV-SLAM++",
    "GO-SLAM",
    "DROID-SLAM",
    "MASt3R-SLAM",
]

UNCALIBRATED_ORDER = [
    "DROID-SLAM*",
    "MASt3R-SLAM*",
    "ViSTA-SLAM",
    "EC3R-SLAM",
    "VGGT-SLAM 2.0",
    "AMB3R",
    "DA3-Streaming",
    "DASH-SLAM",          # this work (repo: DA3-SLAM)
]


def table_order(system: str, calibrated: bool) -> int:
    """Sort key for a system name; unknown systems sort to the end."""
    order = CALIBRATED_ORDER if calibrated else UNCALIBRATED_ORDER
    try:
        return order.index(system)
    except ValueError:
        return len(order)


@dataclass
class SeqContext:
    """Everything a baseline needs for one sequence, plus where to put results."""
    dataset: str
    name: str                       # unique label, e.g. "chess_seq-01"
    seq_dir: Path                   # the system's input directory
    image_paths: list[str]
    timestamps: list[float]
    gt: list[tuple[float, np.ndarray]]
    max_diff: float
    out_dir: Path
    args: object = None
    extra: dict = field(default_factory=dict)


def discover(dataset: str, root: Path) -> list[Path]:
    """Sequence directories for `dataset` under `root`, in protocol order."""
    if dataset == "tum":
        return [root / s for s in TUM_SEQUENCES if (root / s).is_dir()]
    if dataset == "replica":
        return [root / s for s in REPLICA_SCENES if (root / s).is_dir()]
    if dataset == "7scenes":
        # seq-01 only — see the module docstring.
        return [root / s / "seq-01" for s in SEVENSCENES_SCENES
                if (root / s / "seq-01").is_dir()]
    raise ValueError(f"unknown dataset {dataset!r}")


def load_sequence(dataset: str, seq_dir: Path, fps: float = 30.0,
                  max_frames: int | None = None) -> SeqContext | None:
    """Resolve one sequence to images + timestamps + GT via benchmark_common."""
    if dataset == "tum":
        gt_txt = seq_dir / "groundtruth.txt"
        if not gt_txt.exists():
            print(f"  [SKIP] {seq_dir.name}: no groundtruth.txt")
            return None
        paths = sorted(str(p) for p in (seq_dir / "rgb").glob("*.png"))
        if max_frames:
            paths = paths[:max_frames]
        # TUM image filenames ARE their timestamps.
        ts = [float(Path(p).stem) for p in paths]
        return SeqContext(dataset, seq_dir.name, seq_dir, paths, ts,
                          bc.load_groundtruth(gt_txt), 0.02, Path("."))

    if dataset == "replica":
        gt_txt = seq_dir / "gt_tum.txt"
        if not gt_txt.exists():
            print(f"  [SKIP] {seq_dir.name}: no gt_tum.txt")
            return None
        paths, ts = bc.load_replica_images(seq_dir, max_frames, fps)
        return SeqContext(dataset, seq_dir.name, seq_dir, paths, ts,
                          bc.load_groundtruth(gt_txt), 0.5 / fps, Path("."))

    if dataset == "7scenes":
        paths, ts, gt = bc.load_7scenes_sequence(seq_dir, max_frames, fps)
        if len(gt) < 3:
            print(f"  [SKIP] {seq_dir.name}: {len(gt)} usable GT poses")
            return None
        name = f"{seq_dir.parent.name}_{seq_dir.name}"
        return SeqContext(dataset, name, seq_dir, paths, ts, gt,
                          0.5 / fps, Path("."))

    raise ValueError(f"unknown dataset {dataset!r}")


# ── staging ───────────────────────────────────────────────────────────────────

def stage_tum_layout(ctx: SeqContext, dest: Path) -> Path:
    """Present ANY dataset to a baseline as a TUM RGB-D sequence directory.

        <dest>/rgb/<timestamp>.<ext>   symlinks to the real frames
        <dest>/rgb.txt                 "timestamp rgb/<file>" index
        <dest>/groundtruth.txt         TUM-format GT

    Many baselines only speak TUM (Photo-SLAM's tum_mono reads <seq>/rgb.txt;
    DROID-SLAM's TUM path expects the same), and several have no Replica or
    7-Scenes loader at all.  Rather than writing per-dataset loaders inside each
    baseline — or, for the C++ ones, new examples requiring a rebuild — this
    makes every dataset look like the one layout they all support.

    Frames are named by their timestamp, which is TUM's own convention and means
    a system that derives timestamps from filenames (several do) gets the right
    ones without extra wiring.  Symlinks keep it free.
    """
    rgb_dir = dest / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for ts, src in zip(ctx.timestamps, ctx.image_paths):
        ext = Path(src).suffix
        link = rgb_dir / f"{ts:.6f}{ext}"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path(src).resolve())
        lines.append(f"{ts:.6f} rgb/{link.name}")
    (dest / "rgb.txt").write_text(
        "# color images\n# timestamp filename\n" + "\n".join(lines) + "\n")

    from scipy.spatial.transform import Rotation
    gt_lines = []
    for ts, T in ctx.gt:
        T = np.asarray(T, dtype=np.float64)
        q = Rotation.from_matrix(T[:3, :3]).as_quat()          # qx qy qz qw
        t = T[:3, 3]
        gt_lines.append(f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                        f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    (dest / "groundtruth.txt").write_text(
        "# ground truth trajectory\n# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(gt_lines) + "\n")
    return dest


def stage_images(ctx: SeqContext, dest: Path, pattern: str = "{i:06d}{ext}",
                 symlink: bool = True) -> Path:
    """Expose the sequence as a flat directory of images.

    Most baselines take "a folder of images" and sort by filename; the datasets
    here store them differently (TUM rgb/*.png, Replica results/frame*.jpg,
    7-Scenes frame-*.color.png).  Symlinks keep this free — 43k 7-Scenes frames
    would otherwise be copied per run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(ctx.image_paths):
        ext = Path(src).suffix
        link = dest / pattern.format(i=i, ext=ext)
        if link.exists() or link.is_symlink():
            link.unlink()
        if symlink:
            link.symlink_to(Path(src).resolve())
        else:
            shutil.copy2(src, link)
    return dest


# ── invocation ────────────────────────────────────────────────────────────────

def run_external(cmd: list[str], cwd: Path | None, log_path: Path,
                 env: dict | None = None, timeout: float | None = None) -> dict:
    """Run a baseline, tee stdout+stderr to `log_path`, return timings.

    A non-zero exit is reported but NOT raised: a system that dies on one
    sequence should not abort the sweep, and the caller still gets the log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **(env or {})}
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()

    # KILL THE WHOLE PROCESS GROUP ON TIMEOUT, NOT JUST THE CHILD.
    # subprocess.run(timeout=...) kills only the direct child.  Baselines that
    # spawn workers (EC3R-SLAM uses multiprocessing; MASt3R-SLAM and GO-SLAM
    # also fork) leave those grandchildren alive holding GPU memory.  Observed:
    # EC3R's spawn children survived 53 min after their parent was timed out,
    # pinning 5.2 GB, which starved the next system's gpu_wait and previously
    # cascaded into a whole sweep of phantom CUDA-OOM "failures".
    # start_new_session puts the child in its own process group so the entire
    # tree can be signalled.
    with open(log_path, "w") as log:
        proc = subprocess.Popen([str(c) for c in cmd],
                                cwd=str(cwd) if cwd else None,
                                stdout=log, stderr=subprocess.STDOUT,
                                env=full_env, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            rc = proc.poll()
            if rc is None:
                rc = -9
    wall = time.time() - t0
    if timed_out:
        print(f"  [warn] TIMEOUT after {wall:.0f}s — killed the process group; "
              f"see {log_path}")
    elif rc != 0:
        print(f"  [warn] exit code {rc} — see {log_path}")
    return {"wall_s": wall, "returncode": rc, "timed_out": timed_out}


def _kill_tree(proc: "subprocess.Popen") -> None:
    """SIGTERM then SIGKILL the child's whole process group."""
    import signal
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


# ── pose loading ──────────────────────────────────────────────────────────────

def poses_from_matrix_file(path: Path, timestamps: list[float],
                           rows: int = 4) -> dict[float, np.ndarray]:
    """One flattened camera-to-world matrix per line -> {timestamp: 4x4}.

    Handles 16-float (4x4, DA3-Streaming's camera_poses.txt) and 12-float
    (3x4, KITTI-style) rows.  Lines are matched to `timestamps` by order, which
    is the systems' own convention: one pose per input frame, sorted.
    """
    mats = []
    for line in Path(path).read_text().splitlines():
        v = line.split()
        if len(v) not in (12, 16):
            continue
        m = np.array([float(x) for x in v], dtype=np.float64)
        T = np.eye(4)
        T[:3, :4] = m.reshape(rows, 4)[:3] if len(v) == 16 else m.reshape(3, 4)
        mats.append(T)
    if not mats:
        return {}
    n = min(len(mats), len(timestamps))
    if len(mats) != len(timestamps):
        print(f"  [warn] {len(mats)} poses vs {len(timestamps)} frames — "
              f"pairing the first {n} in order")
    return {timestamps[i]: mats[i] for i in range(n)}


def poses_from_tum_file(path: Path) -> dict[float, np.ndarray]:
    """TUM-format trajectory -> {timestamp: 4x4}, via the shared parser."""
    return bc.load_tum_trajectory(path)


def poses_by_index(poses: dict[int, np.ndarray],
                   timestamps: list[float]) -> dict[float, np.ndarray]:
    """Re-key {frame_index: pose} onto real timestamps."""
    return {timestamps[i]: T for i, T in poses.items() if i < len(timestamps)}


def paired(est: np.ndarray, gt: np.ndarray):
    """Score an estimate against GT that the system already emitted 1:1 aligned.

    Several baselines (ViSTA-SLAM, and anything that dumps `trajectory.npy`
    beside `gt_poses.npy`) subsample the input themselves — a stride, plus a cap
    that re-samples linearly when the stride still yields too many frames — and
    then save the estimate and the matching GT in the same view order.  Trying to
    re-derive which input frame produced which view, in order to attach real
    timestamps, would be guesswork that fails silently when the cap kicks in.

    Since the pairing is already exact, synthesise index timestamps for BOTH
    sides: association becomes the identity and the shared scorer is unchanged,
    so these rows stay comparable with every other system's.

    Returns (est_ts_to_pose, gt_all) ready for bc.evaluate_trajectory, and the
    caller should pass max_diff=0.1 (any value < 1 makes the match exact).
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if est.ndim != 3 or est.shape[-2:] != (4, 4):
        raise ValueError(f"estimate must be (N,4,4), got {est.shape}")
    if gt.ndim != 3 or gt.shape[-2:] != (4, 4):
        raise ValueError(f"ground truth must be (N,4,4), got {gt.shape}")
    n = min(len(est), len(gt))
    if len(est) != len(gt):
        print(f"  [warn] {len(est)} estimated vs {len(gt)} GT poses — "
              f"scoring the first {n}")
    return ({float(i): est[i] for i in range(n)},
            [(float(i), gt[i]) for i in range(n)])


# ── sanity ────────────────────────────────────────────────────────────────────

def check_sanity(est: dict[float, np.ndarray], name: str) -> bool:
    """Warn on trajectories that are obviously wrong rather than merely inaccurate.

    Pose-convention errors (c2w vs w2c, quaternion order) do not raise — they
    produce a finite, plausible ATE.  These checks catch the degenerate cases
    that would otherwise be reported as a real result.
    """
    if len(est) < 3:
        print(f"  [warn] {name}: only {len(est)} poses recovered")
        return False
    P = np.array([T[:3, 3] for T in est.values()])
    ok = True
    if not np.all(np.isfinite(P)):
        print(f"  [warn] {name}: non-finite translations")
        ok = False
    extent = float(np.linalg.norm(P.max(0) - P.min(0)))
    if extent < 1e-6:
        print(f"  [warn] {name}: trajectory has no extent — poses may be identity")
        ok = False
    elif extent > 1e4:
        print(f"  [warn] {name}: extent {extent:.1f} m — diverged or wrong units")
        ok = False
    R = np.array([T[:3, :3] for T in est.values()])
    dets = np.linalg.det(R)
    if np.any(~np.isfinite(dets)) or np.any(np.abs(dets - 1.0) > 0.1):
        print(f"  [warn] {name}: rotation determinants off 1.0 "
              f"(min {np.nanmin(dets):.3f}, max {np.nanmax(dets):.3f}) — "
              f"not rigid, check the pose convention")
        ok = False
    return ok


# ── driver ────────────────────────────────────────────────────────────────────

def add_external_cli(parser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data_root", required=True,
                        help="Dataset root: TUM dir, Replica dir, or 7scenes dir")
    parser.add_argument("--out_dir", default=None,
                        help="Default: outputs/benchmark_<dataset>")
    parser.add_argument("--seq", nargs="*", default=None,
                        help="Restrict to these sequence names (default: the "
                             "full protocol set for the dataset)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-sequence wall-clock cap (seconds)")
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep per-sequence point clouds / staged frames. "
                             "Off by default: AMB3R alone writes ~2 GB per "
                             "sequence and a full sweep will fill the disk.")



def _purge_intermediates(ctx: SeqContext) -> None:
    """Delete per-sequence bulk once the trajectory has been scored.

    Some baselines write enormous per-sequence artifacts: AMB3R saves point
    clouds, confidence maps and the input images into its results npz (~2 GB per
    TUM sequence), which over 9 sequences x 3 repeats is ~50 GB and filled this
    machine's disk mid-sweep.  The scored trajectory and results.json are tiny
    and are kept; everything regenerable is removed.
    """
    import shutil as _sh
    for name in ("staged", "frames"):
        d = ctx.out_dir / name
        if d.is_dir():
            _sh.rmtree(d, ignore_errors=True)
    for pat in ("*.npz", "*.ply", "*.npy", "*.pth"):
        for f in ctx.out_dir.rglob(pat):
            try:
                f.unlink()
            except OSError:
                pass


def main_loop(system: str, build_cmd, load_poses, args, headline: str = "sim3",
              cwd: Path | None = None) -> None:
    """Run `system` over the protocol set and score everything identically."""
    root = Path(args.data_root)
    seq_dirs = discover(args.dataset, root)
    if args.seq:
        want = set(args.seq)
        seq_dirs = [d for d in seq_dirs
                    if d.name in want or d.parent.name in want
                    or f"{d.parent.name}_{d.name}" in want]
    if not seq_dirs:
        raise SystemExit(f"no {args.dataset} sequences found under {root}")

    out_root = Path(args.out_dir or f"outputs/benchmark_{args.dataset}")
    all_metrics = []
    for seq_dir in seq_dirs:
        ctx = load_sequence(args.dataset, seq_dir, args.fps, args.max_frames)
        if ctx is None:
            continue
        ctx.out_dir = out_root / ctx.name
        ctx.args = args
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'═'*60}\n  System:   {system}\n  Sequence: {ctx.name}\n"
              f"  Input:    {seq_dir}\n  Output:   {ctx.out_dir}\n{'═'*60}")
        print(f"  {len(ctx.image_paths)} frames, {len(ctx.gt)} GT poses")
        try:
            timings = run_external(build_cmd(ctx), cwd,
                                   ctx.out_dir / "system.log",
                                   timeout=args.timeout)
            est = load_poses(ctx)
            if not est:
                print(f"  [SKIP] {ctx.name}: no poses produced — "
                      f"see {ctx.out_dir/'system.log'}")
                continue
            check_sanity(est, ctx.name)
            m = bc.evaluate_trajectory(
                est, ctx.gt, ctx.out_dir, ctx.name,
                system=system, dataset=args.dataset,
                n_frames=len(ctx.image_paths),
                timings={"total": timings["wall_s"],
                         "n_frames": len(ctx.image_paths)},
                max_diff=ctx.max_diff, headline=headline)
            if m is not None:
                all_metrics.append(m)
            if not getattr(args, "keep_artifacts", False):
                _purge_intermediates(ctx)
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] {ctx.name}: timed out after {args.timeout}s")
        except Exception as exc:
            print(f"  [ERROR] {ctx.name}: {exc}")
            traceback.print_exc()

    if len(all_metrics) > 1:
        bc.print_summary(all_metrics, headline,
                         f"{system} — {args.dataset}  (ATE RMSE, Sim3 = monocular metric)")
    bc.save_summary(all_metrics, out_root, args.dataset)
    print(f"\n  {len(all_metrics)}/{len(seq_dirs)} sequences scored")
