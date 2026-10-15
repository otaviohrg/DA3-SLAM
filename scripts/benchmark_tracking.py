"""
Score the LIVE pose stream against ground truth.

The tracker (da3_slam/frontend/tracker.py) has so far been judged by coverage
and by agreement with the optimised map.  Neither says whether the live
trajectory is any good: coverage counts poses without judging them (a
single-keyframe configuration once scored 39% coverage while being wrong by
34 cm), and the map is itself an estimate.  This scores the tracker the way
the SLAM benchmarks score the map — ATE against ground truth, through
benchmark_common — so the two numbers sit side by side and a change can be
called an improvement or not.

What it reports per sequence:
  * live ATE (SE(3) and Sim(3)) over the frames the tracker answered for
  * the optimised trajectory's ATE on the SAME frames, as the reference point
    the tracker is trying to approach
  * coverage, per-frame cost, and the measured pose latency

The run is PACED to the camera rate, without which the frontend races to the
end of the sequence before the first submap exists and nothing is ever
tracked (see run_slam.py --realtime).

Usage:
    python scripts/benchmark_tracking.py --seq_dir data/tum/rgbd_dataset_freiburg1_teddy
    python scripts/benchmark_tracking.py --seq_dir data/7scenes/chess/seq-03 --dataset 7scenes
    python scripts/benchmark_tracking.py --seq_dir data/tum/* --submap_size 16
Aggregate: <out_dir>/tracking_summary.json
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import signal
import sys
import time
from pathlib import Path

# `docker kill -s USR1 <container>` dumps every thread's stack to stderr.  The
# pipeline is four threads around bounded queues, so when it wedges the only
# useful question is which thread is waiting on what — and py-spy cannot answer
# it here (the container has no SYS_PTRACE).
faulthandler.register(signal.SIGUSR1, all_threads=True)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark_common as bc


def paced(source, fps: float):
    """Yield frames on the camera's clock, as a live stream would."""
    start = time.time()
    for i, item in enumerate(source):
        delay = start + i / fps - time.time()
        if delay > 0:
            time.sleep(delay)
        yield item


def load_sequence(seq_dir: Path, dataset: str, max_frames: int | None, fps: float):
    """(image paths, timestamps, ground-truth poses) for the supported layouts."""
    if dataset == "7scenes":
        return bc.load_7scenes_sequence(seq_dir, max_frames, fps)
    if dataset == "replica":
        # The scene dir is the unit (it holds results/ and traj.txt); accept a
        # path pointing at results/ too, since that is what run_slam takes.
        scene = seq_dir.parent if seq_dir.name == "results" else seq_dir
        paths, stamps = bc.load_replica_images(scene, max_frames, fps)
        traj = np.loadtxt(scene / "traj.txt").reshape(-1, 4, 4)
        gt = [(stamps[i], traj[i]) for i in range(min(len(stamps), len(traj)))]
        return paths, stamps, gt
    paths, stamps = bc.load_rgb_list(seq_dir, max_frames)
    return paths, stamps, bc.load_groundtruth(seq_dir / "groundtruth.txt")


def evaluate(seq_dir: Path, args, slam=None, run_index: int = 0):
    """Run one repeat; returns (metrics, slam) so the model is loaded once.

    The loop-closure detector accumulates per-sequence descriptors, so it is
    reset between repeats — otherwise run 2 matches against run 1's frames and
    closes loops with stale indices (the same discipline SharedSLAM uses).
    """
    from da3_slam.config import load_slam_config
    from da3_slam.slam import DA3SLAM

    name = (f"{seq_dir.parent.name}_{seq_dir.name}" if args.dataset == "7scenes"
            else (seq_dir.parent.name if seq_dir.name == "results" else seq_dir.name))
    paths, stamps, gt = load_sequence(seq_dir, args.dataset, args.max_frames, args.fps)
    if len(gt) < 3:
        print(f"  [SKIP] {name}: no usable ground truth")
        return None, slam
    print(f"\n── {name}: {len(paths)} frames, {len(gt)} GT poses"
          + (f"  (run {run_index + 1}/{args.repeats})" if args.repeats > 1 else ""))

    config = load_slam_config(args.config)
    config.tracking.enable = True
    config.submap_size = args.submap_size
    if args.warmup_size is not None:
        # The dead time before the first pose is dominated by COLLECTING the
        # first batch (4.2-12.7 s measured, versus 2.4-3.3 s of inference), so
        # a smaller first submap is the lever on it.
        config.submap_warmup_size = args.warmup_size
    if args.blur_ratio is not None:
        config.tracking.blur_ratio = args.blur_ratio
    if args.matcher:
        config.tracking.matcher = args.matcher
    if args.refresh is not None:
        config.tracking.provisional = args.refresh > 0
        config.tracking.refresh_keyframes = args.refresh
    if args.bootstrap is not None:
        config.tracking.bootstrap = args.bootstrap
    if args.provisional is not None:
        config.tracking.provisional = args.provisional
    if args.triangulate is not None:
        config.tracking.triangulate = args.triangulate
    for override in args.tracking or []:
        key, _, value = override.partition("=")
        current = getattr(config.tracking, key)
        setattr(config.tracking, key, type(current)(value)
                if not isinstance(current, bool) else value.lower() == "true")
    if args.xfeat_device:
        config.tracking.xfeat_device = args.xfeat_device

    if slam is None:
        slam = DA3SLAM(config)
    else:
        slam.config = config
        if slam.detector is not None:
            slam.detector.reset()
    live: dict[int, np.ndarray] = {}
    arrival: dict[int, float] = {}
    cost: list[float] = []
    start = time.time()

    meta: dict[int, tuple] = {}

    def on_pose(seq_idx, pose, tracker_stats):
        live[int(seq_idx)] = np.asarray(pose, dtype=np.float64).copy()
        arrival[int(seq_idx)] = time.time() - start
        cost.append(tracker_stats.milliseconds)
        meta[int(seq_idx)] = (tracker_stats.inliers, tracker_stats.reference_seq,
                              tracker_stats.predicted)

    def on_pose_correction(transform, bound):
        """The bootstrap ladder's gauge has been reconciled with the map.

        Poses emitted before the first submap were computed in DA3's units for
        a 2-frame batch.  Scoring them in that gauge is not a small error: one
        Sim(3) alignment cannot fit two gauges, so it degrades the RMSE of
        every frame, not just the early ones.
        """
        matrix = np.asarray(transform, dtype=np.float64)
        linear, shift = matrix[:3, :3], matrix[:3, 3]
        scale = float(np.cbrt(max(abs(np.linalg.det(linear)), 1e-18)))
        rotation = linear / scale if scale > 0 else linear
        # Everything collected so far was solved in the bootstrap gauge — not
        # only frames up to `bound`, which is merely the last rung's keyframe.
        for seq in list(live):
            pose = live[seq]
            fixed = np.eye(4, dtype=np.float64)
            fixed[:3, :3] = rotation @ pose[:3, :3]
            fixed[:3, 3] = linear @ pose[:3, 3] + shift
            live[seq] = fixed
        print(f"  bootstrap gauge reconciled after frame {bound} "
              f"(scale {scale:.3f}, {len(live)} poses rewritten)")

    if args.lockstep:
        # Deterministic: the frontend is held `lockstep` frames ahead of the
        # optimised map instead of being paced against the wall clock, so the
        # frontend/backend relationship — and therefore which frames get
        # tracked — is fixed by structure.  Also runs at pipeline speed.
        result = slam.run_stream(slam._disk_frame_source(paths), on_pose=on_pose,
                                 on_pose_correction=on_pose_correction,
                                 replay_lead=args.lockstep)
    else:
        result = slam.run_stream(paced(slam._disk_frame_source(paths), args.fps),
                                 on_pose=on_pose,
                                 on_pose_correction=on_pose_correction)

    # The optimised trajectory, indexed the same way, is the target the live
    # stream is chasing — scoring both on the SAME frames separates "tracking
    # is bad" from "this sequence is hard".
    keyframes = sorted({f.seq_idx for sm in result.submaps for f in sm.frames})
    optimised = {int(s): result.trajectory[i]
                 for i, s in enumerate(keyframes[:len(result.trajectory)])}

    rows, dropped = [], 0
    for seq in sorted(live):
        if seq >= len(stamps):
            continue
        if not np.isfinite(live[seq]).all():
            dropped += 1        # never let one bad pose take down the scoring
            continue
        rows.append((stamps[seq], live[seq], optimised.get(seq), seq))
    if dropped:
        print(f"  [WARN] dropped {dropped} non-finite tracked poses")
    if len(rows) < 3:
        print(f"  [SKIP] {name}: only {len(rows)} tracked poses")
        return None, slam

    gt_stamps = [t for t, _ in gt]
    pairs = bc.associate([r[0] for r in rows], gt_stamps, max_diff=args.max_diff)
    if len(pairs) < 3:
        print(f"  [SKIP] {name}: {len(pairs)} GT associations")
        return None, slam
    gt_matched = [gt[j][1] for _, j in pairs]
    live_matched = [rows[i][1] for i, _ in pairs]

    metrics = {
        "sequence": name,
        "run": run_index,
        "frames": len(paths),
        "tracked": len(live),
        "coverage": len(live) / max(len(paths), 1),
        "ms_per_frame": float(np.mean(cost)) if cost else 0.0,
        "ms_per_frame_p95": float(np.percentile(cost, 95)) if cost else 0.0,
        "pose_lag_ms": float(np.median([
            (arrival[s] - s / args.fps) * 1e3 for s in live])),
        "live_ate_se3": bc.compute_ate(gt_matched, live_matched, "se3"),
        "live_ate_sim3": bc.compute_ate(gt_matched, live_matched, "sim3"),
    }

    # Solved poses and extrapolated ones are not the same claim, and mixing
    # them flatters coverage while wrecking ATE: a 45-frame motion model can
    # add 1.5 s of dead reckoning to every gap.  Score them apart.
    solved_mask = np.array([not meta[rows[i][3]][2] for i, _ in pairs])
    metrics["coverage_solved"] = (
        sum(1 for s in live if not meta[s][2]) / max(len(paths), 1))
    if solved_mask.sum() >= 3:
        metrics["live_ate_sim3_solved"] = bc.compute_ate(
            [g for g, k in zip(gt_matched, solved_mask) if k],
            [l for l, k in zip(live_matched, solved_mask) if k], "sim3")

    # Same treatment for the optimised poses, on the frames both answered for.
    both = [(i, j) for i, j in pairs if rows[i][2] is not None]
    if len(both) >= 3:
        metrics["map_ate_sim3"] = bc.compute_ate(
            [gt[j][1] for _, j in both], [rows[i][2] for i, _ in both], "sim3")
        metrics["compared_frames"] = len(both)

    # ── diagnostics: WHERE the error is, so the next fix is not a guess ────
    # Per-frame errors are derived here rather than read out of compute_ate:
    # the evo engine does not return them, only the numpy fallback does, so
    # reading the key works until evo is installed and then crashes.
    gt_positions = np.array([p[:3, 3] for p in gt_matched])
    live_positions = np.array([p[:3, 3] for p in live_matched])
    align = bc.sim3_align(live_positions, gt_positions)
    homogeneous = np.hstack([live_positions, np.ones((len(live_positions), 1))])
    errors = np.linalg.norm(gt_positions - (align @ homogeneous.T).T[:, :3], axis=1)
    seqs = np.array([rows[i][3] for i, _ in pairs])
    ages = np.array([(s - meta[s][1]) / args.fps if meta[s][1] >= 0 else np.nan
                     for s in seqs])
    inliers = np.array([meta[s][0] for s in seqs])
    predicted = np.array([meta[s][2] for s in seqs])

    def bucket(values, edges, label):
        print(f"  error by {label}:")
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (values >= lo) & (values < hi) & np.isfinite(values)
            if mask.sum() >= 3:
                print(f"    {lo:6.0f}-{hi:<6.0f} n={int(mask.sum()):4d}  "
                      f"median {bc.cm_str(float(np.median(errors[mask])))}")
    bucket(ages, [0, 1, 2, 4, 8, 1e9], "seconds since the reference keyframe")
    bucket(inliers, [0, 50, 150, 400, 1e9], "PnP inlier count")
    if predicted.any():
        print(f"  motion-model frames: n={int(predicted.sum())}  "
              f"median {bc.cm_str(float(np.median(errors[predicted])))} vs "
              f"solved {bc.cm_str(float(np.median(errors[~predicted])))}")
    metrics["diagnostics"] = {
        # seq indices let two configurations be compared on the frames BOTH
        # answered for.  Comparing RMSE across configurations with different
        # coverage is apples-to-oranges: the sparser one only answered on the
        # frames it found easy, which flatters it.
        "seqs": seqs.tolist()[:4000],
        # positions, so a per-reference scale can be fitted offline: a single
        # Sim(3) alignment absorbs ONE global scale, and the per-reference
        # error spread (23-41 cm on teddy, against 4-8 cm within a reference)
        # says the residual is per-segment, which only these can test
        "gt_xyz": gt_positions.tolist()[:4000],
        "live_xyz": live_positions.tolist()[:4000],
        "age_seconds": ages[np.isfinite(ages)].tolist()[:2000],
        "inliers": inliers.tolist()[:4000],
        "predicted": predicted.tolist()[:4000],
        "errors": errors.tolist()[:4000],
    }

    for key in ("live_ate_se3", "live_ate_sim3", "map_ate_sim3",
                "live_ate_sim3_solved"):
        metrics.get(key, {}).pop("per_frame_errors", None)
        metrics.get(key, {}).pop("align_T", None)

    # Dead time before the tracker says anything at all.  This is the metric
    # the warmup ramp is aimed at: measured at warmup 16 it is 14-22% of a
    # clip, and 60-85% of it is COLLECTING the first batch, not inference.
    first_seq = int(seqs.min()) if len(seqs) else -1
    metrics["first_pose_seq"] = first_seq
    metrics["first_pose_seconds"] = first_seq / args.fps if first_seq >= 0 else None
    if first_seq >= 0:
        print(f"  first pose at frame {first_seq} "
              f"({first_seq / args.fps:.1f}s, "
              f"{100 * first_seq / max(len(paths), 1):.0f}% of the clip)")

    live_sim3 = metrics["live_ate_sim3"]["rmse"]
    if "live_ate_sim3_solved" in metrics:
        print(f"  solved only: {100 * metrics['coverage_solved']:5.1f}% coverage, "
              f"{bc.cm_str(metrics['live_ate_sim3_solved']['rmse'])} (Sim3)")
    print(f"  coverage {100 * metrics['coverage']:5.1f}%   "
          f"live ATE {bc.cm_str(live_sim3)} (Sim3), "
          f"{bc.cm_str(metrics['live_ate_se3']['rmse'])} (SE3)")
    if "map_ate_sim3" in metrics:
        print(f"  optimised map on the same {metrics['compared_frames']} frames: "
              f"{bc.cm_str(metrics['map_ate_sim3']['rmse'])} (Sim3)")
    if result.timings.get("provisional"):
        print(f"  provisional inference: {result.timings['provisional']:.1f}s "
              f"of extra GPU over {len(paths) / args.fps:.1f}s of footage")
    print(f"  {metrics['ms_per_frame']:.1f} ms/frame "
          f"(p95 {metrics['ms_per_frame_p95']:.1f}), "
          f"pose lag {metrics['pose_lag_ms']:.0f} ms")
    return metrics, slam


def main() -> None:
    from da3_slam.config import DEFAULT_YAML

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq_dir", nargs="+", required=True)
    p.add_argument("--dataset", choices=["tum", "7scenes", "replica"], default="tum")
    p.add_argument("--out_dir", default="outputs/tracking")
    p.add_argument("--config", default=str(DEFAULT_YAML))
    p.add_argument("--submap_size", type=int, default=16)
    p.add_argument("--warmup_size", type=int, default=None,
                   help="Keyframes in the first submaps (submap_warmup_size). "
                        "Smaller = the tracker starts sooner, at the cost of "
                        "weaker DA3 geometry where errors propagate furthest")
    p.add_argument("--lockstep", type=int, default=0, metavar="BATCHES",
                   help="Deterministic replay: hold the frontend this many "
                        "BATCHES ahead of the optimised map instead of pacing "
                        "to the wall clock.  1 keeps the tracker's references "
                        "about one submap behind, as in a live run")
    p.add_argument("--repeats", type=int, default=1,
                   help="Repeat each sequence N times.  This benchmark is NOT "
                        "deterministic — pacing lets thread timing decide when "
                        "references land — so a single run cannot support a "
                        "comparison (two identical runs measured 20.9 cm and "
                        "59.3 cm on teddy)")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_diff", type=float, default=0.02,
                   help="GT association window (s)")
    p.add_argument("--refresh", type=int, default=None, metavar="KEYFRAMES",
                   help="Rolling provisional refresh: re-infer every N "
                        "keyframes so the tracker's reference never ages past "
                        "that gap (implies --provisional). 0 disables")
    p.add_argument("--bootstrap", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Seed the tracker with a doubling ladder of early DA3 "
                        "inferences (2, 4, 8 keyframes) before the first "
                        "submap exists. Never enters the pose graph")
    p.add_argument("--provisional", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Half-batch DA3 pass to refresh the tracker's geometry "
                        "sooner (never enters the pose graph)")
    p.add_argument("--tracking", action="append", metavar="KEY=VALUE",
                   help="Override any TrackerConfig field, e.g. "
                        "--tracking min_inliers=12 --tracking "
                        "motion_model_frames=45 (repeatable)")
    p.add_argument("--triangulate", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Require several local-map keyframes to agree on a "
                        "replenished point's depth, and average them")
    p.add_argument("--matcher", choices=["orb", "xfeat"], default=None,
                   help="Relocalisation matcher (default: the config's)")
    p.add_argument("--xfeat_device", default=None,
                   help="Device for XFeat; cpu is ~1 s per call here, cuda ~6 ms")
    p.add_argument("--blur_ratio", type=float, default=None,
                   help="Override the tracker's blur gate (0 = off, the "
                        "measured-better setting)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for pattern in args.seq_dir:
        for seq in sorted(Path().glob(pattern) if any(c in pattern for c in "*?")
                          else [Path(pattern)]):
            slam = None
            for run_index in range(args.repeats):
                metrics, slam = evaluate(seq, args, slam, run_index)
                if metrics:
                    all_metrics.append(metrics)

    if not all_metrics:
        sys.exit("No sequence produced metrics")
    print(f"\n{'sequence':34} {'cover':>6} {'live Sim3':>10} {'map Sim3':>9} {'ms/f':>6}")
    for m in all_metrics:
        print(f"{m['sequence']:34} {100 * m['coverage']:5.1f}% "
              f"{bc.cm_str(m['live_ate_sim3']['rmse']):>10} "
              f"{bc.cm_str(m.get('map_ate_sim3', {}).get('rmse', float('nan'))):>9} "
              f"{m['ms_per_frame']:6.1f}")
    # Repeat spread: the error bar every comparison has to clear.
    by_sequence: dict[str, list[dict]] = {}
    for m in all_metrics:
        by_sequence.setdefault(m["sequence"], []).append(m)
    if args.repeats > 1:
        print(f"\n{'sequence':34} {'coverage':>18} {'live Sim3 ATE':>24}")
        for name, runs in by_sequence.items():
            cov = np.array([r["coverage"] for r in runs]) * 100
            ate = np.array([r["live_ate_sim3"]["rmse"] for r in runs]) * 100
            print(f"{name:34} {cov.mean():6.1f}% +/-{cov.std():4.1f} "
                  f"[{cov.min():.0f}-{cov.max():.0f}] "
                  f"{ate.mean():8.1f}cm +/-{ate.std():5.1f} "
                  f"[{ate.min():.0f}-{ate.max():.0f}]")
            # what the same frames cost across repeats: pose noise without the
            # coverage noise mixed in
            shared = set.intersection(*[set(r["diagnostics"]["seqs"]) for r in runs])
            if len(shared) >= 10:
                per_run = []
                for r in runs:
                    lookup = dict(zip(r["diagnostics"]["seqs"], r["diagnostics"]["errors"]))
                    e = np.array([lookup[s] for s in sorted(shared)])
                    per_run.append(np.sqrt((e ** 2).mean()) * 100)
                print(f"{'':34} on the {len(shared)} frames every run answered: "
                      f"{np.mean(per_run):.1f}cm +/-{np.std(per_run):.1f} "
                      f"[{min(per_run):.0f}-{max(per_run):.0f}]")
    mean_live = float(np.mean([m["live_ate_sim3"]["rmse"] for m in all_metrics]))
    print(f"{'mean':34} {100 * np.mean([m['coverage'] for m in all_metrics]):5.1f}% "
          f"{bc.cm_str(mean_live):>10}")
    with open(out_dir / "tracking_summary.json", "w") as f:
        json.dump({"sequences": all_metrics, "mean_live_ate_sim3": mean_live}, f, indent=2)
    print(f"\nSaved {out_dir / 'tracking_summary.json'}")


if __name__ == "__main__":
    main()
