"""
Benchmark DA3-SLAM on the KITTI odometry benchmark.

Uses the shared SLAM benchmark standard (scripts/benchmark_common.py): ATE
(SE3 + Sim3) and RPE against each sequence's ground truth, with identical
output files to the other systems in this workspace.

KITTI is the outdoor driving domain, and the shipped config is tuned for
INDOOR HANDHELD (see config/default.yaml).  Three of its defaults are
documented as diverging on km-scale footage — `boundary_scale_damping: 0.0`,
`loop_closure.distance_threshold: 0.80` and `submap_size: 32` — so
`--kitti_profile` applies the documented km-scale values (damping 1.0,
threshold 0.45, submap 16) instead.  Run both to see the domain gap rather
than assuming it.

Layout (the official odometry archives, unzipped together):
    data/kitti/dataset/sequences/XX/image_0/NNNNNN.png   left grey camera
    data/kitti/dataset/sequences/XX/times.txt            per-frame timestamps
    data/kitti/dataset/sequences/XX/calib.txt            P0..P3, Tr
    data/kitti/dataset/poses/XX.txt                      GT, sequences 00-10

Ground truth is one 3x4 row-major CAMERA-TO-WORLD matrix per frame in the left
camera's frame — the same convention TUM and 7-Scenes use, so no frame
conversion is needed.  Timestamps are real seconds from times.txt, so no
synthetic rate is involved.

DA3-SLAM is monocular, so the headline is the Sim3 ATE.  KITTI's own
convention (translation error as a percentage of path length) is reported
alongside it, since that is how KITTI results are usually quoted.

Usage:
    python scripts/benchmark_kitti.py --seq_dir data/kitti/dataset/sequences/04
    python scripts/benchmark_kitti.py --seq_dir data/kitti/dataset/sequences/* --kitti_profile
Aggregate: <out_dir>/kitti_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import benchmark_common as bc
from da3_runner import add_da3_cli, build_config, run_benchmark, run_da3

SYSTEM = "DA3-SLAM"
DATASET = "kitti"
HEADLINE = "sim3"


def load_kitti_sequence(seq_dir: Path, max_frames: int | None):
    """(image paths, timestamps, ground truth) for one odometry sequence."""
    image_dir = seq_dir / "image_0"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_0/ not found in {seq_dir}")
    image_paths = [str(p) for p in sorted(image_dir.glob("*.png"))]
    if not image_paths:
        raise FileNotFoundError(f"No frames in {image_dir}")

    times_file = seq_dir / "times.txt"
    if not times_file.exists():
        raise FileNotFoundError(f"times.txt not found in {seq_dir}")
    timestamps = [float(t) for t in times_file.read_text().split()]

    # Ground truth exists for sequences 00-10 only; 11-21 are the held-out
    # test split, which is still worth running (it just cannot be scored).
    pose_file = seq_dir.parent.parent / "poses" / f"{seq_dir.name}.txt"
    gt: list[tuple[float, np.ndarray]] = []
    if pose_file.exists():
        for i, line in enumerate(pose_file.read_text().splitlines()):
            values = [float(v) for v in line.split()]
            if len(values) != 12 or i >= len(timestamps):
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = np.array(values).reshape(3, 4)
            gt.append((timestamps[i], pose))

    if max_frames:
        image_paths = image_paths[:max_frames]
        timestamps = timestamps[:max_frames]
        gt = gt[:max_frames]
    return image_paths, timestamps, gt


def path_length(poses) -> float:
    positions = np.array([p[:3, 3] for _, p in poses])
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def benchmark_sequence(seq_dir: Path, out_dir: Path, args, model) -> dict | None:
    seq_name = seq_dir.name
    try:
        image_paths, timestamps, gt_all = load_kitti_sequence(seq_dir, args.max_frames)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {seq_name}: {exc}")
        return None
    if len(gt_all) < 3:
        print(f"  [SKIP] {seq_name}: no ground truth (sequences 11-21 are the "
              f"held-out test split)")
        return None

    distance = path_length(gt_all)
    rate = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    print(f"\n  {len(image_paths)} frames  "
          f"({timestamps[0]:.2f}s – {timestamps[-1]:.2f}s @ {rate:.1f} Hz)  "
          f"{len(gt_all)} GT poses  {distance:.0f} m driven")

    est_ts_to_pose, timings, counts = run_da3(image_paths, timestamps, model, args)

    metrics = bc.evaluate_trajectory(
        est_ts_to_pose, gt_all, out_dir, seq_name,
        system=SYSTEM, dataset=DATASET, n_frames=timings["n_frames"],
        timings=timings, max_diff=0.5 / rate, headline=HEADLINE,
        config=_resolved_config(args), **counts)

    # KITTI is normally quoted as a percentage of distance travelled; an ATE in
    # metres is not comparable between a 400 m sequence and a 4 km one.
    if metrics:
        for key in ("ate_sim3", "ate_se3"):
            if key in metrics and metrics[key].get("rmse") is not None:
                metrics[f"{key}_pct"] = 100 * metrics[key]["rmse"] / max(distance, 1e-9)
        metrics["gt_path_length_m"] = distance
        # evaluate_trajectory has already written results.json, so the added
        # fields have to be merged back in or they exist only in the console.
        # run_benchmark already hands this function the PER-SEQUENCE directory,
        # so the file sits directly in it; the nested form is kept as a fallback
        # in case a caller passes the root.
        results = out_dir / "results.json"
        if not results.exists():
            results = out_dir / seq_name / "results.json"
        if results.exists():
            import json
            stored = json.loads(results.read_text())
            stored.update({k: metrics[k] for k in
                           ("ate_sim3_pct", "ate_se3_pct", "gt_path_length_m")
                           if k in metrics})
            results.write_text(json.dumps(stored, indent=2))
        print(f"  path {distance:.0f} m   "
              f"Sim3 ATE {metrics.get('ate_sim3_pct', float('nan')):.2f}% of path")
    return metrics


def _resolved_config(args):
    """The SLAMConfig actually used, as a dict, for results.json provenance."""
    try:
        from dataclasses import asdict, is_dataclass
        cfg = build_config(args)
        return asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    except Exception as exc:            # never fail a run over provenance
        return {"error": f"{type(exc).__name__}: {exc}"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DA3-SLAM on KITTI odometry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_dir", nargs="+", required=True,
                        help="Sequence dir(s), e.g. data/kitti/dataset/sequences/04")
    parser.add_argument("--out_dir", default="outputs/benchmark_kitti",
                        help="Root output directory")
    parser.add_argument("--kitti_profile", action="store_true",
                        help="Apply the documented km-scale settings instead of "
                             "the indoor defaults: boundary_scale_damping 1.0, "
                             "loop distance 0.45, submap_size 16 (see the header "
                             "of config/default.yaml)")
    add_da3_cli(parser)
    args = parser.parse_args()
    if args.kitti_profile:
        # Only fill in what the user did not set explicitly.
        if args.boundary_scale_damping is None:
            args.boundary_scale_damping = 1.0
        if args.loop_distance_threshold is None:
            args.loop_distance_threshold = 0.45
        if args.submap_size is None:
            args.submap_size = 16
    return args


def main() -> None:
    args = parse_args()
    run_benchmark(
        args, args.seq_dir, benchmark_sequence, DATASET,
        title=f"{SYSTEM} — KITTI odometry  (ATE RMSE, Sim3 = monocular metric)",
        headline=HEADLINE, item_label="Sequence")


if __name__ == "__main__":
    main()
