"""
Re-score a saved trajectory against ground truth at a different --max_diff.

WHY THIS EXISTS
---------------
The UAS comparison was scored at the harness default `max_diff = 0.02` s, and
on this dataset that silently produces near-meaningless associations:

  * UAS ground truth is sampled at 10 Hz (100.0 ms spacing).
  * The camera runs at 20 Hz on a grid offset from GT's by ~46 ms.
  * For DA3-SLAM's keyframes the nearest-GT gap has a MINIMUM of 32.2 ms, so a
    20 ms window matches exactly ZERO poses — deterministically, not by luck.
  * VGGT-SLAM emitted denser poses and so matched a fraction of them (e.g.
    campus_fog 89 of 1226, 7%), which is not a fair basis for comparison
    either: which poses matched is essentially arbitrary.

A 50 ms window matches every estimate lying inside the GT time span (366 of the
386 in-span poses for frozen_lake — the same 366 the earlier successful runs
scored), so both systems are compared on the same footing.

Scoring happens offline from `trajectory_est.txt`, so nothing needs re-running
and the SLAM output is untouched.

Usage:
    python scripts/rescore_trajectory.py --est PATH/trajectory_est.txt \\
        --gt data/UAS/frozen_lake/gt_odometry.tum --max_diff 0.05
    python scripts/rescore_trajectory.py --glob 'outputs/p6_uas/*/*/trajectory_est.txt' \\
        --gt_root data/UAS --max_diff 0.05
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.spatial.transform import Rotation

import benchmark_common as bc

# Sequence name -> ground-truth file, for --glob mode.  hornbill lives one
# level deeper than the others.
UAS_GT = {
    "fyllingsdalen_tunnel": "fyllingsdalen_tunnel/gt_odometry.tum",
    "hornbill": "runehamar_tunnel/hornbill/gt_odometry.tum",
    "campus_fog": "campus_fog/gt_odometry.tum",
    "frozen_lake": "frozen_lake/gt_odometry.tum",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--est", default=None, help="One trajectory_est.txt")
    p.add_argument("--gt", default=None, help="Ground-truth .tum for --est")
    p.add_argument("--glob", default=None,
                   help="Glob of trajectory_est.txt files; sequence name is "
                        "inferred from the path and matched against --gt_root")
    p.add_argument("--gt_root", default="data/UAS")
    p.add_argument("--max_diff", type=float, default=0.05)
    p.add_argument("--label", default=None, help="Label for the printed table")
    p.add_argument("--json_out", default=None)
    return p.parse_args()


def load_tum(path: Path) -> dict[float, np.ndarray]:
    """Read a TUM trajectory into {timestamp: 4x4 pose}."""
    out: dict[float, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 8:
            continue
        ts = float(f[0])
        t = np.array([float(x) for x in f[1:4]])
        q = np.array([float(x) for x in f[4:8]])          # qx qy qz qw
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3, 3] = t
        out[ts] = T
    return out


def score(est_path: Path, gt_path: Path, max_diff: float) -> dict | None:
    est = load_tum(est_path)
    gt_all = bc.load_groundtruth(gt_path)
    if not est or not gt_all:
        return None

    est_ts = sorted(est)
    gt_ts = [e[0] for e in gt_all]
    pairs = bc.associate(est_ts, gt_ts, max_diff=max_diff)
    if len(pairs) < 3:
        return {"n_matched": len(pairs), "insufficient": True}

    # compute_ate takes (gt_poses, est_poses) as 4x4 matrices — GT FIRST.
    # Swapping them inverts the alignment and reports a reciprocal scale.
    est_poses = [est[est_ts[ia]] for ia, _ in pairs]
    gt_poses = [np.asarray(gt_all[ib][1]) for _, ib in pairs]

    sim3 = bc.compute_ate(gt_poses, est_poses, align="sim3")
    se3 = bc.compute_ate(gt_poses, est_poses, align="se3")

    gt_full = np.array([np.asarray(e[1])[:3, 3] for e in gt_all])
    path_len = float(np.linalg.norm(np.diff(gt_full, axis=0), axis=1).sum())
    return {
        "n_est": len(est_ts), "n_matched": len(pairs),
        "ate_sim3": sim3["rmse"], "ate_se3": se3["rmse"],
        "scale": sim3.get("scale"), "gt_path_length_m": path_len,
        "ate_sim3_pct": 100.0 * sim3["rmse"] / path_len if path_len else None,
    }


def main() -> None:
    args = parse_args()
    rows = []

    if args.est:
        r = score(Path(args.est), Path(args.gt), args.max_diff)
        rows.append(("(single)", r))
    elif args.glob:
        for p in sorted(globmod.glob(args.glob)):
            name = next((k for k in UAS_GT if k in p), None)
            if name is None:
                print(f"  [skip] cannot infer sequence from {p}")
                continue
            gt = Path(args.gt_root) / UAS_GT[name]
            rows.append((name, score(Path(p), gt, args.max_diff)))
    else:
        raise SystemExit("need --est or --glob")

    label = f"  [{args.label}]" if args.label else ""
    print(f"\n  re-scored at max_diff = {args.max_diff * 1000:.0f} ms{label}")
    print(f"  {'sequence':<22}{'matched':>9}{'Sim3 (m)':>11}{'SE3 (m)':>10}"
          f"{'scale':>9}{'Sim3 %path':>12}")
    for name, r in rows:
        if r is None:
            print(f"  {name:<22}{'FAILED':>9}")
        elif r.get("insufficient"):
            print(f"  {name:<22}{r['n_matched']:>9}   too few matches")
        else:
            print(f"  {name:<22}{r['n_matched']:>9}{r['ate_sim3']:>11.2f}"
                  f"{r['ate_se3']:>10.2f}{r['scale']:>9.3f}"
                  f"{r['ate_sim3_pct']:>11.1f}%")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {n: r for n, r in rows}, indent=2, default=float))


if __name__ == "__main__":
    main()
