"""
Plot a SLAM trajectory against ground truth (TUM format).

Associates the estimate to GT by timestamp, applies Sim(3) Umeyama alignment
(monocular scale is arbitrary — see CLAUDE.md), and renders a two-panel chart:
  • top-down view in the two axes with the largest GT spread (GT vs aligned est)
  • per-axis position vs time

Defaults target a run dir written by run_slam.py, which holds
both trajectory_tum.txt and ground_truth_tum.txt.

    # plot a run directory (auto-finds both files)
    python scripts/plot_gt_vs_est.py --run_dir outputs/run1

    # explicit files, no scale correction (SE3 alignment only)
    python scripts/plot_gt_vs_est.py --est traj.txt --gt gt.txt --no-correct_scale

Saves a PNG (default plots/gt_vs_est.png) and prints APE RMSE + the Sim3 scale.
Note: container-written out_dirs are often root-owned; --out defaults to the
host-writable plots/ dir for that reason.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot SLAM trajectory vs ground truth (TUM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_dir", default=None,
                    help="Run directory containing trajectory_tum.txt and "
                         "ground_truth_tum.txt (used to default --est/--gt)")
    ap.add_argument("--est", default=None, help="Estimated trajectory (TUM)")
    ap.add_argument("--gt", default=None, help="Ground-truth trajectory (TUM)")
    ap.add_argument("--out", default="plots/gt_vs_est.png", help="Output image path")
    ap.add_argument("--max_diff", type=float, default=0.05,
                    help="Max timestamp difference for est↔gt association (s)")
    ap.add_argument("--align", action=argparse.BooleanOptionalAction, default=True,
                    help="Umeyama-align the estimate to GT")
    ap.add_argument("--correct_scale", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Also solve for scale (Sim3). Off = SE3 alignment only")
    ap.add_argument("--title", default=None, help="Extra title prefix")
    ap.add_argument("--show", action="store_true",
                    help="Display the figure instead of only saving it")
    return ap.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Fill in --est/--gt from --run_dir or each other's directory."""
    est, gt = args.est, args.gt
    if args.run_dir:
        rd = Path(args.run_dir)
        est = est or str(rd / "trajectory_tum.txt")
        gt = gt or str(rd / "ground_truth_tum.txt")
    if est and not gt:
        gt = str(Path(est).with_name("ground_truth_tum.txt"))
    if not est:
        est = "outputs/ros_run/trajectory_tum.txt"
        gt = gt or "outputs/ros_run/ground_truth_tum.txt"
    for p, name in ((est, "estimate"), (gt, "ground truth")):
        if not Path(p).exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    return est, gt


def main() -> None:
    """Associate estimate ↔ GT by timestamp (evo), align, plot the two-panel
    chart, and print the APE RMSE + Sim3 scale."""
    args = parse_args()
    est_path, gt_path = resolve_paths(args)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")  # headless save
    import matplotlib.pyplot as plt
    from evo.core import metrics, sync
    from evo.tools import file_interface

    ref = file_interface.read_tum_trajectory_file(gt_path)
    est = file_interface.read_tum_trajectory_file(est_path)
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=args.max_diff)
    print(f"associated pairs: {est_s.num_poses} "
          f"(est {est.num_poses} ↔ gt {ref.num_poses})")

    est_al = copy.deepcopy(est_s)
    scale = 1.0
    if args.align:
        _, _, scale = est_al.align(ref_s, correct_scale=args.correct_scale)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_s, est_al))
    rmse = ape.get_statistic(metrics.StatisticsType.rmse)
    mode = "Sim3" if (args.align and args.correct_scale) else \
           "SE3" if args.align else "unaligned"
    print(f"alignment: {mode}   scale: {scale:.4f}   APE RMSE: {rmse:.3f} m")

    P, Q = ref_s.positions_xyz, est_al.positions_xyz
    names = np.array(["x", "y", "z"])
    a, b = sorted(np.argsort(P.max(0) - P.min(0))[-2:])  # widest two GT axes

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.plot(P[:, a], P[:, b], "-", color="k", lw=2, label="ground truth")
    ax.plot(Q[:, a], Q[:, b], "-", color="tab:red", lw=1.5,
            label=f"DA3-SLAM ({mode}-aligned)")
    ax.scatter(P[0, a], P[0, b], c="g", s=60, zorder=5, label="start")
    ax.set_xlabel(names[a])
    ax.set_ylabel(names[b])
    ax.axis("equal")
    ax.grid(alpha=.3)
    ax.legend()
    prefix = f"{args.title}  " if args.title else ""
    ax.set_title(f"{prefix}Top-down ({names[a]}{names[b]})  APE RMSE={rmse:.2f} m")

    axt = axs[1]
    for i, n in enumerate(names):
        axt.plot(ref_s.timestamps, P[:, i], "-", label=f"gt {n}")
        axt.plot(est_al.timestamps, Q[:, i], "--", label=f"est {n}")
    axt.set_xlabel("t [s]")
    axt.set_ylabel("position [m]")
    axt.grid(alpha=.3)
    axt.legend(ncol=2, fontsize=8)
    axt.set_title("Per-axis position vs time")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
