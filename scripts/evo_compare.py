"""
EVO-based trajectory visualization, comparison, and metrics for DA3-SLAM.

This script provides a unified interface for:
1. Visualizing single trajectories (2D/3D plots)
2. Comparing estimated vs ground-truth trajectories
3. Computing ATE (Absolute Trajectory Error) and RPE (Relative Pose Error)
4. Aligning trajectories (Sim3, SE3, with/without scale correction)
5. Batch comparison across multiple runs/datasets

All metrics and plots are generated using the EVO library, which is the
de-facto standard for SLAM trajectory evaluation.

Usage:
    # Compare a single trajectory against ground truth
    python evo_compare.py tum --gt data/tum/groundtruth.txt --est outputs/slam/trajectory_tum.txt

    # Compare KITTI format
    python evo_compare.py kitti --gt groundtruth.txt --est trajectory_kitti.txt

    # Save plots to a directory
    python evo_compare.py tum --gt gt.txt --est est.txt --out_dir plots/

    # Batch comparison (all runs in a directory)
    python evo_compare.py batch --gt_dir data/tum/ --est_dir logs/ --out_dir comparison/

Dependencies:
    pip install evo
    # Or: pip install --user evo
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------

def _find_evo():
    """Try to import EVO and provide a helpful message if it's missing."""
    try:
        import evo
        return evo
    except ImportError:
        print("ERROR: EVO is not installed.")
        print("Install it with:  pip install --user evo")
        print("Or:              python -m pip install --user evo")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data structures -----------------------------------------------------------

@dataclass
class MetricResult:
    """Stores metric name, value, unit, and optional details."""
    name: str
    value: float | None
    unit: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        if self.value is None:
            return f"{self.name}: N/A"
        return f"{self.name}: {self.value:.6f} {self.unit}"


@dataclass
class ComparisonResult:
    """Stores all metrics and paths for a single comparison."""
    label: str
    gt_path: Path
    est_path: Path
    ate_metrics: list[MetricResult] = field(default_factory=list)
    rpe_trans_metrics: list[MetricResult] = field(default_factory=list)
    rpe_rot_metrics: list[MetricResult] = field(default_factory=list)
    alignment_info: dict[str, Any] = field(default_factory=dict)

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"  {self.label}")
        print(f"{'=' * 60}")
        print(f"  Ground Truth: {self.gt_path}")
        print(f"  Estimated:    {self.est_path}")
        if self.alignment_info:
            print(f"  Alignment:    {self.alignment_info.get('type', 'N/A')}")
        print("\n  --- ATE (m) ---")
        for m in self.ate_metrics:
            print(f"    {m}")
        if self.rpe_trans_metrics:
            print("\n  --- RPE Translation (m) ---")
            for m in self.rpe_trans_metrics:
                print(f"    {m}")
        if self.rpe_rot_metrics:
            print("\n  --- RPE Rotation (deg) ---")
            for m in self.rpe_rot_metrics:
                print(f"    {m}")


# ---------------------------------------------------------------------------
# EVO core functions --------------------------------------------------------

class EVOTools:
    """Wraps EVO functionality for trajectory comparison."""

    def __init__(self):
        _find_evo()
        from evo.core import trajectory, metrics, units, sync
        from evo.tools import file_interface, plot, user
        self.mod_trajectory = trajectory
        self.mod_metrics = metrics
        self.mod_sync = sync
        self.fi = file_interface
        self.plot_tools = plot
        self.user = user
        self.unit = units.Unit

    def load_traj(self, path: Path, format: str = "tum") -> Any:
        """Load a trajectory in EVO format."""
        path_str = str(path)
        if format == "tum":
            return self.fi.read_tum_trajectory_file(path_str)
        elif format == "kitti":
            return self.fi.read_kitti_poses_file(path_str)
        elif format == "euroc":
            return self.fi.read_euroc_csv_trajectory(path_str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def align_trajs(self, traj_est, traj_ref, correct_scale: bool = True,
                    align_type: str = "sim3") -> Any:
        """Align estimated trajectory to reference using specified method."""
        # Associate timestamps first (for TUM format)
        traj_ref, traj_est = self.mod_sync.associate_trajectories(
            traj_ref, traj_est, max_diff=0.01
        )

        # Perform alignment
        if align_type == "sim3":
            traj_est.align(traj_ref, correct_scale=correct_scale)
        elif align_type == "se3":
            traj_est.align(traj_ref, correct_scale=False)
        else:
            raise ValueError(f"Unknown alignment type: {align_type}")

        return traj_est, traj_ref

    def compute_ate(self, traj_est, traj_ref) -> list[MetricResult]:
        """Compute ATE (Absolute Trajectory Error)."""
        # Ensure alignment
        traj_est, traj_ref = self.align_trajs(traj_est, traj_ref,
                                               correct_scale=True, align_type="sim3")

        pose_relation = self.mod_metrics.PoseRelation.translation_part
        metric = self.mod_metrics.APE(pose_relation)
        metric.process_data((traj_ref, traj_est))
        stats = metric.get_all_statistics()

        return [
            MetricResult("rmse", float(stats.get("rmse", None)), "m"),
            MetricResult("mean", float(stats.get("mean", None)), "m"),
            MetricResult("median", float(stats.get("median", None)), "m"),
            MetricResult("std", float(stats.get("std", None)), "m"),
            MetricResult("min", float(stats.get("min", None)), "m"),
            MetricResult("max", float(stats.get("max", None)), "m"),
        ]

    def compute_rpe(self, traj_est, traj_ref, delta: float = 1.0,
                    delta_unit: str = "m") -> tuple[list[MetricResult], list[MetricResult]]:
        """Compute RPE (Relative Pose Error) for translation and rotation."""
        traj_est, traj_ref = self.align_trajs(traj_est, traj_ref,
                                               correct_scale=True, align_type="sim3")

        # Translation RPE
        pose_relation_trans = self.mod_metrics.PoseRelation.translation_part
        metric_trans = self.mod_metrics.RPE(
            pose_relation_trans, delta, self.unit(delta_unit)
        )
        metric_trans.process_data((traj_ref, traj_est))
        stats_t = metric_trans.get_all_statistics()

        # Rotation RPE
        pose_relation_rot = self.mod_metrics.PoseRelation.rotation_angle_deg
        metric_rot = self.mod_metrics.RPE(
            pose_relation_rot, delta, self.unit(delta_unit)
        )
        metric_rot.process_data((traj_ref, traj_est))
        stats_r = metric_rot.get_all_statistics()

        trans_metrics = [
            MetricResult("rmse", float(stats_t.get("rmse", None)), "m"),
            MetricResult("mean", float(stats_t.get("mean", None)), "m"),
            MetricResult("median", float(stats_t.get("median", None)), "m"),
            MetricResult("std", float(stats_t.get("std", None)), "m"),
        ]
        rot_metrics = [
            MetricResult("rmse", float(stats_r.get("rmse", None)), "deg"),
            MetricResult("mean", float(stats_r.get("mean", None)), "deg"),
            MetricResult("median", float(stats_r.get("median", None)), "deg"),
            MetricResult("std", float(stats_r.get("std", None)), "deg"),
        ]
        return trans_metrics, rot_metrics

    def save_traj_plot(self, traj_est, traj_ref, out_path: Path,
                       title: str = "Trajectory Comparison") -> None:
        """Save a 2D/3D trajectory plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectories
        est_xyz = np.array(traj_est.positions_xyz).T
        ref_xyz = np.array(traj_ref.positions_xyz).T

        ax.plot(ref_xyz[0], ref_xyz[1], ref_xyz[2],
                label="Ground Truth", color="green", linewidth=1.5)
        ax.plot(est_xyz[0], est_xyz[1], est_xyz[2],
                label="Estimated", color="royalblue", linewidth=1.5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    def save_xy_plot(self, traj_est, traj_ref, out_dir: Path,
                     title: str = "Top-Down View") -> None:
        """Save a 2D top-down (XZ) plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        est_xyz = np.array(traj_est.positions_xyz).T
        ref_xyz = np.array(traj_ref.positions_xyz).T

        ax.plot(ref_xyz[0], ref_xyz[2], label="Ground Truth",
                color="green", linewidth=1.5)
        ax.plot(est_xyz[0], est_xyz[2], label="Estimated",
                color="royalblue", linewidth=1.5)

        # Mark start/end
        ax.scatter(ref_xyz[0, 0], ref_xyz[2, 0], color="lime", s=80, zorder=5, label="Start")
        ax.scatter(ref_xyz[0, -1], ref_xyz[2, -1], color="red", s=80, zorder=5, label="End")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        path = out_dir / "trajectory_xy.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Main commands --------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Compare a single estimated trajectory against ground truth.

    EuRoC: when --gt points at a EuRoC sequence directory (one containing
    mav0/), its body-frame ground truth is converted to a camera-frame TUM
    file automatically, and nanosecond timestamps (run_slam.py on raw EuRoC
    frames) are rescaled to seconds for association.
    """
    from evo_euroc import is_euroc_sequence, euroc_gt_tum, normalize_to_seconds

    tools = EVOTools()

    # Load trajectories (EuRoC GT is converted from the sequence dir first)
    euroc = is_euroc_sequence(args.ground_truth)
    gt_source = euroc_gt_tum(args.ground_truth) if euroc else args.ground_truth
    traj_format = "tum" if euroc else args.format
    est = tools.load_traj(args.est, traj_format)
    ref = tools.load_traj(gt_source, traj_format)
    if euroc:
        est = normalize_to_seconds(est)
        ref = normalize_to_seconds(ref)

    # Align
    est, ref = tools.align_trajs(est, ref,
                                  correct_scale=args.correct_scale,
                                  align_type=args.align)

    result = ComparisonResult(
        label=args.label or f"{args.est.name} vs {args.ground_truth.name}",
        gt_path=args.ground_truth,
        est_path=args.est,
        alignment_info={"type": args.align, "scale_corrected": args.correct_scale},
    )

    # Compute ATE
    if args.ate:
        result.ate_metrics = tools.compute_ate(est, ref)

    # Compute RPE
    if args.rpe:
        result.rpe_trans_metrics, result.rpe_rot_metrics = tools.compute_rpe(
            est, ref, delta=args.rpe_delta, delta_unit=args.rpe_delta_unit
        )

    # Print results
    result.print_summary()

    # Save outputs
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save plots
        if args.plot_3d:
            tools.save_traj_plot(est, ref, out_dir / "trajectory_3d.png",
                                 title=result.label)
        if args.plot_xy:
            tools.save_xy_plot(est, ref, out_dir, title=result.label)

        # Save metrics JSON
        metrics_dict = {
            "label": result.label,
            "ground_truth": str(args.ground_truth),
            "estimated": str(args.est),
            "alignment": result.alignment_info,
            "ate": {m.name: m.value for m in result.ate_metrics},
            "rpe_translation": {m.name: m.value for m in result.rpe_trans_metrics},
            "rpe_rotation": {m.name: m.value for m in result.rpe_rot_metrics},
        }
        json_path = out_dir / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved metrics: {json_path}")


def cmd_batch(args: argparse.Namespace) -> None:
    """Batch comparison across multiple runs."""
    gt_dir = Path(args.gt_dir)
    est_dir = Path(args.est_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path("evo_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    tools = EVOTools()
    results = []

    # Find all estimated trajectory files
    if args.format == "tum":
        est_files = list(est_dir.rglob("*trajectory_tum.txt"))
    else:
        # For KITTI, make assumptions or allow --pattern
        est_files = list(est_dir.rglob("*trajectory_kitti.txt"))

    print(f"Found {len(est_files)} estimated trajectories.")

    for est_path in est_files:
        # Infer ground truth path from estimated path structure
        # This is heuristic; users may need to adjust
        rel = est_path.relative_to(est_dir)
        gt_path = gt_dir / rel
        if not gt_path.exists():
            # Try common alternatives
            gt_candidates = list(gt_dir.rglob(f"*{rel.name}"))
            if not gt_candidates:
                print(f"  [WARN] Could not find GT for {est_path}, skipping")
                continue
            gt_path = gt_candidates[0]

        print(f"\n  Processing: {est_path.name}")
        try:
            est = tools.load_traj(est_path, args.format)
            ref = tools.load_traj(gt_path, args.format)
        except Exception as e:
            print(f"  [ERROR] Loading trajectories: {e}")
            continue

        try:
            result = ComparisonResult(
                label=str(rel.with_suffix("")),
                gt_path=gt_path,
                est_path=est_path,
                ate_metrics=tools.compute_ate(est, ref),
            )
            results.append(result)
            result.print_summary()
        except Exception as e:
            print(f"  [ERROR] Computing metrics: {e}")
            continue

    # Save summary
    summary = {
        "comparisons": [
            {
                "label": r.label,
                "ate": {m.name: m.value for m in r.ate_metrics},
            }
            for r in results
        ]
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Batch comparison complete: {len(results)} successful")
    print(f"Summary written to: {out_dir / 'summary.json'}")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Visualize a single trajectory (no GT required)."""
    tools = EVOTools()
    traj = tools.load_traj(args.traj, args.format)

    print(f"Loaded trajectory: {len(traj.positions)} poses")
    print(f"  Length: {len(traj.positions_xyz[0])} m (approx)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    xyz = np.array(traj.positions_xyz).T
    ax.plot(xyz[0], xyz[1], xyz[2], "-o", markersize=2, linewidth=1.0, color="royalblue")
    ax.scatter(xyz[0, 0], xyz[1, 0], xyz[2, 0], color="green", s=80, label="Start")
    ax.scatter(xyz[0, -1], xyz[1, -1], xyz[2, -1], color="red", s=80, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Trajectory ({len(traj.positions)} poses)")
    ax.legend()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path_3d = out_dir / "trajectory_3d.png"
    else:
        path_3d = Path("trajectory_3d.png")

    fig.savefig(path_3d, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D plot: {path_3d}")

    # XY (top-down) plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xyz[0], xyz[2], "-o", markersize=2, linewidth=1.0, color="royalblue")
    ax.scatter(xyz[0, 0], xyz[2, 0], color="green", s=80, label="Start")
    ax.scatter(xyz[0, -1], xyz[2, -1], color="red", s=80, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Top-down View (XZ plane)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if args.out_dir:
        path_xy = out_dir / "trajectory_xy.png"
    else:
        path_xy = Path("trajectory_xy.png")

    fig.savefig(path_xy, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved XY plot: {path_xy}")


# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------

def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--format", choices=["tum", "kitti"], default="tum",
                        help="Trajectory file format")
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="EVO-based trajectory comparison & visualization for DA3-SLAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single comparison
  python evo_compare.py compare --gt groundtruth.txt --est trajectory_tum.txt

  # With plots and output directory
  python evo_compare.py compare --gt gt.txt --est est.txt --out_dir plots/

  # Batch comparison over a directory
  python evo_compare.py batch --gt_dir data/tum/ --est_dir logs/ --out_dir comparison/

  # Visualize a single trajectory (no GT needed)
  python evo_compare.py viz --traj trajectory_tum.txt --out_dir plots/
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- compare ---
    p_compare = sub.add_parser("compare", help="Compare estimated vs ground-truth")
    p_compare.add_argument("--gt", "--ground_truth", dest="ground_truth",
                           type=Path, required=True, help="Ground-truth trajectory file")
    p_compare.add_argument("--est", type=Path, required=True,
                           help="Estimated trajectory file")
    p_compare.add_argument("--format", choices=["tum", "kitti"], default="tum")
    p_compare.add_argument("--align", choices=["sim3", "se3"], default="sim3",
                           help="Alignment type (Sim3 corrects scale, SE3 does not)")
    p_compare.add_argument("--correct_scale", action="store_true", default=True,
                           help="Correct scale in alignment (default: True)")
    p_compare.add_argument("--no_correct_scale", action="store_true",
                           help="Disable scale correction")
    p_compare.add_argument("--label", type=str, help="Label for this comparison")
    p_compare.add_argument("--out_dir", type=Path,
                           help="Directory to save plots and metrics JSON")
    p_compare.add_argument("--ate", action="store_true", default=True,
                           help="Compute ATE (default)")
    p_compare.add_argument("--rpe", action="store_true", default=True,
                           help="Compute RPE (default)")
    p_compare.add_argument("--rpe_delta", type=float, default=1.0,
                           help="RPE delta parameter (default: 1.0)")
    p_compare.add_argument("--rpe_delta_unit", type=str, default="m",
                           help="RPE delta unit (default: 'm')")
    p_compare.add_argument("--plot_3d", action="store_true", default=True,
                           help="Save 3D trajectory plot")
    p_compare.add_argument("--plot_xy", action="store_true", default=True,
                           help="Save XY (top-down) plot")
    p_compare.set_defaults(func=cmd_compare)

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Batch comparison over multiple runs")
    p_batch.add_argument("--gt_dir", type=Path, required=True,
                         help="Directory containing ground-truth files")
    p_batch.add_argument("--est_dir", type=Path, required=True,
                         help="Directory containing estimated files")
    p_batch.add_argument("--format", choices=["tum", "kitti"], default="tum")
    p_batch.add_argument("--out_dir", type=Path, default=Path("evo_comparison"),
                         help="Output directory (default: evo_comparison/)")
    p_batch.set_defaults(func=cmd_batch)

    # --- visualize ---
    p_viz = sub.add_parser("viz", help="Visualize a single trajectory")
    p_viz.add_argument("--traj", type=Path, required=True,
                       help="Trajectory file to visualize")
    p_viz.add_argument("--format", choices=["tum", "kitti"], default="tum")
    p_viz.add_argument("--out_dir", type=Path,
                       help="Output directory for saved plots")
    p_viz.set_defaults(func=cmd_visualize)

    args = parser.parse_args()

    if args.command == "compare" and args.no_correct_scale:
        args.correct_scale = False

    args.func(args)


if __name__ == "__main__":
    main()
