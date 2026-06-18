#!/usr/bin/env python3
"""
Batch EVO evaluation for DA3-SLAM benchmark datasets (TUM, Replica).

This script automates trajectory comparison against ground truth for
entire benchmark suites, generating summary tables, plots, and LaTeX
report snippets.

Usage:
    # Evaluate a single DA3-SLAM output directory (TUM format)
    python evo_batch.py --gt data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
                        --est outputs/slam/trajectory_tum.txt \
                        --dataset tum

    # Evaluate all runs in a directory against a TUM dataset directory
    python evo_batch.py --gt_dir data/tum/ \
                        --est_dir logs/ \
                        --out_dir evaluation/ \
                        --dataset tum

    # Evaluate Replica results (after converting to TUM)
    python evo_batch.py --gt_dir data/replica/ \
                        --est_dir logs/ \
                        --out_dir evaluation/ \
                        --dataset replica

Dependencies:
    pip install --user evo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures -----------------------------------------------------------

@dataclass
class RunMetrics:
    """Metrics for a single run."""
    run_name: str
    dataset_name: str
    n_poses: int = 0
    ate_rmse: float | None = None
    ate_mean: float | None = None
    ate_median: float | None = None
    ate_std: float | None = None
    rpe_trans_rmse: float | None = None
    rpe_trans_mean: float | None = None
    rpe_rot_rmse: float | None = None
    rpe_rot_mean: float | None = None


@dataclass
class DatasetSummary:
    """Summary across all runs for a single dataset."""
    dataset_name: str
    runs: list[RunMetrics] = field(default_factory=list)

    @property
    def best_ate(self) -> RunMetrics | None:
        valid = [r for r in self.runs if r.ate_rmse is not None]
        return min(valid, key=lambda r: r.ate_rmse, default=None)


# ---------------------------------------------------------------------------
# Core EVOTools (reused from evo_compare.py) --------------------------------

class EVOTools:
    """Wraps EVO functionality."""

    def __init__(self):
        try:
            from evo.core import trajectory, metrics, sync, units
            from evo.tools import file_interface, plot
            self.mod_trajectory = trajectory
            self.mod_metrics = metrics
            self.mod_sync = sync
            self.fi = file_interface
            self.plot_tools = plot
            self.unit = units.Unit
        except ImportError:
            print("ERROR: EVO is not installed. Install with: pip install --user evo")
            sys.exit(1)

    def load_traj(self, path: Path, format: str = "tum") -> Any:
        if format == "tum":
            return self.fi.read_tum_trajectory_file(str(path))
        elif format == "kitti":
            return self.fi.read_kitti_poses_file(str(path))
        else:
            raise ValueError(f"Unsupported format: {format}")

    def align_and_compute(self, est_path: Path, gt_path: Path,
                           format: str = "tum",
                           correct_scale: bool = True,
                           normalize_seconds: bool = False) -> dict[str, Any]:
        """Load, align, and compute all metrics.

        normalize_seconds rescales nanosecond timestamps to seconds (EuRoC),
        so the 0.01 s association tolerance below applies regardless of source.
        """
        est = self.load_traj(est_path, format)
        ref = self.load_traj(gt_path, format)

        if normalize_seconds:
            from evo_euroc import normalize_to_seconds
            est = normalize_to_seconds(est)
            ref = normalize_to_seconds(ref)

        # Sync/Associate timestamps
        ref, est = self.mod_sync.associate_trajectories(ref, est, max_diff=0.01)

        # Sim3 alignment
        est.align(ref, correct_scale=correct_scale)

        # ATE
        ape = self.mod_metrics.APE(self.mod_metrics.PoseRelation.translation_part)
        ape.process_data((ref, est))
        ate_stats = ape.get_all_statistics()

        # RPE Translation — try progressively smaller deltas if 1.0m fails
        rpe_t_stats = self._compute_rpe_safe(
            est, ref, self.mod_metrics.PoseRelation.translation_part, "m"
        )

        # RPE Rotation
        rpe_r_stats = self._compute_rpe_safe(
            est, ref, self.mod_metrics.PoseRelation.rotation_angle_deg, "m"
        )

        return {
            "n_poses": len(est.positions_xyz),
            "ate": ate_stats,
            "rpe_trans": rpe_t_stats,
            "rpe_rot": rpe_r_stats,
            "est_traj": est,
            "ref_traj": ref,
        }

    def _compute_rpe_safe(self, est, ref, pose_relation, delta_unit: str) -> dict:
        """Compute RPE, falling back to smaller deltas if the default 1.0m fails."""
        from evo.core import filters

        deltas_m = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
        for delta in deltas_m:
            try:
                rpe = self.mod_metrics.RPE(pose_relation, delta, self.unit(delta_unit))
                rpe.process_data((ref, est))
                return rpe.get_all_statistics()
            except filters.FilterException:
                continue
        # Last resort: frame-based delta
        try:
            rpe = self.mod_metrics.RPE(pose_relation, 1, self.unit("f"))
            rpe.process_data((ref, est))
            return rpe.get_all_statistics()
        except filters.FilterException:
            return {}

    def save_plot_xy(self, est, ref, out_path: Path, title: str = "") -> None:
        """Save a 2D top-down XY plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        ref_xyz = np.array(ref.positions_xyz).T
        est_xyz = np.array(est.positions_xyz).T

        ax.plot(ref_xyz[0], ref_xyz[2], label="GT", color="green", linewidth=1.5)
        ax.plot(est_xyz[0], est_xyz[2], label="Est", color="royalblue", linewidth=1.5)
        ax.scatter(ref_xyz[0, 0], ref_xyz[2, 0], color="lime", s=80, zorder=5)
        ax.scatter(ref_xyz[0, -1], ref_xyz[2, -1], color="red", s=80, zorder=5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title or "Top-Down View")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------

def find_est_files(est_dir: Path, dataset: str) -> list[Path]:
    """Find all estimated trajectory files in a directory tree."""
    if dataset in ("tum", "replica", "euroc"):
        # Look for trajectory_tum.txt (produced by run_slam.py)
        return sorted(
            p
            for pattern in ("trajectory_tum.txt", "trajectory_est.txt")
            for p in est_dir.rglob(pattern)
        )
    elif dataset == "kitti":
        return sorted(est_dir.rglob("*trajectory_kitti.txt"))
    else:
        return sorted(est_dir.rglob("*.txt"))


def find_gt_file(gt_dir: Path, est_path: Path, dataset: str) -> Path | None:
    """Infer the ground-truth file for an estimated trajectory.

    For EuRoC the body-frame ground truth is converted to a camera-frame TUM
    file on demand (see scripts/evo_euroc.py) and that path is returned.
    """
    if dataset == "euroc":
        from evo_euroc import find_euroc_seq_dir, euroc_gt_tum, sequence_name_from_run
        seq_name = sequence_name_from_run(est_path.parent.name)
        seq_dir = find_euroc_seq_dir(gt_dir, seq_name)
        if seq_dir is None:
            return None
        return euroc_gt_tum(seq_dir)
    if dataset == "tum":
        # Heuristic: the parent directory name should match a TUM dataset
        # e.g., logs/rgbd_dataset_freiburg1_xyz_run1/trajectory_tum.txt
        # → find data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt
        parts = est_path.parts
        for part in parts:
            # Strip common run suffixes like _run1, _w20, etc.
            clean = part
            for suffix in ["_run1", "_run2", "_run3", "_w20", "_w10", "_nested-giant", "_raydpose"]:
                clean = clean.split(suffix)[0]
            gt_path = gt_dir / clean / "groundtruth.txt"
            if gt_path.exists():
                return gt_path
            # Also try just part name
            gt_path2 = gt_dir / part / "groundtruth.txt"
            if gt_path2.exists():
                return gt_path2
    elif dataset == "replica":
        # Heuristic: find known Replica scene names (office0-4, room0-2) in path parts
        # Replica GT is stored as <gt_dir>/<scene>/gt_tum.txt (TUM format)
        # or <gt_dir>/<scene>/traj.txt (raw 4x4, fallback)
        KNOWN_SCENES = [
            "office0", "office1", "office2", "office3", "office4",
            "room0", "room1", "room2",
        ]
        for part in est_path.parts:
            clean = part.lower()
            # Try removing known experiment prefix/suffix patterns
            for prefix in ["replica_", "run_", "test_", "batch_", "exp_"]:
                clean = clean.split(prefix, 1)[-1]
            for suffix in ["_run1", "_run2", "_run3", "_w20", "_w10",
                            "_nested-giant", "_large", "_small", "_base",
                            "_raydpose", "_leangate", "_t066"]:
                clean = clean.split(suffix)[0]
            # Check exact match first
            scene_name = clean.strip("_")
            if scene_name in KNOWN_SCENES:
                # Prefer gt_tum.txt (already TUM format) over traj.txt
                gt_tum = gt_dir / scene_name / "gt_tum.txt"
                if gt_tum.exists():
                    return gt_tum
                gt_raw = gt_dir / scene_name / "traj.txt"
                if gt_raw.exists():
                    return gt_raw
            # Fallback: try the raw part name too
            gt_tum = gt_dir / part / "gt_tum.txt"
            if gt_tum.exists():
                return gt_tum
            gt_raw = gt_dir / part / "traj.txt"
            if gt_raw.exists():
                return gt_raw
    return None


# ---------------------------------------------------------------------------
# Report generation ---------------------------------------------------------

def print_summary_table(results: list[RunMetrics]) -> None:
    """Print a formatted summary table to stdout."""
    print(f"\n{'=' * 90}")
    print(f"{'Dataset':<25} {'Poses':>8} {'ATE RMSE':>10} {'Mean':>10} {'RPE Trans':>10} {'RPE Rot':>10}")
    print(f"{'':25} {'':8} {'(m)':>10} {'(m)':>10} {'(m)':>10} {'(deg)':>10}")
    print(f"{'-' * 90}")
    for r in results:
        ate_r = f"{r.ate_rmse:.4f}" if r.ate_rmse is not None else "N/A"
        ate_m = f"{r.ate_mean:.4f}" if r.ate_mean is not None else "N/A"
        rpe_t = f"{r.rpe_trans_rmse:.4f}" if r.rpe_trans_rmse is not None else "N/A"
        rpe_r = f"{r.rpe_rot_rmse:.4f}" if r.rpe_rot_rmse is not None else "N/A"
        print(f"{r.dataset_name:<25} {r.n_poses:>8} {ate_r:>10} {ate_m:>10} {rpe_t:>10} {rpe_r:>10}")
    print(f"{'=' * 90}\n")


def save_csv(path: Path, results: list[RunMetrics]) -> None:
    """Save results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "run", "n_poses", "ate_rmse", "ate_mean", "ate_median", "ate_std",
            "rpe_trans_rmse", "rpe_trans_mean", "rpe_rot_rmse", "rpe_rot_mean"
        ])
        for r in results:
            writer.writerow([
                r.dataset_name, r.run_name, r.n_poses,
                r.ate_rmse, r.ate_mean, r.ate_median, r.ate_std,
                r.rpe_trans_rmse, r.rpe_trans_mean,
                r.rpe_rot_rmse, r.rpe_rot_mean,
            ])
    print(f"Saved CSV: {path}")


def save_latex_table(path: Path, results: list[RunMetrics]) -> None:
    """Generate a LaTeX table snippet."""
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Dataset & Poses & ATE RMSE (m) & RPE Trans (m) & RPE Rot (deg) \\",
        r"    \midrule",
    ]
    for r in results:
        ate = f"{r.ate_rmse:.4f}" if r.ate_rmse is not None else "N/A"
        rpe_t = f"{r.rpe_trans_rmse:.4f}" if r.rpe_trans_rmse is not None else "N/A"
        rpe_r = f"{r.rpe_rot_rmse:.4f}" if r.rpe_rot_rmse is not None else "N/A"
        lines.append(f"    {r.dataset_name} & {r.n_poses} & {ate} & {rpe_t} & {rpe_r} \\\\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Trajectory evaluation metrics.}",
        r"\end{table}",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX: {path}")


# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch EVO evaluation for DA3-SLAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="Directory containing ground-truth files")
    parser.add_argument("--est_dir", type=Path, required=True,
                        help="Directory containing estimated trajectory outputs")
    parser.add_argument("--out_dir", type=Path, default=Path("evo_batch_results"),
                        help="Output directory for plots and summary")
    parser.add_argument("--dataset", choices=["auto", "tum", "replica", "kitti", "euroc"],
                        default="auto", help="Dataset type (auto = detect from GT dir)")
    parser.add_argument("--format", choices=["tum", "kitti"],
                        default="tum", help="Trajectory format")
    parser.add_argument("--correct_scale", action="store_true", default=True,
                        help="Sim3 scale correction (default)")
    parser.add_argument("--no_correct_scale", action="store_true",
                        help="Disable scale correction (use SE3)"),
    parser.add_argument("--latex", action="store_true",
                        help="Generate LaTeX table snippet")
    parser.add_argument("--csv", action="store_true", default=True,
                        help="Save CSV summary (default)")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Save per-run XY trajectory plots (default)")
    parser.add_argument("--gt_mapping", type=str, default=None,
                        help='Explicit GT file mapping as JSON, e.g. '
                             '{"office0": "data/Replica/office0/gt_tum.txt"}')
    args = parser.parse_args()

    if args.no_correct_scale:
        args.correct_scale = False

    # Parse explicit GT mapping if provided
    gt_mapping: dict[str, str] = {}
    if args.gt_mapping:
        try:
            gt_mapping = json.loads(args.gt_mapping)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid --gt_mapping JSON: {e}")
            sys.exit(1)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect dataset from GT directory structure
    if args.dataset == "auto":
        from evo_euroc import is_euroc_sequence
        # Check for EuRoC structure first (a mav0/ dir somewhere under gt_dir)
        if is_euroc_sequence(args.gt_dir):
            args.dataset = "euroc"
            print("  [Auto-detect] Detected EuRoC dataset")
        if args.dataset == "auto":
            # Check for Replica structure
            for candidate in ["office0", "office1", "room0"]:
                if (args.gt_dir / candidate / "gt_tum.txt").exists():
                    args.dataset = "replica"
                    print("  [Auto-detect] Detected Replica dataset")
                    break
        if args.dataset == "auto":
            # Check for TUM structure
            for subdir in args.gt_dir.iterdir():
                if subdir.is_dir() and (subdir / "groundtruth.txt").exists():
                    args.dataset = "tum"
                    print("  [Auto-detect] Detected TUM dataset")
                    break
        if args.dataset == "auto":
            print(f"  [WARN] Could not auto-detect dataset from {args.gt_dir}, assuming TUM")
            args.dataset = "tum"

    tools = EVOTools()

    # Find all estimated trajectories
    est_files = find_est_files(args.est_dir, args.dataset)
    print(f"Found {len(est_files)} estimated trajectory files")

    if not est_files:
        print(f"No trajectory files found in {args.est_dir}")
        sys.exit(1)

    results: list[RunMetrics] = []

    for est_path in est_files:
        # Find matching ground truth (try explicit mapping first)
        gt_path: Path | None = None
        run_name = est_path.parent.name
        if run_name in gt_mapping:
            gt_path = Path(gt_mapping[run_name])
            if not gt_path.exists():
                print(f"  [WARN] Mapped GT not found: {gt_path}")
                gt_path = None
        if gt_path is None:
            gt_path = find_gt_file(args.gt_dir, est_path, args.dataset)
        if gt_path is None:
            print(f"  [WARN] No GT found for {est_path}, skipping")
            continue

        run_name = est_path.parent.name
        print(f"\n  [{run_name}]")
        print(f"    GT:   {gt_path}")
        print(f"    EST:  {est_path}")

        try:
            # EuRoC est + converted GT are both TUM files; ns timestamps from
            # run_slam are normalized to seconds for association.
            traj_format = "tum" if args.dataset == "euroc" else args.format
            metrics = tools.align_and_compute(
                est_path, gt_path, traj_format, args.correct_scale,
                normalize_seconds=(args.dataset == "euroc"),
            )
        except Exception as e:
            print(f"    [ERROR] {e}")
            traceback.print_exc()
            continue

        rm = RunMetrics(
            run_name=run_name,
            dataset_name=est_path.parent.name,
            n_poses=metrics["n_poses"],
            ate_rmse=float(metrics["ate"].get("rmse", None)),
            ate_mean=float(metrics["ate"].get("mean", None)),
            ate_median=float(metrics["ate"].get("median", None)),
            ate_std=float(metrics["ate"].get("std", None)),
            rpe_trans_rmse=float(metrics["rpe_trans"].get("rmse", None)),
            rpe_trans_mean=float(metrics["rpe_trans"].get("mean", None)),
            rpe_rot_rmse=float(metrics["rpe_rot"].get("rmse", None)),
            rpe_rot_mean=float(metrics["rpe_rot"].get("mean", None)),
        )
        results.append(rm)

        # Save per-run plot
        if args.plots:
            plot_path = out_dir / f"{run_name}_xy.png"
            try:
                tools.save_plot_xy(
                    metrics["est_traj"], metrics["ref_traj"], plot_path,
                    title=f"{run_name} (ATE RMSE: {rm.ate_rmse:.4f} m)"
                )
                print(f"    Saved plot: {plot_path}")
            except Exception as e:
                print(f"    [WARN] Plot save failed: {e}")

    # Summary outputs
    print_summary_table(results)

    if args.csv:
        save_csv(out_dir / "results.csv", results)

    if args.latex:
        save_latex_table(out_dir / "table.tex", results)

    # JSON dump for programmatic access
    json_data = {
        "dataset_type": args.dataset,
        "correct_scale": args.correct_scale,
        "runs": [
            {
                "run_name": r.run_name,
                "dataset_name": r.dataset_name,
                "n_poses": r.n_poses,
                "ate_rmse": r.ate_rmse,
                "ate_mean": r.ate_mean,
                "ate_median": r.ate_median,
                "ate_std": r.ate_std,
                "rpe_trans_rmse": r.rpe_trans_rmse,
                "rpe_trans_mean": r.rpe_trans_mean,
                "rpe_rot_rmse": r.rpe_rot_rmse,
                "rpe_rot_mean": r.rpe_rot_mean,
            }
            for r in results
        ]
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {out_dir / 'results.json'}")

    print(f"\n{'=' * 60}")
    print(f"  Evaluated {len(results)}/{len(est_files)} runs")
    print(f"  Results saved to: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
