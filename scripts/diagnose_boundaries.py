"""
Measure how well consecutive DA3 submaps actually agree at their boundary.

THE QUESTION
------------
Splitting a sequence into submaps costs accuracy: on TUM, going from one giant
batch (0 boundaries) to two batches (1 boundary) raises ATE by 19-67%, while
adding *further* boundaries costs almost nothing.  A step-then-plateau, not
accumulation.  Two explanations fit that shape:

  (a) BOUNDARY ALIGNMENT IS DEFECTIVE — composing poses through the shared
      anchor frame introduces a systematic error.
  (b) JOINT ESTIMATION IS SIMPLY BETTER — with one batch every pose comes from
      a single DA3 forward with cross-view attention over all frames, so it is
      globally consistent by construction; any split replaces some of that with
      composition, and you pay once.

ATE cannot separate these.  This script measures the boundary directly.

HOW
---
With `submap_overlap >= 2` the first shared frame *pair* is measured by BOTH
DA3 batches: it appears at the end of submap N and the start of submap N+1.
Two independent measurements of the same relative pose.  Their disagreement is
a direct read on boundary quality, with no pose graph or ground truth involved:

    rot_deg     angle between the two relative rotations
    norm_ratio  ratio of the two translation norms (1.0 = perfect agreement)

`_check_boundary_consistency` in slam.py computes both and now prints every
boundary (not only the ones it flags as broken), which this script parses.

READING THE RESULT
------------------
  Large disagreement (rot >> 1 deg, ratio far from 1.0) -> explanation (a):
      the boundary measurement itself is unreliable, and the anchor-frame
      composition is the thing to fix.
  Small disagreement (rot ~ tenths of a degree, ratio ~ 1.0) -> explanation
      (b): the two batches agree about the geometry, so the split is not
      corrupting anything and the ATE step is the cost of losing joint
      estimation.  Fixing "alignment" would then be chasing the wrong bug.

Usage:
    python scripts/diagnose_boundaries.py \\
        --seq_dir data/tum/rgbd_dataset_freiburg1_desk \\
                  data/tum/rgbd_dataset_freiburg1_xyz \\
                  data/tum/rgbd_dataset_freiburg1_room \\
        --submap_sizes 8 16 32 --out_dir outputs/boundary_diag
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BOUNDARY_RE = re.compile(
    r"\[boundary\] (\d+)->(\d+) rot_deg=([-\d.eE+]+) "
    r"norm_ratio=([-\d.eE+inf]+) broken=(\d)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure DA3 submap boundary agreement (needs overlap>=2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seq_dir", nargs="+", required=True,
                   help="Sequence directories (TUM: contains rgb/) or image "
                        "directories directly (UAS: the extracted frame cache)")
    p.add_argument("--submap_sizes", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--out_dir", default="outputs/boundary_diag")
    p.add_argument("--depth_model", default="nested-giant")
    p.add_argument("--backbone_dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--keyframes_dir", default=None,
                   help="Reuse frozen keyframe lists (e.g. "
                        "outputs/kfgrid_segment/keyframes); the density-1 list "
                        "is used when present")
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


def run_one(seq_dir: Path, submap_size: int, args, out_dir: Path) -> list[dict]:
    """Run one sequence at submap_overlap=2 and parse its boundary lines."""
    # TUM keeps frames in <seq>/rgb; UAS sequences are ROS bags whose frames
    # are pre-extracted into a cache directory, so accept either a sequence
    # directory or an image directory directly.
    images = seq_dir / "rgb" if (seq_dir / "rgb").is_dir() else seq_dir
    cmd = [
        "python", "scripts/run_slam.py",
        "--image_dir", str(images),
        "--out_dir", str(out_dir / f"{seq_dir.name}_s{submap_size}"),
        "--submap_size", str(submap_size),
        "--submap_overlap", "2",          # <- what enables the measurement
        "--depth_model", args.depth_model,
        "--backbone_dtype", args.backbone_dtype,
        "--no_loop_closure",              # isolate odometry from closure repair
        "--skip_ply",
    ]
    if args.max_frames:
        cmd += ["--max_frames", str(args.max_frames)]
    kf = _keyframe_list(args, seq_dir)
    if kf:
        cmd += ["--keyframes_from", str(kf)]

    print(f"\n  {seq_dir.name}  submap={submap_size}  overlap=2")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    [ERROR] exit {proc.returncode}")
        print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-6:]))
        return []

    out = []
    for line in proc.stdout.splitlines():
        m = BOUNDARY_RE.search(line)
        if m:
            out.append({
                "sequence": seq_dir.name, "submap_size": submap_size,
                "from": int(m.group(1)), "to": int(m.group(2)),
                "rot_deg": float(m.group(3)),
                "norm_ratio": float(m.group(4)),
                "broken": bool(int(m.group(5))),
            })
    print(f"    {len(out)} boundaries measured")
    return out


def _keyframe_list(args, seq_dir: Path) -> Path | None:
    """The density-1 frozen list for this sequence, when one exists."""
    if not args.keyframes_dir:
        return None
    directory = Path(args.keyframes_dir)
    for candidate in (f"{seq_dir.name}__d1.txt", f"{seq_dir.name}.txt"):
        path = directory / candidate
        if path.exists():
            return path
    return None


def summarise(records: list[dict]) -> None:
    if not records:
        print("\n  no boundaries measured — did submap_overlap=2 take effect?")
        return

    def stats(values):
        values = sorted(values)
        return {
            "n": len(values), "median": statistics.median(values),
            "p90": values[int(0.9 * (len(values) - 1))],
            "max": values[-1],
        }

    print(f"\n{'=' * 74}\n  BOUNDARY AGREEMENT — the two batches' independent "
          f"measurement\n  of the same shared frame pair\n{'=' * 74}")
    print(f"\n  {'submap':>7}{'n bnd':>7}{'rot med':>10}{'rot p90':>10}"
          f"{'rot max':>10}{'|1-ratio| med':>15}{'p90':>9}{'broken':>8}")
    for submap in sorted({r["submap_size"] for r in records}):
        sel = [r for r in records if r["submap_size"] == submap]
        rot = stats([abs(r["rot_deg"]) for r in sel])
        dev = stats([abs(1.0 - r["norm_ratio"]) for r in sel
                     if r["norm_ratio"] < 1e6])
        broken = sum(1 for r in sel if r["broken"])
        print(f"  {submap:>7}{len(sel):>7}{rot['median']:>10.3f}"
              f"{rot['p90']:>10.3f}{rot['max']:>10.3f}"
              f"{dev['median']:>15.4f}{dev['p90']:>9.4f}"
              f"{broken:>8}")

    all_rot = [abs(r["rot_deg"]) for r in records]
    all_dev = [abs(1.0 - r["norm_ratio"]) for r in records
               if r["norm_ratio"] < 1e6]
    med_rot = statistics.median(all_rot)
    med_dev = statistics.median(all_dev)
    n_broken = sum(1 for r in records if r["broken"])

    print(f"\n  Overall: {len(records)} boundaries, median rotation "
          f"disagreement {med_rot:.3f}°, median translation-norm deviation "
          f"{100 * med_dev:.1f}%, {n_broken} flagged broken "
          f"({100 * n_broken / len(records):.0f}%).")
    print("\n  VERDICT")
    if med_rot > 5.0 or med_dev > 0.25:
        print("    Boundaries DISAGREE substantially -> the composition across")
        print("    the shared anchor is unreliable.  Alignment is the bug.")
    elif med_rot > 1.0 or med_dev > 0.10:
        print("    Boundaries agree only roughly -> a real but partial")
        print("    contribution from alignment; worth improving, not the whole")
        print("    story.")
    else:
        print("    Boundaries AGREE closely -> the two batches see the same")
        print("    geometry, so the split is not corrupting the measurement.")
        print("    The ATE step from 0->1 boundary is then the cost of losing")
        print("    JOINT estimation, not broken alignment; look elsewhere.")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for seq in [Path(s) for s in args.seq_dir]:
        for submap in sorted(args.submap_sizes):
            records += run_one(seq, submap, args, out_dir)

    path = out_dir / "boundary_records.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"\n  records → {path}")
    summarise(records)


if __name__ == "__main__":
    main()
