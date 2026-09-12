"""
Report the phase-3 runs for fr1/floor and fr1/plant.

Arm F asks whether boundary scale CHAINING rescues fr1/floor.  The shipped
default (`boundary_scale_damping: 1.0`) forces every boundary depth-ratio to
1.0, betting that DA3's metric scale is consistent across a sequence.  Phase 2
showed floor is exactly where that bet loses — its bias is flat across depth
and texture bins but varies frame to frame.  desk is the control: chaining
should HURT it, which is why damping 1.0 shipped.

Arm P asks which gate rejects fr1/plant's loop candidates.  Retrieval is not
the problem (39 genuine revisit pairs sit below the 0.80 threshold, and the
tuned head-to-head already ran at 0.80), so the rejection is downstream.  The
`[LoopClosure]` trace distinguishes the possibilities:

    "REJECTED (confidence=...)"      the DA3 re-inference quality gate
    "rejected by dense gate"         RoMa (off in these runs)
    "ACCEPTED"                       reached the graph, where slam.py's
                                     geometric gate may still hold it

Usage:
    python scripts/report_phase3.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TUM_PREFIX = "rgbd_dataset_freiburg1_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="outputs/phase3")
    p.add_argument("--logs", default="/tmp",
                   help="Where the per-run driver logs were written")
    return p.parse_args()


def load_result(run_dir: Path) -> dict | None:
    hits = list(run_dir.glob(f"{TUM_PREFIX}*/results.json"))
    if not hits:
        return None
    return json.loads(hits[0].read_text())


def gate_counts(log_path: Path) -> dict:
    """Count loop-closure gate outcomes from a run's stdout."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    counts = {
        "candidates": len(re.findall(r"\[LoopClosure\].*?(?:ACCEPTED|REJECTED|"
                                     r"rejected by dense gate)", text)),
        "accepted": text.count("ACCEPTED"),
        "rejected_confidence": len(re.findall(r"REJECTED \(confidence=", text)),
        "rejected_dense": text.count("rejected by dense gate"),
    }
    # The confidence values themselves say whether the gate is marginal or the
    # re-inference is outright degenerate.
    confs = [float(m) for m in
             re.findall(r"conf(?:idence)?=([0-9.]+)", text)]
    if confs:
        confs.sort()
        counts["conf_min"] = confs[0]
        counts["conf_median"] = confs[len(confs) // 2]
        counts["conf_max"] = confs[-1]
    return counts


def main() -> None:
    args = parse_args()
    runs = Path(args.runs)
    logs = Path(args.logs)

    print("=" * 78)
    print("ARM F — does boundary scale CHAINING rescue fr1/floor?")
    print("=" * 78)
    print(f"\n  {'sequence':<8}{'damping':>9}{'Sim3 ATE':>11}{'SE3 ATE':>10}"
          f"{'scale':>9}{'closures':>10}")
    baseline: dict[str, float] = {}
    for seq in ("floor", "desk"):
        for damp in ("1.0", "0.5", "0.0"):
            tag = f"F_{seq}_damp{damp}"
            r = load_result(runs / tag)
            if r is None:
                print(f"  {seq:<8}{damp:>9}{'(missing)':>11}")
                continue
            sim3 = r["ate_sim3"]["rmse"]
            if damp == "1.0":
                baseline[seq] = sim3
            mark = ""
            if seq in baseline and damp != "1.0":
                delta = 100.0 * (sim3 - baseline[seq]) / baseline[seq]
                mark = f"   {delta:+.0f}%"
            print(f"  {seq:<8}{damp:>9}{sim3:>11.4f}"
                  f"{r['ate_se3']['rmse']:>10.4f}"
                  f"{r['ate_sim3']['scale']:>9.3f}"
                  f"{r['n_loop_closures']:>10}{mark}")
        print()
    print("  damping 1.0 = shipped (chaining OFF, trust DA3's metric consistency)")
    print("  PREDICTION: chaining helps floor and hurts desk.  If floor does not")
    print("  improve, the instability is WITHIN submaps, not at their boundaries,")
    print("  and no boundary-level fix can reach it.")

    print("\n" + "=" * 78)
    print("ARM P — which gate rejects fr1/plant's candidates?")
    print("=" * 78)
    print(f"\n  {'run':<20}{'cands':>7}{'accept':>8}{'rej conf':>10}"
          f"{'conf min':>10}{'conf med':>10}{'Sim3 ATE':>11}")
    for conf in ("default", "0.02"):
        for seq in ("plant", "desk"):
            tag = f"P_{seq}_conf{conf}"
            g = gate_counts(logs / f"phase3_{tag}.log")
            r = load_result(runs / tag)
            ate = f"{r['ate_sim3']['rmse']:.4f}" if r else "—"
            if not g:
                print(f"  {tag:<20}{'(no log)':>7}")
                continue
            print(f"  {tag:<20}{g['candidates']:>7}{g['accepted']:>8}"
                  f"{g['rejected_confidence']:>10}"
                  f"{g.get('conf_min', float('nan')):>10.3f}"
                  f"{g.get('conf_median', float('nan')):>10.3f}{ate:>11}")
    print("\n  If plant shows ZERO candidates, retrieval never proposed them "
          "despite\n  39 revisit pairs scoring below 0.80 — the top-K/dedupe "
          "stage is discarding\n  them before the gate.  If candidates appear "
          "but all are rejected on\n  confidence, the wide-baseline re-inference "
          "is degenerate at plant's\n  0.048 parallax, and loosening the gate "
          "should show it (conf 0.02 arm).")


if __name__ == "__main__":
    main()
