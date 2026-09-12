"""
F1 figure for the resolution sweep (plan Step 1): ATE-vs-backbone-latency curve
over resolution, with the map-detail proxy overlaid so map quality can be seen
degrading before (or after) trajectory ATE does.

Consumes a T1 JSON written by the sweep-compilation step:
    {"<res>": [ate_mean, ate_std, backbone_s, peak_mb, map_points, chamfer_m], ...}

Two panels:
  left  — ATE (mean ± repeat-band) vs backbone latency, one point per resolution,
          the knee (min-ATE within the repeat band) circled.
  right — the same x (backbone latency) with two y-lines: ATE and the map-detail
          proxy (Chamfer to the native-res cloud), showing the dissociation.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description="F1: ATE-vs-latency over resolution")
    p.add_argument("--t1", required=True, help="T1 JSON {res: [ate,std,bb,mem,pts,cham]}")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="Resolution sweep")
    p.add_argument("--ate_unit", default="m", help="ATE axis unit label")
    args = p.parse_args()
    ate_lbl = f"ATE Sim3 RMSE ({args.ate_unit})"

    t1 = json.load(open(args.t1))
    res = sorted(int(r) for r in t1)
    ate = [t1[str(r)][0] for r in res]
    std = [t1[str(r)][1] for r in res]
    bb = [t1[str(r)][2] for r in res]
    cham = [t1[str(r)][5] for r in res]
    knee = res[int(min(range(len(res)), key=lambda i: ate[i]))]
    has_map_detail = any(c > 0 for c in cham)

    if not has_map_detail:
        # No map-detail proxy (e.g. UAS km-scale clouds not dumped) → single
        # ATE-vs-latency panel, resolutions annotated so non-monotonic
        # ("bistable") curves are legible.
        fig, axL = plt.subplots(1, 1, figsize=(7, 5.2))
        axL.errorbar(bb, ate, yerr=std, fmt="o-", color="tab:blue", capsize=3,
                     zorder=3)
        for r, x, y in zip(res, bb, ate):
            axL.annotate(f"{r}", (x, y), textcoords="offset points",
                         xytext=(6, 6), fontsize=9)
        ki = res.index(knee)
        axL.scatter([bb[ki]], [ate[ki]], s=380, facecolor="none",
                    edgecolor="red", linewidth=2, zorder=4,
                    label=f"best (res {knee})")
        axL.set_xlabel("backbone latency (s)")
        axL.set_ylabel(ate_lbl)
        axL.set_title("ATE vs backbone latency")
        axL.grid(True, alpha=0.3)
        axL.legend(fontsize=9)
        fig.suptitle(args.title)
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"  F1 → {args.out}  (best res {knee}; no map-detail overlay)")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: ATE vs backbone latency, knee circled.
    axL.errorbar(bb, ate, yerr=std, fmt="o-", color="tab:blue", capsize=3, zorder=3)
    for r, x, y in zip(res, bb, ate):
        axL.annotate(f"{r}", (x, y), textcoords="offset points", xytext=(6, 6),
                     fontsize=9)
    ki = res.index(knee)
    axL.scatter([bb[ki]], [ate[ki]], s=380, facecolor="none", edgecolor="red",
                linewidth=2, zorder=4, label=f"knee (res {knee})")
    axL.set_xlabel("backbone latency (s)")
    axL.set_ylabel(ate_lbl)
    axL.set_title("ATE vs backbone latency")
    axL.grid(True, alpha=0.3)
    axL.legend(fontsize=9)

    # Right: ATE + map-detail (Chamfer) vs backbone latency — the dissociation.
    axR.plot(bb, ate, "o-", color="tab:blue", label="ATE Sim3 (m)", zorder=3)
    axR.set_xlabel("backbone latency (s)")
    axR.set_ylabel(ate_lbl, color="tab:blue")
    axR.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = axR.twinx()
    ax2.plot(bb, cham, "s--", color="tab:orange",
             label="map detail: Chamfer vs native (m)", zorder=3)
    ax2.set_ylabel("Chamfer to native-res cloud (m)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    for r, x, y in zip(res, bb, ate):
        axR.annotate(f"{r}", (x, y), textcoords="offset points", xytext=(6, -12),
                     fontsize=8, color="tab:blue")
    axR.set_title("Trajectory vs map-detail dissociation")
    axR.grid(True, alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"  F1 → {args.out}  (knee at res {knee})")


if __name__ == "__main__":
    main()
